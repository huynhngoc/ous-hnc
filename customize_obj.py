import gc
from itertools import product
import shutil
from deoxys.data.preprocessor import BasePreprocessor
from deoxys_image.patch_sliding import get_patch_indice, get_patches, \
    check_drop
import h5py
from tensorflow import image
from tensorflow.keras.layers import Input, concatenate, Lambda, \
    Add, Activation, Multiply
from tensorflow.keras.models import Model as KerasModel
import numpy as np
import tensorflow as tf
from deoxys.model.callbacks import PredictionCheckpoint
from deoxys.loaders.architecture import BaseModelLoader
from deoxys.experiment import Experiment, SegmentationExperimentPipeline
from deoxys.experiment.postprocessor import DefaultPostProcessor, SegmentationPostProcessor
from deoxys.utils import file_finder, load_json_config
from deoxys.customize import custom_architecture, custom_datareader, custom_layer
from deoxys.loaders import load_data
from deoxys.data.data_reader import HDF5Reader, HDF5DataGenerator, \
    DataReader, DataGenerator, H5PatchGenerator
from deoxys.model.layers import layer_from_config
# from tensorflow.python.ops.gen_math_ops import square
import tensorflow_addons as tfa
from deoxys.model.losses import Loss, loss_from_config
from deoxys.model.metrics import Metric
from deoxys.customize import custom_loss, custom_preprocessor, custom_metric
from deoxys.data import ImageAugmentation2D
from elasticdeform import deform_random_grid
# import new_layer
import os

multi_input_layers = ['Add', 'AddResize', 'Concatenate', 'Multiply', 'Average']
resize_input_layers = ['Concatenate', 'AddResize']


@custom_layer
class InstanceNormalization(tfa.layers.InstanceNormalization):
    pass


@custom_layer
class AddResize(Add):
    pass


# new loss function
# dice all GTV
# dice GTVp and GTVn
# penalize if overlap GTVp and GTVn
# cases no GTVp or no GTVn

@custom_loss
class MultiLabelDiceLoss(Loss):
    def __init__(self, reduction='auto', name="combine_dice", beta=1):
        super().__init__(reduction, name)

        self.beta = beta

    def call(self, target, prediction):
        size = len(prediction.get_shape().as_list())
        reduce_ax = list(range(1, size-1))
        eps = 1e-8

        true_positive = tf.reduce_sum(prediction * target, axis=reduce_ax)
        target_positive = tf.reduce_sum(tf.square(target), axis=reduce_ax)
        predicted_positive = tf.reduce_sum(
            tf.square(prediction), axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )

        return tf.reduce_mean(1 - fb_numerator / fb_denominator, axis=-1)


@custom_loss
class MultiLabelDiceLossNoOverlap(Loss):
    def __init__(self, reduction='auto', name="combine_dice_no_overlap",
                 beta=1, overlap_penalty=0.5):
        super().__init__(reduction, name)

        self.beta = beta
        self.overlap_penalty = overlap_penalty

    def call(self, target, prediction):
        size = len(prediction.get_shape().as_list())
        reduce_ax = list(range(1, size-1))
        eps = 1e-8

        true_positive = tf.reduce_sum(prediction * target, axis=reduce_ax)
        target_positive = tf.reduce_sum(target, axis=reduce_ax)
        predicted_positive = tf.reduce_sum(
            tf.square(prediction), axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )

        dice_loss = 1 - fb_numerator / fb_denominator

        if prediction.shape[-1] >= 2:
            # Calculate IoU between channel 0 and channel 1
            pred_ch0 = prediction[..., 0] > 0.5
            pred_ch1 = prediction[..., 1] > 0.5
            intersection = tf.reduce_sum(pred_ch0 & pred_ch1, axis=reduce_ax)  # shape: (batch,)
            union = tf.reduce_sum(pred_ch0 | pred_ch1, axis=reduce_ax)  # shape: (batch,)
            iou = intersection / (union + eps)
            # Add penalty to loss (broadcast to match dice_loss shape if needed)
            dice_loss = dice_loss + self.overlap_penalty * tf.expand_dims(iou, axis=-1)

        return tf.reduce_mean(dice_loss, axis=-1)  # shape: (batch,)


@custom_loss
class BinaryMacroFbetaLoss(Loss):
    def __init__(self, reduction='auto', name="binary_macro_fbeta",
                 beta=1, square=False):
        super().__init__(reduction, name)

        self.beta = beta
        self.square = square

    def call(self, target, prediction):
        eps = 1e-8
        target = tf.cast(target, prediction.dtype)

        true_positive = tf.math.reduce_sum(prediction * target)
        if self.square:
            target_positive = tf.math.reduce_sum(tf.math.square(target))
            predicted_positive = tf.math.reduce_sum(
                tf.math.square(prediction))
        else:
            target_positive = tf.math.reduce_sum(target)
            predicted_positive = tf.math.reduce_sum(prediction)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )

        return 1 - fb_numerator / fb_denominator


@custom_loss
class FusedLoss(Loss):
    """Used to sum two or more loss functions.
    """

    def __init__(
            self, loss_configs, loss_weights=None,
            reduction="auto", name="fused_loss"):
        super().__init__(reduction, name)
        self.losses = [loss_from_config(loss_config)
                       for loss_config in loss_configs]

        if loss_weights is None:
            loss_weights = [1] * len(self.losses)
        self.loss_weights = loss_weights

    def call(self, target, prediction):
        loss = None
        for loss_class, loss_weight in zip(self.losses, self.loss_weights):
            if loss is None:
                loss = loss_weight * loss_class(target, prediction)
            else:
                loss += loss_weight * loss_class(target, prediction)

        return loss


class EnsemblePostProcessor(DefaultPostProcessor):
    def __init__(self, log_base_path='logs',
                 log_path_list=None,
                 map_meta_data=None, **kwargs):

        self.log_base_path = log_base_path
        self.log_path_list = []
        for path in log_path_list:
            merge_file = path + self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            if os.path.exists(merge_file):
                self.log_path_list.append(merge_file)
            else:
                print('Missing file from', path)

        # check if there are more than 1 to ensemble
        assert len(self.log_path_list) > 1, 'Cannot ensemble with 0 or 1 item'

        if map_meta_data:
            if type(map_meta_data) == str:
                self.map_meta_data = map_meta_data.split(',')
            else:
                self.map_meta_data = map_meta_data
        else:
            self.map_meta_data = ['patient_idx']

        # always run test
        self.run_test = True

    def ensemble_results(self):
        # initialize the folder
        if not os.path.exists(self.log_base_path):
            print('Creating output folder')
            os.makedirs(self.log_base_path)

        output_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        if not os.path.exists(output_folder):
            print('Creating ensemble folder')
            os.makedirs(output_folder)

        output_file = output_folder + self.PREDICT_TEST_NAME
        if not os.path.exists(output_file):
            print('Copying template for output file')
            shutil.copy(self.log_path_list[0], output_folder)

        print('Creating ensemble results...')
        y_preds = []
        for file in self.log_path_list:
            with h5py.File(file, 'r') as hf:
                y_preds.append(hf['predicted'][:])

        with h5py.File(output_file, 'a') as mf:
            mf['predicted'][:] = np.mean(y_preds, axis=0)
        print('Ensembled results saved to file')

        return self

    def concat_results(self):
        # initialize the folder
        if not os.path.exists(self.log_base_path):
            print('Creating output folder')
            os.makedirs(self.log_base_path)

        output_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        if not os.path.exists(output_folder):
            print('Creating ensemble folder')
            os.makedirs(output_folder)

        # first check the template
        with h5py.File(self.log_path_list[0], 'r') as f:
            ds_names = list(f.keys())
        ds = {name: [] for name in ds_names}

        # get the data
        for file in self.log_path_list:
            with h5py.File(file, 'r') as hf:
                for key in ds:
                    ds[key].append(hf[key][:])

        # now merge them
        print('creating merged file')
        output_file = output_folder + self.PREDICT_TEST_NAME
        with h5py.File(output_file, 'w') as mf:
            for key, val in ds.items():
                mf.create_dataset(key, data=np.concatenate(val, axis=0))


@custom_preprocessor
class CombineMaskPreprocessor(BasePreprocessor):
    def __init__(self, operator='or', mask_channels=None):
        if mask_channels is None:
            self.mask_channels = [0, 1]
        elif '__iter__' not in dir(mask_channels):
            self.mask_channels = [mask_channels]
        else:
            self.mask_channels = mask_channels
        self.operator = operator

    def transform(self, inputs, target=None):
        if target is None:
            return inputs, None
        if len(self.mask_channels) == 1:
            return inputs, target[..., self.mask_channels[0]]
        else:
            if self.operator == 'or':
                return inputs, np.max(target[..., self.mask_channels], axis=-1)
            elif self.operator == 'and':
                return inputs, np.min(target[..., self.mask_channels], axis=-1)


@custom_preprocessor
class ZScoreDensePreprocessor(BasePreprocessor):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def transform(self, inputs, target=None):
        mean, std = self.mean, self.std
        if mean is None:
            mean = inputs.mean(axis=0)
            std = inputs.std(axis=0)
        else:
            mean = np.array(mean)
            std = np.array(std)
        std[std == 0] = 1

        return (inputs - mean)/std, target


@custom_preprocessor
class ChannelRepeater(BasePreprocessor):
    def __init__(self, channel=0):
        if '__iter__' not in dir(channel):
            self.channel = [channel]
        else:
            self.channel = channel

    def transform(self, images, targets):
        return np.concatenate([images, images[..., self.channel]], axis=-1), targets


@custom_preprocessor
class DynamicWindowing(BasePreprocessor):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95, channel=0):
        self.lower_quantile, self.upper_quantile = lower_quantile, upper_quantile
        self.channel = channel

    def perform_windowing(self, image):
        axis = list(np.arange(len(image.shape)))[1:]
        lower = np.quantile(image, self.lower_quantile, axis=axis)
        upper = np.quantile(image, self.upper_quantile, axis=axis)
        for i in range(len(image)):
            image[i][image[i] < lower[i]] = lower[i]
            image[i][image[i] > upper[i]] = upper[i]
        return image

    def transform(self, images, targets):
        images = images.copy()
        images[..., self.channel] = self.perform_windowing(
            images[..., self.channel])
        return images, targets


@custom_preprocessor
class ElasticDeform(BasePreprocessor):
    def __init__(self, sigma=4, points=3):
        self.sigma = sigma
        self.points = points

    def transform(self, x, y):
        return deform_random_grid([x, y], axis=[(1, 2, 3), (1, 2, 3)],
                                  sigma=self.sigma, points=self.points)


@custom_metric
class CombineDice(Metric):
    def __init__(self, threshold=None, name='dice', dtype=None, beta=1, channel=0):
        super().__init__(name=name, dtype=dtype)

        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta
        self.channel = channel

        self.total = self.add_weight(
            'total', initializer='zeros')
        self.count = self.add_weight(
            'count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        size = len(y_pred.get_shape().as_list())
        reduce_ax = list(range(1, size))
        eps = 1e-8

        y_true = tf.cast(y_true[..., [self.channel]], y_pred.dtype)
        y_pred = tf.cast(y_pred[..., [self.channel]] > self.threshold, y_true.dtype)

        true_positive = tf.reduce_sum(y_pred * y_true, axis=reduce_ax)
        target_positive = tf.reduce_sum(y_true, axis=reduce_ax)
        predicted_positive = tf.reduce_sum(y_pred, axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )
        if sample_weight:
            weight = tf.cast(sample_weight, self.dtype)
            self.total.assign_add(
                tf.reduce_sum(weight * fb_numerator / fb_denominator)
            )
        else:
            self.total.assign_add(
                tf.reduce_sum(fb_numerator / fb_denominator)
            )

        count = tf.reduce_sum(weight) if sample_weight else tf.cast(
            tf.shape(y_pred)[0], y_pred.dtype)

        self.count.assign_add(count)

    def result(self):
        return self.total / self.count

    def get_config(self):
        config = {'threshold': self.threshold,
                  'beta': self.beta,
                  'channel': self.channel}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

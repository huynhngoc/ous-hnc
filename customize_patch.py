import gc
from itertools import product
import shutil
from deoxys.data.preprocessor import BasePreprocessor
from deoxys_image.patch_sliding import get_patch_indice, get_patches, \
    check_drop
import h5py
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
from deoxys.customize import custom_loss, custom_preprocessor
from elasticdeform import deform_random_grid
# import new_layer
import os


@custom_datareader
class H5PatchReaderV2(DataReader):
    @property
    def train_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for training
        """
        return H5PatchGeneratorRandom(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.train_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=self.shuffle,
            augmentations=self.augmentations,
            preprocess_first=self.preprocess_first,
            drop_fraction=self.drop_fraction,
            check_drop_channel=self.check_drop_channel,
            bounding_box=self.bounding_box)

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return H5PatchGeneratorSliding(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=False, preprocess_first=self.preprocess_first,
            drop_fraction=0)

    @property
    def val_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for validation
        """
        return H5PatchGeneratorSliding(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=False, preprocess_first=self.preprocess_first,
            drop_fraction=0)


class H5PatchGeneratorRandom(H5PatchGenerator):
    @property
    def total_batch(self):
        """Total number of batches to iterate all data.
        It will be used as the number of steps per epochs when training or
        validating data in a model.

        Returns
        -------
        int
            Total number of batches to iterate all data
        """
        print('counting total iter')
        if self._total_batch is None:
            total_batch = 0
            fold_names = self.folds

            if self.drop_fraction == 0:
                # just calculate based on the size of each fold
                for fold_name in fold_names:
                    with h5py.File(self.h5_filename, 'r') as hf:
                        shape = hf[fold_name][self.y_name].shape[:-1]
                    indices = get_patch_indice(
                        shape[1:], self.patch_size, self.overlap)
                    patches_per_img = len(indices)
                    patches_per_cache = patches_per_img * self.batch_cache

                    num_cache = shape[0] // self.batch_cache
                    remainder_img = shape[0] % self.batch_cache

                    batch_per_cache = np.ceil(
                        patches_per_cache / self.batch_size)

                    total_batch += num_cache * batch_per_cache

                    total_batch += np.ceil(
                        remainder_img * patches_per_img / self.batch_size)

            else:
                # CHANGES: just calculate total number of images
                total_batch = 0
                for fold_name in fold_names:
                    print(fold_name)
                    with h5py.File(self.h5_filename, 'r') as hf:
                        num_image = hf[fold_name][self.y_name].shape[0]

                        total_batch += np.ceil(
                            np.sum(num_image) / self.batch_size)

        self._total_batch = int(total_batch)
        print('done counting iter_num', self._total_batch)
        if self.augmentations:
            print('number of iters may be larger than '
                  'number of iters to go through all images in this set')
        return self._total_batch

    def next_seg(self):
        gc.collect()
        if self.seg_idx == len(self.seg_list):
            # move to next fold
            self.next_fold()

            # reset seg index
            self.seg_idx = 0
            # recalculate seg_num
            cur_fold = self.folds[self.fold_idx]
            with h5py.File(self.h5_filename, 'r') as hf:
                seg_num = np.ceil(
                    hf[cur_fold][self.y_name].shape[0] / self.batch_cache)
                self.fold_shape = hf[self.folds[0]][self.y_name].shape[1:-1]

            self.seg_list = np.arange(seg_num).astype(int)

            if self.shuffle:
                np.random.shuffle(self.seg_list)

        cur_fold = self.folds[self.fold_idx]
        cur_seg_idx = self.seg_list[self.seg_idx]

        start, end = cur_seg_idx * \
            self.batch_cache, (cur_seg_idx + 1) * self.batch_cache

        # print(cur_fold, cur_seg_idx, start, end)

        with h5py.File(self.h5_filename, 'r') as hf:
            seg_x_raw = hf[cur_fold][self.x_name][start: end]
            seg_y_raw = hf[cur_fold][self.y_name][start: end]

        # indices = get_patch_indice(
        #     self.fold_shape, self.patch_size, self.overlap)

        # if preprocess first, apply preprocess here
        if self.preprocessors and self.preprocess_first:
            seg_x_raw, seg_y_raw = self._apply_preprocess(seg_x_raw, seg_y_raw)
        # CHANGES: get patches
        seg_x, seg_y = get_patches_random(
            seg_x_raw, seg_y_raw,
            patch_size=self.patch_size,
            drop_fraction=self.drop_fraction,
            check_drop_channel=self.check_drop_channel)

        # if preprocess after patch, apply preprocess here
        if self.preprocessors and not self.preprocess_first:
            seg_x, seg_y = self._apply_preprocess(seg_x, seg_y)

        # increase seg index
        self.seg_idx += 1
        return seg_x, seg_y


class H5PatchGeneratorSliding(DataGenerator):
    def next_seg(self):
        gc.collect()
        if self.seg_idx == len(self.seg_list):
            # move to next fold
            self.next_fold()

            # reset seg index
            self.seg_idx = 0
            # recalculate seg_num
            cur_fold = self.folds[self.fold_idx]
            with h5py.File(self.h5_filename, 'r') as hf:
                seg_num = np.ceil(
                    hf[cur_fold][self.y_name].shape[0] / self.batch_cache)
                self.fold_shape = hf[self.folds[0]][self.y_name].shape[1:-1]

            self.seg_list = np.arange(seg_num).astype(int)

            if self.shuffle:
                np.random.shuffle(self.seg_list)

        cur_fold = self.folds[self.fold_idx]
        cur_seg_idx = self.seg_list[self.seg_idx]

        start, end = cur_seg_idx * \
            self.batch_cache, (cur_seg_idx + 1) * self.batch_cache

        # print(cur_fold, cur_seg_idx, start, end)

        with h5py.File(self.h5_filename, 'r') as hf:
            seg_x_raw = hf[cur_fold][self.x_name][start: end]
            seg_y_raw = hf[cur_fold][self.y_name][start: end]

        indices = get_patch_indice(
            self.fold_shape, self.patch_size, self.overlap)

        # if preprocess first, apply preprocess here
        if self.preprocessors and self.preprocess_first:
            seg_x_raw, seg_y_raw = self._apply_preprocess(seg_x_raw, seg_y_raw)
        # CHANGES: get patches
        seg_x, seg_y = get_patches_sliding(
            seg_x_raw, seg_y_raw,
            patch_indice=indices, patch_size=self.patch_size)

        # if preprocess after patch, apply preprocess here
        if self.preprocessors and not self.preprocess_first:
            seg_x, seg_y = self._apply_preprocess(seg_x, seg_y)

        # finally apply augmentation, if any
        # if self.augmentations:
        #     total = len(seg_y)
        #     seg_x, seg_y = self._apply_augmentation(
        #         seg_x[:total], seg_y[:total])

        # increase seg index
        self.seg_idx += 1
        return seg_x, seg_y


def get_patches_random(images, target=None, patch_size=None,
                       drop_fraction=0.1, check_drop_channel=None):
    try:
        images = np.array(images)
    except Exception:  # compatity check  # pragma: no cover
        images = np.array(images, dtype=object)

    if images.dtype == object:  # pragma: no cover
        raise NotImplementedError(
            'Cannot handle images with different sizes.'
            'Consider using get_patches_different_images')

    if target is not None:
        try:
            target = np.array(target)
        except Exception:  # compatity check  # pragma: no cover
            target = np.array(target, dtype=object)
        if not images.shape[1:-1] == target.shape[1:-1]:
            raise ValueError(
                'Image and target shape are mismatched ({} != {})'.format(
                    images.shape[:-1], target.shape[:-1]))

    # total volume/area of one patch of image
    pixel_sum = np.prod(patch_size)

    patch_img = np.zeros((len(images), *patch_size, images.shape[-1]))

    if target is not None:
        patch_label = np.zeros((len(images), *patch_size, target.shape[-1]))

    threshold = drop_fraction * pixel_sum
    m, n, p = patch_size
    x, y, z = images.shape[1:-1]
    count = 0
    for i in range(len(images)):
        check_data = target[i][..., check_drop_channel]
        for _ in range(30):
            start_x = np.random.randint(0, x - m + 1)
            start_y = np.random.randint(0, y - n + 1)
            start_z = np.random.randint(0, z - p + 1)

            patch_target = check_data[start_x:start_x + m,
                                      start_y:start_y + n,
                                      start_z:start_z + p, :]
            if np.sum(patch_target.sum(axis=-1) > 0) > threshold:
                patch_label[count] = patch_target
                patch_img[count] = images[i, start_x:start_x + m,
                                          start_y:start_y + n,
                                          start_z:start_z + p, :]
                count += 1
                break

    # Trim unused slots if some images failed to produce valid patches
    patch_img = patch_img[:count]
    patch_label = patch_label[:count]

    # Shuffle the patches
    indice = np.arange(count)
    np.random.shuffle(indice)

    return patch_img[indice], patch_label[indice]


def get_patches_sliding(images, target=None, patch_indice=None,
                        patch_size=None):
    try:
        images = np.array(images)
    except Exception:  # compatity check  # pragma: no cover
        images = np.array(images, dtype=object)

    if images.dtype == object:  # pragma: no cover
        raise NotImplementedError(
            'Cannot handle images with different sizes.'
            'Consider using get_patches_different_images')

    if target is not None:
        try:
            target = np.array(target)
        except Exception:  # compatity check  # pragma: no cover
            target = np.array(target, dtype=object)
        if not images.shape[1:-1] == target.shape[1:-1]:
            raise ValueError(
                'Image and target shape are mismatched ({} != {})'.format(
                    images.shape[:-1], target.shape[:-1]))

    if patch_indice is None or patch_size is None:
        raise ValueError('patch_indice and patch_size are required.')

    # create indice for each images
    # [(0, (x,y,z)), (0, (x',y',z')), (1, (x,y,z)), (1, (x',y',z'))]

    patch_indice = np.array(
        list((product(np.arange(len(images)), patch_indice))), dtype=object)

    patch_img = np.zeros((len(patch_indice), *patch_size, images.shape[-1]))

    if target is not None:
        patch_label = np.zeros((len(patch_indice), *patch_size,
                                target.shape[-1]))
    for i, (im_i, indice) in enumerate(patch_indice):
        if len(indice) == 2:
            x, y = indice
            w, h = patch_size
            patch_img[i] = images[im_i][x:x+w, y:y+h]

            if target is not None:
                patch_label[i] = target[im_i][x:x+w, y:y+h]

        elif len(indice) == 3:
            x, y, z = indice
            w, h, d = patch_size

            patch_img[i] = images[im_i][x:x+w, y:y+h, z:z+d]
            if target is not None:
                patch_label[i] = target[im_i][x:x+w, y:y+h, z:z+d]

    if target is None:
        return patch_img
    else:
        return patch_img, patch_label


class H5MergePatchesV2:  # pragma: no cover
    def _merge_patches_to_merge_file(self, meta, start_cursor):
        with h5py.File(self.merge_file, 'r') as mf:
            shape = mf[self.target][meta].shape[:-1]

        # fix patch size
        if '__iter__' not in dir(self.patch_size):
            self.patch_size = [self.patch_size] * len(shape)

        indice = get_patch_indice(shape, self.patch_size, self.overlap)
        next_cursor = start_cursor + len(indice)

        with h5py.File(self.predicted_file, 'r') as f:
            data = f[self.predicted][start_cursor: next_cursor]

        # merge per channel
        predicted_data = np.zeros(data.shape[1:])
        for c in range(data.shape[-1]):
            predicted = np.zeros(shape)
            weight = np.zeros(shape)

            for i in range(len(indice)):
                x, y, z = indice[i]
                w, h, d = self.patch_size
                predicted[x:x+w, y:y+h, z:z+d] = predicted[x:x+w, y:y+h, z:z+d] \
                    + data[i][..., 0]
                weight[x:x+w, y:y+h, z:z+d] = weight[x:x+w, y:y+h, z:z+d] \
                    + np.ones(self.patch_size)

            predicted_data[..., c] = (predicted/weight)

        with h5py.File(self.merge_file, 'a') as mf:
            mf[self.predicted].create_dataset(
                meta, data=predicted_data, compression="gzip")

        return next_cursor


class MultiLabelSegmentationPostProcessor(SegmentationPostProcessor):
    # may need to rewite calculate metrics
    def merge_3d_patches(self):  # pragma: no cover
        print('merge 3d patches to 3d images')
        if not self.run_test:
            predicted_path = self.temp_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME
            # map_folder = self.log_base_path + self.SINGLE_MAP_PATH
            # map_filename = map_folder + self.SINGLE_MAP_NAME

            merge_path = self.analysis_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME

            main_log_folder = self.log_base_path + self.MAP_PATH

            if not os.path.exists(main_log_folder):
                os.makedirs(main_log_folder)
            main_log_filename = main_log_folder + self.MAP_NAME

            for epoch in self.epochs:
                H5MergePatchesV2(
                    ref_file=self.dataset_filename,
                    predicted_file=predicted_path.format(epoch=epoch),
                    map_column=self.main_meta_data,
                    merge_file=merge_path.format(epoch=epoch),
                    save_file=main_log_filename.format(epoch=epoch),
                    patch_size=self.data_reader.patch_size,
                    overlap=self.data_reader.overlap,
                    folds=self.data_reader.val_folds,
                    fold_prefix='',
                    original_input_dataset=self.data_reader.x_name,
                    original_target_dataset=self.data_reader.y_name,
                ).post_process()
        else:
            predicted_path = self.temp_base_path + \
                self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            merge_path = test_folder + self.PREDICT_TEST_NAME
            main_result_file_name = test_folder + self.TEST_MAP_NAME

            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            H5MergePatchesV2(
                ref_file=self.dataset_filename,
                predicted_file=predicted_path,
                map_column=self.main_meta_data,
                merge_file=merge_path,
                save_file=main_result_file_name,
                patch_size=self.data_reader.patch_size,
                overlap=self.data_reader.overlap,
                folds=self.data_reader.test_folds,
                fold_prefix='',
                original_input_dataset=self.data_reader.x_name,
                original_target_dataset=self.data_reader.y_name,
            ).post_process()

        return self


class SegmentationExperimentPipelineV2(SegmentationExperimentPipeline):
    DEFAULT_PP = SegmentationPostProcessorV2

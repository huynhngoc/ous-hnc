"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

# import h5py
# from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from deoxys.experiment import Experiment, ExperimentPipeline, SegmentationExperimentPipeline
from deoxys.model.callbacks import PredictionCheckpoint
from deoxys.model import load_model
from deoxys_image.patch_sliding import get_patches
# from deoxys.utils import read_file
import argparse
# import os
import numpy as np
import pandas as pd
import surface_distance.metrics as metrics
import h5py
# from pathlib import Path
# fmt: off
import sys
sys.path.append('.')
import customize_obj
import customize_postprocessor
# fmt: on


def dice_score(y_true, y_pred, channel=[0, 1], postfix='GTVall'):
    threshold = 0.5
    eps = 1e-8
    if '__iter__' in dir(channel):
        y_true = np.max(y_true[..., channel], axis=-1)
        y_pred = (y_pred > threshold).astype(y_pred.dtype)
        y_pred = np.max(y_pred[..., channel], axis=-1)
    else:
        y_pred = (y_pred > threshold).astype(y_pred.dtype)
        y_true = y_true[..., channel]
        y_pred = y_pred[..., channel]

    true_positive = np.sum(y_true * y_pred)
    target_positive = np.sum(y_true)
    predicted_positive = np.sum(y_pred)

    dice_score = (2. * true_positive + eps) / (target_positive + predicted_positive + eps)
    precision = (true_positive + eps) / (predicted_positive + eps)
    recall = (true_positive + eps) / (target_positive + eps)
    return {
        f'dice_{postfix}': dice_score,
        f'precision_{postfix}': precision,
        f'recall_{postfix}': recall
    }


def distance_metrics(y_true, y_pred, channel=[0, 1], postfix='_GTVall'):
    threshold = 0.5
    if '__iter__' in dir(channel):
        y_true = np.max(y_true[..., channel], axis=-1)
        y_pred = (y_pred > threshold).astype(y_pred.dtype)
        y_pred = np.max(y_pred[..., channel], axis=-1)
    else:
        y_pred = (y_pred > threshold).astype(y_pred.dtype)
        y_true = y_true[..., channel]
        y_pred = y_pred[..., channel]

    output = {}
    surface_distances = metrics.compute_surface_distances(
        y_true > 0, y_pred > threshold, (1, 1, 1))
    for tolerance in [1, 2, 3]:
        output[f'surface_distance_{tolerance}mm_{postfix}'] = metrics.compute_surface_dice_at_tolerance(
            surface_distances, tolerance)

    output[f'HD_{postfix}'] = metrics.compute_robust_hausdorff(
        surface_distances, 100)
    output[f'HD95_{postfix}'] = metrics.compute_robust_hausdorff(
        surface_distances, 95)
    asd = metrics.compute_average_surface_distance(surface_distances)
    output[f'ASD_{postfix}[0]'] = asd[0]
    output[f'ASD_{postfix}[1]'] = asd[1]
    return output


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--analysis_folder",
                        default='', type=str)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--model_checkpoint_period", default=2, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=2, type=int)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument("--monitor", default='f1_score', type=str)
    parser.add_argument("--memory_limit", default=0, type=int)

    args, unknown = parser.parse_known_args()

    if args.memory_limit:
        # Restrict TensorFlow to only allocate X-GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=1024 * args.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    analysis_folder = ''

    meta = args.meta
    print('Log files: ', args.log_folder)
    print('Unprocessed predictions are saved to', args.temp_folder)
    if analysis_folder:
        print('Intermediate processed files for merging patches are saved to',
              analysis_folder)

    PredictionCheckpoint._max_size = 0.5

    model = load_model(args.log_folder + f'/model/model.{args.epochs:03d}.h5')
    prediction_file = args.temp_folder + f'/prediction/prediction.{args.epochs:03d}.h5'
    with h5py.File('../datasets/all_hnc_3d.h5', 'r') as f:
        patient_idx = f['fold_3'][args.meta][:]

    all_results = []
    for i, pid in enumerate(patient_idx):
        print(f'Patient {i}: {pid}')
        with h5py.File(prediction_file, 'r') as f:
            image = f['x'][i:i+1]
            y_true = f['y'][i]
            y_pred = f['predicted'][i]

        # get bounding box of predicted
        # find the new middle point
        position_indice = np.where(y_pred > 0.5)
        # ax0_lower, ax0_upper = position_indice[0].min(), position_indice[0].max()
        ax1_lower, ax1_upper = position_indice[1].min(), position_indice[1].max()
        ax2_lower, ax2_upper = position_indice[2].min(), position_indice[2].max()
        middle_point = np.array([176 // 2, (ax1_upper +
                                            ax1_lower) // 2, (ax2_upper + ax2_lower) // 2]).astype(int)
        middle_origin = middle_point - np.array([176, 144, 128])//2
        middle_origin.clip(0, [0, 192 - 144, 256 - 128])
        print(ax2_lower, ax2_upper, ax1_lower, ax1_upper)
        print('middle point', middle_point, middle_origin)

        range_x, step_x = 1, 1
        range_y, step_y = 5, 5
        range_z, step_z = 5, 5
        indice = [(x, y, z) for x in [0]
                  for y in range(middle_origin[1]-range_y, middle_origin[1]+range_y+1, step_y)
                  for z in range(middle_origin[2]-range_z, middle_origin[2]+range_z+1, step_z)]
        patches = get_patches(image, patch_indice=indice, patch_size=[
            176, 144, 128], stratified=False, drop_fraction=0)
        print(patches.shape, indice)
        print('Preprocessing finished. Running model prediction')

        mask_list = []
        for item in range(0, len(indice), 2):
            mask_list.append(model.model.predict(patches[item: item+2], None))
            print(f'Predicted {item+2}/{len(indice)}')
        mask_list = np.concatenate(mask_list)
        print('Model prediction finished! Post-processing...')
        weight = np.zeros([*image.shape[1:-1], 2])
        predicted = np.zeros([*image.shape[1:-1], 2])

        for i in range(len(indice)):
            x, y, z = indice[i]
            w, h, d = [176, 144, 128]
            if np.any(mask_list[i].shape == 0):
                print('skipped', indice[i])
                continue
            predicted[x: x+w, y: y+h, z: z+d] = predicted[x: x+w, y: y+h, z: z+d] + mask_list[i]
            weight[x: x+w, y: y+h, z: z+d] = weight[x: x+w,
                                                    y: y+h, z: z+d] + np.ones([176, 144, 128, 2])

        predicted = (predicted / (weight))
        print('Post-processing finished! Calculating metrics...')
        output = {'patient_idx': pid}
        output.update(dice_score(y_true, predicted, channel=0, postfix='GTVp'))
        output.update(dice_score(y_true, predicted, channel=1, postfix='GTVn'))
        output.update(dice_score(y_true, predicted, channel=[0, 1], postfix='GTVall'))
        output.update(distance_metrics(y_true, predicted, channel=0, postfix='GTVp'))
        output.update(distance_metrics(y_true, predicted, channel=1, postfix='GTVn'))
        output.update(distance_metrics(y_true, predicted, channel=[0, 1], postfix='GTVall'))
        print(output)
        all_results.append(output)
        print('\n***To excel****')
        print(','.join([
            str(pid),
            f"{output['dice_GTVp']:.4f}",
            f"{output['dice_GTVn']:.4f}",
            f"{output['dice_GTVall']:.4f}",
            f"{output['surface_distance_1mm_GTVp']:.4f}",
            f"{output['surface_distance_1mm_GTVn']:.4f}",
            f"{output['surface_distance_1mm_GTVall']:.4f}",
            f"{output['HD95_GTVp']:.1f}",
            f"{output['HD95_GTVn']:.1f}",
            f"{output['HD95_GTVall']:.1f}",
            f"{output['ASD_GTVp[0]']:.2f}",
            f"{output['ASD_GTVn[0]']:.2f}",
            f"{output['ASD_GTVall[0]']:.2f}",
            f"{output['ASD_GTVp[1]']:.2f}",
            f"{output['ASD_GTVn[1]']:.2f}",
            f"{output['ASD_GTVall[1]']:.2f}",
        ]))
        print('************\n')

    df = pd.DataFrame(all_results)
    df.to_csv(args.log_folder + f'/results.{args.epochs:03d}.csv', index=False)

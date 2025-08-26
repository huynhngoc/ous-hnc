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
# from deoxys.utils import read_file
import argparse
# import os
# import numpy as np
# from pathlib import Path
# fmt: off
import sys
sys.path.append('.')
import customize_obj
import customize_postprocessor
# fmt: on

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
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
    print('training from configuration', args.config_file,
          'and saving log files to', args.log_folder)
    print('Unprocesssed prediciton are saved to', args.temp_folder)
    if analysis_folder:
        print('Intermediate processed files for merging patches are saved to',
              analysis_folder)

    PredictionCheckpoint._max_size = 0.5

    exp = SegmentationExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).from_full_config(
        args.config_file
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=10,
        prediction_checkpoint_period=50,
        epochs=50,
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=args.model_checkpoint_period,
        prediction_checkpoint_period=args.prediction_checkpoint_period,
        epochs=args.epochs,
        initial_epoch=50,
    ).apply_post_processors(
        recipe='3d',
        post_processor_class=customize_postprocessor.MultiLabelSegmentationPostProcessor,
        analysis_base_path=analysis_folder,
        map_meta_data=meta,
        metrics=['f1_score', 'f1_score', 'f1_score',
                 'precision', 'precision', 'precision',
                 'recall', 'recall', 'recall',
                 'surface_dice', 'surface_dice', 'surface_dice',
                 'surface_dice', 'surface_dice', 'surface_dice',
                 'surface_dice', 'surface_dice', 'surface_dice',
                 'HD', 'HD', 'HD',
                 'HD', 'HD', 'HD',
                 'ASD', 'ASD', 'ASD',
                 'ASD', 'ASD', 'ASD'
                 ],
        metrics_kwargs=[
            {'metric_name': 'dice_GTVp', 'channel': 0},
            {'metric_name': 'dice_GTVn', 'channel': 1},
            {'metric_name': 'dice_GTVall', 'channel': [0, 1]},
            {'metric_name': 'precision_GTVp', 'channel': 0},
            {'metric_name': 'precision_GTVn', 'channel': 1},
            {'metric_name': 'precision_GTVall', 'channel': [0, 1]},
            {'metric_name': 'recall_GTVp', 'channel': 0},
            {'metric_name': 'recall_GTVn', 'channel': 1},
            {'metric_name': 'recall_GTVall', 'channel': [0, 1]},
            {'metric_name': 'surface_dice_1mm_GTVp', 'channel': 0},
            {'metric_name': 'surface_dice_1mm_GTVn', 'channel': 1},
            {'metric_name': 'surface_dice_1mm_GTVall', 'channel': [0, 1]},
            {'metric_name': 'surface_dice_2mm_GTVp', 'channel': 0, 'tolerance': 2.0},
            {'metric_name': 'surface_dice_2mm_GTVn', 'channel': 1, 'tolerance': 2.0},
            {'metric_name': 'surface_dice_2mm_GTVall', 'channel': [0, 1], 'tolerance': 2.0},
            {'metric_name': 'surface_dice_3mm_GTVp', 'channel': 0, 'tolerance': 3.0},
            {'metric_name': 'surface_dice_3mm_GTVn', 'channel': 1, 'tolerance': 3.0},
            {'metric_name': 'surface_dice_3mm_GTVall', 'channel': [0, 1], 'tolerance': 3.0},
            {'metric_name': 'HD_GTVp', 'channel': 0},
            {'metric_name': 'HD_GTVn', 'channel': 1},
            {'metric_name': 'HD_GTVall', 'channel': [0, 1]},
            {'metric_name': 'HD95_GTVp', 'channel': 0, 'percentile': 95},
            {'metric_name': 'HD95_GTVn', 'channel': 1, 'percentile': 95},
            {'metric_name': 'HD95_GTVall', 'channel': [0, 1], 'percentile': 95},
            {'metric_name': 'ASD_GTVp[0]', 'channel': 0},
            {'metric_name': 'ASD_GTVn[0]', 'channel': 1},
            {'metric_name': 'ASD_GTVall[0]', 'channel': [0, 1]},
            {'metric_name': 'ASD_GTVp[1]', 'channel': 0, 'index': 1},
            {'metric_name': 'ASD_GTVn[1]', 'channel': 1, 'index': 1},
            {'metric_name': 'ASD_GTVall[1]', 'channel': [0, 1], 'index': 1}
        ]
    ).plot_performance().load_best_model(
        monitor='surface_dice_1mm_GTVall', keep_best_only=False)

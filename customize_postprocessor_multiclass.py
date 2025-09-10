from deoxys.experiment.postprocessor import SegmentationPostProcessor, \
    H5CalculateFScore, H5CalculateRecall, H5CalculatePrecision, H5CalculateFPR
from deoxys.experiment.pipeline import SegmentationExperimentPipeline
import os
from time import time
import numpy as np
import surface_distance.metrics as metrics


class MultiLabelH5CalculateFScore(H5CalculateFScore):
    def __init__(self, ref_file, save_file, metric_name='',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4, beta=1, threshold=None,
                 map_file=None, map_column=None, channel=0):
        super().__init__(ref_file, save_file,
                         metric_name or self.__class__.name,
                         predicted_dataset,
                         target_dataset, batch_size,
                         map_file, map_column)
        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta
        self.channel = channel

    def calculate_metrics(self, y_true, y_pred, **kwargs):
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        if '__iter__' in dir(self.channel):
            y_true = (np.isin(y_true, self.channel)).astype(y_pred.dtype)[..., np.newaxis]
            y_pred = (np.isin(y_pred, self.channel)).astype(y_pred.dtype)[..., np.newaxis]
        else:
            y_true = (y_true == self.channel).astype(y_pred.dtype)[..., np.newaxis]
            y_pred = (y_pred == self.channel).astype(y_pred.dtype)[..., np.newaxis]
        return super().calculate_metrics(y_true, y_pred, **kwargs)


class MultiLabelH5CalculateRecall(H5CalculateRecall):
    def __init__(self, ref_file, save_file, metric_name='',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4, beta=1, threshold=None,
                 map_file=None, map_column=None, channel=0):
        super().__init__(ref_file, save_file,
                         metric_name or self.__class__.name,
                         predicted_dataset,
                         target_dataset, batch_size,
                         map_file, map_column)
        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta
        self.channel = channel

    def calculate_metrics(self, y_true, y_pred, **kwargs):
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        if '__iter__' in dir(self.channel):
            y_true = (np.isin(y_true, self.channel)).astype(y_pred.dtype)[..., np.newaxis]
            y_pred = (np.isin(y_pred, self.channel)).astype(y_pred.dtype)[..., np.newaxis]
        else:
            y_true = (y_true == self.channel).astype(y_pred.dtype)[..., np.newaxis]
            y_pred = (y_pred == self.channel).astype(y_pred.dtype)[..., np.newaxis]
        return super().calculate_metrics(y_true, y_pred, **kwargs)


class MultiLabelH5CalculatePrecision(H5CalculatePrecision):
    def __init__(self, ref_file, save_file, metric_name='',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4, beta=1, threshold=None,
                 map_file=None, map_column=None, channel=0):
        super().__init__(ref_file, save_file,
                         metric_name or self.__class__.name,
                         predicted_dataset,
                         target_dataset, batch_size,
                         map_file, map_column)
        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta
        self.channel = channel

    def calculate_metrics(self, y_true, y_pred, **kwargs):
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        if '__iter__' in dir(self.channel):
            y_true = (np.isin(y_true, self.channel)).astype(y_pred.dtype)[..., np.newaxis]
            y_pred = (np.isin(y_pred, self.channel)).astype(y_pred.dtype)[..., np.newaxis]
        else:
            y_true = (y_true == self.channel).astype(y_pred.dtype)[..., np.newaxis]
            y_pred = (y_pred == self.channel).astype(y_pred.dtype)[..., np.newaxis]
        return super().calculate_metrics(y_true, y_pred, **kwargs)


class MultiLabelH5CalculateFPR(H5CalculateFPR):
    def __init__(self, ref_file, save_file, metric_name='',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4, beta=1, threshold=None,
                 map_file=None, map_column=None, channel=0):
        super().__init__(ref_file, save_file,
                         metric_name or self.__class__.name,
                         predicted_dataset,
                         target_dataset, batch_size,
                         map_file, map_column)
        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta
        self.channel = channel

    def calculate_metrics(self, y_true, y_pred, **kwargs):
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        if '__iter__' in dir(self.channel):
            y_true = (np.isin(y_true, self.channel)).astype(y_pred.dtype)[..., np.newaxis]
            y_pred = (np.isin(y_pred, self.channel)).astype(y_pred.dtype)[..., np.newaxis]
        else:
            y_true = (y_true == self.channel).astype(y_pred.dtype)[..., np.newaxis]
            y_pred = (y_pred == self.channel).astype(y_pred.dtype)[..., np.newaxis]
        return super().calculate_metrics(y_true, y_pred, **kwargs)


class MultiLabelH5CalculateSurfaceDice(H5CalculateFPR):
    name = 'SurfaceDice'

    def __init__(self, ref_file, save_file, metric_name='',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4, beta=1, threshold=None,
                 map_file=None, map_column=None, channel=0, tolerance=1):
        super().__init__(ref_file, save_file,
                         metric_name or self.__class__.name,
                         predicted_dataset,
                         target_dataset, batch_size,
                         map_file, map_column)
        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta
        self.channel = channel
        self.tolerance = tolerance

    def calculate_metrics(self, y_true, y_pred, **kwargs):
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        if '__iter__' in dir(self.channel):
            y_true = (np.isin(y_true, self.channel)).astype(y_pred.dtype)
            y_pred = (np.isin(y_pred, self.channel)).astype(y_pred.dtype)
        else:
            y_true = (y_true == self.channel).astype(y_pred.dtype)
            y_pred = (y_pred == self.channel).astype(y_pred.dtype)

        outputs = []
        for y_t, y_p in zip(y_true, y_pred):
            surface_distances = metrics.compute_surface_distances(
                y_t > 0, y_p > self.threshold, (1, 1, 1))
            outputs.append(metrics.compute_surface_dice_at_tolerance(
                surface_distances, self.tolerance))
        return np.array(outputs)


class MultiLabelH5CalculateHausdorff(H5CalculateFPR):
    name = 'HausdorffDistance'

    def __init__(self, ref_file, save_file, metric_name='',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4, beta=1, threshold=None,
                 map_file=None, map_column=None, channel=0, percentile=100):
        super().__init__(ref_file, save_file,
                         metric_name or self.__class__.name,
                         predicted_dataset,
                         target_dataset, batch_size,
                         map_file, map_column)
        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta
        self.channel = channel
        self.percentile = percentile

    def calculate_metrics(self, y_true, y_pred, **kwargs):
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        if '__iter__' in dir(self.channel):
            y_true = (np.isin(y_true, self.channel)).astype(y_pred.dtype)
            y_pred = (np.isin(y_pred, self.channel)).astype(y_pred.dtype)
        else:
            y_true = (y_true == self.channel).astype(y_pred.dtype)
            y_pred = (y_pred == self.channel).astype(y_pred.dtype)

        outputs = []
        for y_t, y_p in zip(y_true, y_pred):
            surface_distances = metrics.compute_surface_distances(
                y_t > 0, y_p > self.threshold, (1, 1, 1))
            outputs.append(metrics.compute_robust_hausdorff(
                surface_distances, self.percentile))
        return np.array(outputs)


class MultiLabelH5CalculateASD(H5CalculateFPR):
    name = 'ASD'

    def __init__(self, ref_file, save_file, metric_name='',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4, beta=1, threshold=None,
                 map_file=None, map_column=None, channel=0, index=0):
        super().__init__(ref_file, save_file,
                         metric_name or self.__class__.name,
                         predicted_dataset,
                         target_dataset, batch_size,
                         map_file, map_column)
        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta
        self.channel = channel
        self.index = index

    def calculate_metrics(self, y_true, y_pred, **kwargs):
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        if '__iter__' in dir(self.channel):
            y_true = (np.isin(y_true, self.channel)).astype(y_pred.dtype)
            y_pred = (np.isin(y_pred, self.channel)).astype(y_pred.dtype)
        else:
            y_true = (y_true == self.channel).astype(y_pred.dtype)
            y_pred = (y_pred == self.channel).astype(y_pred.dtype)

        outputs = []
        for y_t, y_p in zip(y_true, y_pred):
            surface_distances = metrics.compute_surface_distances(
                y_t > 0, y_p > self.threshold, (1, 1, 1))
            outputs.append(metrics.compute_average_surface_distance(
                surface_distances)[self.index])
        return np.array(outputs)


class MultiLabelSegmentationPostProcessor(SegmentationPostProcessor):
    METRIC_NAME_MAP = {
        'f1_score': MultiLabelH5CalculateFScore,
        'dice': MultiLabelH5CalculateFScore,
        'dice_score': MultiLabelH5CalculateFScore,
        'fbeta': MultiLabelH5CalculateFScore,
        'recall': MultiLabelH5CalculateRecall,
        'sensitivity': MultiLabelH5CalculateRecall,
        'TPR': MultiLabelH5CalculateRecall,
        'precision': MultiLabelH5CalculatePrecision,
        'FPR': MultiLabelH5CalculateFPR,
        'surface_dice': MultiLabelH5CalculateSurfaceDice,
        'HD': MultiLabelH5CalculateHausdorff,
        'ASD': MultiLabelH5CalculateASD,
    }

    def calculate_metrics_single_3d(self, metrics=None, metrics_kwargs=None):
        self.calculate_metrics_single(metrics, metrics_kwargs)
        if not self.run_test:
            map_folder = self.log_base_path + self.SINGLE_MAP_PATH

            main_log_folder = self.log_base_path + self.MAP_PATH
            try:
                os.rename(map_folder, main_log_folder)
            except Exception as e:
                print("Files exist:", e)
                print("Copying new logs file")
                os.rename(main_log_folder,
                          main_log_folder + '-' + str(time()))
                os.rename(map_folder, main_log_folder)

        else:
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            map_filename = test_folder + self.TEST_SINGLE_MAP_NAME

            main_result_file_name = test_folder + self.TEST_MAP_NAME
            try:
                os.rename(map_filename, main_result_file_name)
            except Exception as e:
                print("Files exist:", e)
                print("Copying new result file")
                os.rename(main_result_file_name,
                          main_result_file_name + '-' + str(time()) + '.csv')
                os.rename(map_filename, main_result_file_name)

        return self

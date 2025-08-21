from deoxys.experiment.postprocessor import SegmentationPostProcessor, \
    H5CalculateFScore, H5CalculateRecall, H5CalculatePrecision, H5CalculateFPR


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
        y_true = y_true[..., [self.channel]]
        y_pred = y_pred[..., [self.channel]]
        super().calculate_metrics(y_true, y_pred, **kwargs)


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
        y_true = y_true[..., [self.channel]]
        y_pred = y_pred[..., [self.channel]]
        super().calculate_metrics(y_true, y_pred, **kwargs)


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
        y_true = y_true[..., [self.channel]]
        y_pred = y_pred[..., [self.channel]]
        super().calculate_metrics(y_true, y_pred, **kwargs)


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
        y_true = y_true[..., [self.channel]]
        y_pred = y_pred[..., [self.channel]]
        super().calculate_metrics(y_true, y_pred, **kwargs)


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
        'FPR': MultiLabelH5CalculateFPR
    }

import torch



class ConditionalRatioEstimator(object):

    def forward(self, xs, ys):
        r""""""
        raise NotImplementedError

    def log_ratio(self, xs, ys):
        r""""""
        raise NotImplementedError



class BaseConditionalRatioEstimator(torch.nn.Module, ConditionalRatioEstimator):
    r""""""

    def __init__(self):
        super(ConditionalRatioEstimator, self).__init__()

    def forward(self, xs, ys):
        r""""""
        raise NotImplementedError

    def log_ratio(self, xs, ys):
        r""""""
        raise NotImplementedError



class ConditionalRatioEstimatorLoss(torch.nn.Module):

    def __init__(self, ratio_estimator, criterion=torch.nn.BCE):
        self.ratio_estimator = ratio_estimator
        self.criterion = criterion()

    def forward(self, batch_a, batch_b):
        raise NotImplementedError



class ConditionalRatioEstimatorBCEWithLogitsLoss(torch.nn.Module):

    def __init__(self, ratio_estimator):
        self.ratio_estimator = ratio_estimator
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch_a, batch_b):
        raise NotImplementedError

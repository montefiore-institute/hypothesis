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
        ConditionalRatioEstimator.__init__(self)
        torch.nn.Module.__init__(self)

    def forward(self, xs, ys):
        r""""""
        raise NotImplementedError

    def log_ratio(self, xs, ys):
        r""""""
        raise NotImplementedError



class ConditionalRatioEstimatorLoss(torch.nn.Module):

    def __init__(self, ratio_estimator, criterion=None, ones=None, zeros=None):
        self.criterion = criterion
        self.ones = ones
        self.zeros = zeros
        self.ratio_estimator = ratio_estimator

    def forward(self, batch_a, batch_b):
        inputs_a, outputs_a = batch_a
        inputs_b, outputs_b = batch_b
        y_dependent_a, _ = self.ratio_estimator(inputs_a, outputs_a)
        y_independent_a, _ = self.ratio_estimator(inputs_a, outputs_b)
        y_dependent_b, _ = self.ratio_estimator(inputs_b, outputs_b)
        y_independent_b, _ = self.ratio_estimator(inputs_b, outputs_a)
        loss_a = self.criterion(y_dependent_a, ones) + self.criterion(y_independent_a, zeros)
        loss_b = self.criterion(y_dependent_b, ones) + self.criterion(y_independent_b, zeros)
        loss = loss_a + loss_b

        return loss



class ConditionalRatioEstimatorBCEWithLogitsLoss(torch.nn.Module):

    def __init__(self, ratio_estimator):
        self.ratio_estimator = ratio_estimator
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch_a, batch_b):
        raise NotImplementedError

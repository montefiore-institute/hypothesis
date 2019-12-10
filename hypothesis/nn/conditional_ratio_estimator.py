import torch

from hypothesis.exception import NotDivisibleByTwoException



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



class ConditionalRatioEstimatorEnsemble(BaseConditionalRatioEstimator):
    r""""""

    def __init__(self, ratio_estimators, reduction=torch.mean):
        super(ConditionalRatioEstimatorEnsemble, self).__init__()
        self.ratio_estimators = ratio_estimators
        self.reduction = reduction

    def forward(self, xs, ys):
        log_ratios = self.log_ratios(xs, ys)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, xs, ys):
        log_ratios = []
        for ratio_estimator in self.ratio_estimators:
            log_ratios.append(ratio_estimator.log_ratio(xs, ys))
        log_ratios = torch.cat(log_ratios, dim=1)
        log_ratios = self.reduction(log_ratios, axis=1)

        return log_ratios



class ConditionalRatioEstimatorCriterion(torch.nn.Module):
    r""""""

    def __init__(self, ratio_estimator, batch_size):
        super(ConditionalRatioEstimatorCriterion, self).__init__()
        # Check if a valid batch size has been supplied.
        if batch_size % 2 != 0:
            raise NotDivisibleByTwoException(
                "Only batch sizes divisible by two are permitted.")
        assert(batch_size % 2 == 0)
        self.chunked_batch_size = batch_size // 2
        self.criterion = torch.nn.BCELoss()
        self.ratio_estimator = ratio_estimator
        self.ones = torch.ones(self.chunked_batch_size, 1)
        self.zeros = torch.zeros(self.chunked_batch_size, 1)

    def to(self, device):
        self.criterion = self.criterion.to(device)
        self.ones = self.ones.to(device)
        self.zeros = self.zeros.to(device)

        return self

    def forward(self, xs, ys):
        xs = xs.chunk(2)
        ys = ys.chunk(2)
        xs_a = xs[0]
        xs_b = xs[1]
        ys_a = ys[0]
        ys_b = ys[1]
        y_dependent_a, _ = self.ratio_estimator(xs_a, ys_a)
        y_independent_a, _ = self.ratio_estimator(xs_a, ys_b)
        y_dependent_b, _ = self.ratio_estimator(xs_b, ys_b)
        y_independent_b, _ = self.ratio_estimator(xs_b, ys_a)
        loss_a = self.criterion(y_dependent_a, self.ones) + \
                 self.criterion(y_independent_a, self.zeros)
        loss_b = self.criterion(y_dependent_b, self.ones) + \
                 self.criterion(y_independent_b, self.zeros)
        loss = loss_a + loss_b

        return loss



class ConditionalRatioEstimatorLogitsCriterion(torch.nn.Module):
    r""""""

    def __init__(self, ratio_estimator, batch_size):
        super(ConditionalRatioEstimatorLogitsCriterion, self).__init__()
        # Check if a valid batch size has been supplied.
        if batch_size % 2 != 0:
            raise NotDivisibleByTwoException(
                "Only batch sizes divisible by two are permitted.")
        self.chunked_batch_size = batch_size // 2
        self.ratio_estimator = ratio_estimator
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.ones = torch.ones(self.chunked_batch_size, 1)
        self.zeros = torch.zeros(self.chunked_batch_size, 1)

    def to(self, device):
        self.criterion = self.criterion.to(device)
        self.ones = self.ones.to(device)
        self.zeros = self.zeros.to(device)

        return self

    def forward(self, xs, ys):
        xs = xs.chunk(2)
        ys = ys.chunk(2)
        xs_a = xs[0]
        xs_b = xs[1]
        ys_a = ys[0]
        ys_b = ys[1]
        _, y_dependent_a = self.ratio_estimator(xs_a, ys_a)
        _, y_independent_a = self.ratio_estimator(xs_a, ys_b)
        _, y_dependent_b = self.ratio_estimator(xs_b, ys_b)
        _, y_independent_b = self.ratio_estimator(xs_b, ys_a)
        loss_a = self.criterion(y_dependent_a, self.ones) + \
                 self.criterion(y_independent_a, self.zeros)
        loss_b = self.criterion(y_dependent_b, self.ones) + \
                 self.criterion(y_independent_b, self.zeros)
        loss = loss_a + loss_b

        return loss

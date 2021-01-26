import hypothesis as h
import numpy as np
import torch


class BaseRatioEstimator(torch.nn.Module):

    def __init__(self):
        super(BaseRatioEstimator, self).__init__()

    def forward(self, **kwargs):
        log_ratios = self.log_ratio(**kwargs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, **kwargs):
        raise NotImplementedError


class RatioEstimatorEnsemble(BaseRatioEstimator):

    KEYWORD_REDUCE = "reduce"
    REDUCE_MEAN = "mean"
    REDUCE_MEDIAN = "median"

    def __init__(self, estimators, reduce=REDUCE_MEAN):
        super(RatioEstimatorEnsemble, self).__init__()
        self._estimators = estimators
        self.reduce_as(reduce)

    def to(self, device):
        for index, estimator in enumerate(self._estimators):
            self._estimators[index] = self._estimators.to(device)

    def reduce_as(self, reducer):
        self._reduce = self._allocate_reduce(reducer)

    def log_ratio(self, **kwargs):
        log_ratios = []
        for r in self.estimators:
            log_ratios.append(r.log_ratio(**kwargs))
        log_ratios = torch.cat(log_ratios, dim=1)
        if self._reduce is not None:
            log_ratios = self._reduce(log_ratios).view(-1, 1)

        return log_ratios

    @staticmethod
    def _allocate_reduce(f):
        reducers = {
            REDUCE_MEAN: RatioEstimatorEnsemble._reduce_mean,
            REDUCE_MEDIAN: RatioEstimatorEnsemble._reduce_median}
        reduce = None
        if hasattr(f, "__call__"):
            return f
        elif f in reducers:
            return reducers[f]
        else:
            raise ValueError("The specified reduce method does not exist.")

    @staticmethod
    def _reduce_mean(log_ratios):
        return log_ratios.mean(dim=1)

    @staticmethod
    def _reduce_median(log_ratios):
        return log_ratios.median(dim=1).values

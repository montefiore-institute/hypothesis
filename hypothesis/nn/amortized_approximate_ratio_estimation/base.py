import hypothesis
import hypothesis.nn
import torch



class BaseAmortizedApproximateRatioEstimator(torch.nn.Module):

    def __init__(self):
        super(BaseAmortizedApproximateRatioEstimator, self).__init__()

    def forward(self, **kwargs):
        log_ratios = self.log_ratio(**kwargs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, **kwargs):
        raise NotImplementedError



class AmortizedApproximateRatioEstimatorEnsemble(BaseAmortizedApproximateRatioEstimator):

    def __init__(self, estimators, reduce="mean"):
        super(AmortizedApproximateRatioEstimatorEnsemble, self).__init__()
        self.estimators = estimators
        self.reduce = self._allocate_reduce(reduce)

    def log_ratio(self, **kwargs, reduce=True):
        log_ratios = []
        for estimator in self.estimators:
            log_ratios.append(estimator.log_ratio(**kwargs))
        log_ratios = torch.cat(log_ratios, dim=1)
        if reduce:
            log_ratios = self.reduce(log_ratios)

        return log_ratios

    @staticmethod
    def _allocate_reduce(f):
        reductions = {
            "mean": AmortizedApproximateRatioEstimatorEnsemble._reduce_mean,
            "median": AmortizedApproximateRatioEstimatorEnsemble._reduce_median}
        reduce = None
        if hasattr(f, "__call__"):
            return f
        else:
            return reductions[f]

    @staticmethod
    def _reduce_mean(log_ratios):
        return log_ratios.mean(dim=1)

    @staticmethod
    def _reduce_median(log_ratios):
        return log_ratios.median(dim=1).values



class BaseCriterion(torch.nn.Module):

    def __init__(self, estimator,
        batch_size=hypothesis.default.batch_size)

    def forward(self, **kwargs):
        raise NotImplementedError

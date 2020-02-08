import hypothesis
import hypothesis.nn
import re
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

    def log_ratio(self, **kwargs):
        reduce = kwargs.get("reduce", True)
        log_ratios = []
        for estimator in self.estimators:
            log_ratios.append(estimator.log_ratio(**kwargs))
        log_ratios = torch.cat(log_ratios, dim=1)
        if reduce:
            log_ratios = self.reduce(log_ratios).view(-1, 1)

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

    def __init__(self,
        estimator,
        denominator,
        criterion=torch.nn.BCELoss,
        batch_size=hypothesis.default.batch_size):
        super(BaseCriterion, self).__init__()
        self.criterion = criterion()
        self.estimator = estimator
        self.independent_random_variables = self._derive_independent_random_variables(denominator)
        self.ones = torch.ones(batch_size, 1)
        self.random_variables = self._derive_random_variables(denominator)
        self.zeros = torch.zeros(batch_size, 1)

    def _derive_random_variables(self, denominator):
        random_variables = denominator.replace(hypothesis.default.dependent_delimiter, " ") \
            .replace(hypothesis.default.independent_delimiter, " ") \
            .split(" ")

        return random_variables

    def _derive_independent_random_variables(self, denominator):
        splitted = denominator.split(hypothesis.default.independent_delimiter)
        independent_variables = [v for v in splitted if hypothesis.default.dependent_delimiter not in v]

        return independent_variables

    def variables(self):
        return self.random_variables

    def independent_variables(self):
        return self.independent_random_variables

    def to(self, device):
        self.criterion = self.criterion.to(device)
        self.ones = self.ones.to(device)
        self.zeros = self.zeros.to(device)

    def forward(self, **kwargs):
        y_dependent = self.estimator(**kwargs)
        for variable in self.independent_random_variables:
            kwargs[variable] = kwargs[variable][torch.randperm(self.batch_size)] # Make variable independent.
        y_independent = self.estimator(**kwargs)
        loss = self.criterion(y_dependent, self.ones) + self.criterion(y_independent_a, self.zeros)

        return loss

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

        return self

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


class BaseCriterion(torch.nn.Module):

    def __init__(self,
        estimator,
        denominator,
        batch_size=h.default.batch_size,
        logits=False):
        super(BaseCriterion, self).__init__()
        if logits:
            self._criterion = torch.nn.BCEWithLogitsLoss()
            self._forward = self._forward_with_logits
        else:
            self._criterion = torch.nn.BCELoss()
            self._forward = self._forward_without_logits
        self._batch_size = batch_size
        self._estimator = estimator
        self._independent_random_variables = self._derive_independent_random_variables(denominator)
        self._ones = torch.ones(self._batch_size, 1)
        self._random_variables = self._derive_random_variables(denominator)
        self._zeros = torch.zeros(self._batch_size, 1)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def variables(self):
        return self._random_variables

    @property
    def independent_variables(self):
        return self._independent_random_variables

    def to(self, device):
        self._criterion = self._criterion.to(device)
        self._ones = self._ones.to(device)
        self._zeros = self._zeros.to(device)

        return self

    def forward(self, **kwargs):
        return self._forward(**kwargs)

    def _forward_without_logits(self, **kwargs):
        y_dependent, _ = self.estimator(**kwargs)
        for group in self._independent_random_variables:
            random_indices = torch.randperm(self._batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make variable independent.
        y_independent, _ = self._estimator(**kwargs)
        loss = self._criterion(y_dependent, self._ones) + self.criterion(y_independent, self._zeros)

        return loss

    def _forward_with_logits(self, **kwargs):
        y_dependent = self._estimator.log_ratio(**kwargs)
        for group in self._independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make variable independent.
        y_independent = self._estimator.log_ratio(**kwargs)
        loss = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)

        return loss

    @staticmethod
    def _derive_random_variables(denominator):
        random_variables = denominator.replace(h.default.dependent_delimiter, " ") \
            .replace(h.default.independent_delimiter, " ") \
            .split(" ")
        random_variables.sort()  # The random variables have to appear in a sorted order.

        return random_variables

    @staticmethod
    def _derive_independent_random_variables(self, denominator):
        groups = denominator.split(hypothesis.default.independent_delimiter)
        for index in range(len(groups)):
            groups[index] = groups[index].split(h.default.dependent_delimiter)

        return groups

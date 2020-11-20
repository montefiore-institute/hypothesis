import hypothesis
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

    def __init__(self, estimators, reduce="mean"):
        super(RatioEstimatorEnsemble, self).__init__()
        self.estimators = estimators
        self.reduce = self._allocate_reduce(reduce)

    def reduce_as(self, reduce):
        self.reduce = self._allocate_reduce(reduce)

    def to(self, device):
        for index in range(len(self.estimators)):
            self.estimators[index] = self.estimators[index].to(device)

        return self

    def log_ratio(self, **kwargs):
        # Check if the 'reduce' keyword is an argument.
        if RatioEstimatorEnsemble.KEYWORD_REDUCE in kwargs.keys():
            reduce = kwargs[RatioEstimatorEnsemble.KEYWORD_REDUCE]
            del kwargs[RatioEstimatorEnsemble.KEYWORD_REDUCE]
        else:
            reduce = True # Default value
        # Estimate the log ratios
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
            "mean": RatioEstimatorEnsemble._reduce_mean,
            "median": RatioEstimatorEnsemble._reduce_median}
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
        batch_size=hypothesis.default.batch_size,
        logits=False):
        super(BaseCriterion, self).__init__()
        if logits:
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self._forward = self._forward_with_logits
        else:
            self.criterion = torch.nn.BCELoss()
            self._forward = self._forward_without_logits
        self.batch_size = batch_size
        self.estimator = estimator
        self.independent_random_variables = self._derive_independent_random_variables(denominator)
        self.ones = torch.ones(self.batch_size, 1)
        self.random_variables = self._derive_random_variables(denominator)
        self.zeros = torch.zeros(self.batch_size, 1)

    def _derive_random_variables(self, denominator):
        random_variables = denominator.replace(hypothesis.default.dependent_delimiter, " ") \
            .replace(hypothesis.default.independent_delimiter, " ") \
            .split(" ")
        random_variables.sort()

        return random_variables

    def _derive_independent_random_variables(self, denominator):
        groups = denominator.split(hypothesis.default.independent_delimiter)
        for index in range(len(groups)):
            groups[index] = groups[index].split(hypothesis.default.dependent_delimiter)

        return groups

    def _forward_without_logits(self, **kwargs):
        y_dependent, _ = self.estimator(**kwargs)
        for group in self.independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices] # Make variable independent.
        y_independent, _ = self.estimator(**kwargs)
        loss = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)

        return loss

    def _forward_with_logits(self, **kwargs):
        _, y_dependent = self.estimator(**kwargs)
        for group in self.independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices] # Make variable independent.
        _, y_independent = self.estimator(**kwargs)
        loss = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)

        return loss

    def variables(self):
        return self.random_variables

    def independent_variables(self):
        return self.independent_random_variables

    def to(self, device):
        self.criterion = self.criterion.to(device)
        self.ones = self.ones.to(device)
        self.zeros = self.zeros.to(device)

        return self

    def forward(self, **kwargs):
        return self._forward(**kwargs)



class BaseConservativeCriterion(BaseCriterion):

    def __init__(self,
        estimator,
        denominator,
        batch_size=hypothesis.default.batch_size,
        beta=0.001,
        logits=False):
        super(BaseConservativeCriterion, self).__init__(
            estimator=estimator,
            denominator=denominator,
            batch_size=batch_size,
            logits=logits)
        self.beta = beta

    def _forward_without_logits(self, **kwargs):
        beta = self.beta
        y_dependent, _ = self.estimator(**kwargs)
        for group in self.independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices] # Make variable independent.
        y_independent, _ = self.estimator(**kwargs)
        loss = ((1 - beta) * self.criterion(y_dependent, self.ones) + beta * self.criterion(y_independent, self.ones)) + self.criterion(y_independent, self.zeros)

        return loss

    def _forward_with_logits(self, **kwargs):
        beta = self.beta
        _, y_dependent = self.estimator(**kwargs)
        for group in self.independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices] # Make variable independent.
        _, y_independent = self.estimator(**kwargs)
        loss = ((1 - beta) * self.criterion(y_dependent, self.ones) + beta * self.criterion(y_independent, self.ones)) + self.criterion(y_independent, self.zeros)


        return loss


class BaseExperimentalCriterion(BaseCriterion):

    def __init__(self,
        estimator,
        denominator,
        batch_size=hypothesis.default.batch_size,
        logits=False):
        super(BaseExperimentalCriterion, self).__init__(
            estimator=estimator,
            denominator=denominator,
            batch_size=batch_size,
            logits=logits)
        self.base = np.log(4)

    def _forward_without_logits(self, **kwargs):
        y_dependent, log_mi = self.estimator(**kwargs)
        for group in self.independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices] # Make variable independent.
        y_independent, _ = self.estimator(**kwargs)
        loss = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)
        # TODO Add MI regularizer

        return loss

    def _forward_with_logits(self, **kwargs):
        _, y_dependent = self.estimator(**kwargs)
        for group in self.independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices] # Make variable independent.
        _, y_independent = self.estimator(**kwargs)
        loss = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)
        # TODO Add MI regularizer

        return loss

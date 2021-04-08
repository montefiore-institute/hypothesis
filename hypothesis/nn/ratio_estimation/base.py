import hypothesis as h
import numpy as np
import re
import torch


class BaseRatioEstimator(torch.nn.Module):

    def __init__(self, denominator=None, random_variables=None, r=None):
        super(BaseRatioEstimator, self).__init__()
        if r is None:
            denominator_rv = set(denominator.replace(',', ' ').replace('|', ' ').split(' '))
            assert denominator_rv == set(random_variables.keys())
        else:
            denominator = r.denominator
            random_variables = r.random_variables
        self._denominator = denominator  # Denominator of the ratio
        self._random_variables = random_variables  # A dictionary with the name and shape of the random variable.

    @property
    def denominator(self):
        return self._denominator

    @property
    def random_variables(self):
        return self._random_variables

    def forward(self, **kwargs):
        log_ratios = self.log_ratio(**kwargs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, **kwargs):
        raise NotImplementedError


class RatioEstimatorEnsemble(BaseRatioEstimator):

    KEYWORD_REDUCE = "reduce"
    REDUCE_MEAN = "mean"
    REDUCE_MEDIAN = "median"

    def __init__(self, estimators, reduce="mean"):
        denominator = estimators[0].denominator
        random_variables = estimators[0].random_variables
        super(RatioEstimatorEnsemble, self).__init__(
            denominator=denominator,
            random_variables=random_variables)
        self._estimators = estimators
        self.reduce_as(reduce)

    def parameters(self):
        parameters = []
        for estimator in self._estimators:
            parameters.extend(estimator.parameters())

        return parameters

    def to(self, device):
        for index, estimator in enumerate(self._estimators):
            self._estimators[index] = self._estimators[index].to(device)

        return self

    def reduce_as(self, reducer):
        self._reduce = self._allocate_reduce(reducer)

    def log_ratio(self, **kwargs):
        log_ratios = []
        for r in self._estimators:
            log_ratios.append(r.log_ratio(**kwargs))
        log_ratios = torch.cat(log_ratios, dim=1)
        if self._reduce is not None:
            log_ratios = self._reduce(log_ratios).view(-1, 1)

        return log_ratios

    @staticmethod
    def _allocate_reduce(f):
        reducers = {
            RatioEstimatorEnsemble.REDUCE_MEAN: RatioEstimatorEnsemble._reduce_mean,
            RatioEstimatorEnsemble.REDUCE_MEDIAN: RatioEstimatorEnsemble._reduce_median}
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
        denominator = self._estimator.denominator
        self._independent_random_variables = self._derive_independent_random_variables(denominator)
        self._ones = torch.ones(self._batch_size, 1)
        self._zeros = torch.zeros(self._batch_size, 1)

    @property
    def batch_size(self):
        return self._batch_size

    def to(self, device):
        self._criterion = self._criterion.to(device)
        self._ones = self._ones.to(device)
        self._zeros = self._zeros.to(device)

        return self

    def forward(self, **kwargs):
        return self._forward(**kwargs)

    def _forward_without_logits(self, **kwargs):
        y_dependent, _ = self._estimator(**kwargs)
        for group in self._independent_random_variables:
            random_indices = torch.randperm(self._batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make variable independent.
        y_independent, _ = self._estimator(**kwargs)
        loss = self._criterion(y_dependent, self._ones) + self._criterion(y_independent, self._zeros)

        return loss

    def _forward_with_logits(self, **kwargs):
        y_dependent = self._estimator.log_ratio(**kwargs)
        for group in self._independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make variable independent.
        y_independent = self._estimator.log_ratio(**kwargs)
        loss = self._criterion(y_dependent, self._ones) + self._criterion(y_independent, self._zeros)

        return loss

    @staticmethod
    def _derive_independent_random_variables(denominator):
        groups = denominator.split(h.default.independent_delimiter)
        for index in range(len(groups)):
            groups[index] = groups[index].split(h.default.dependent_delimiter)

        return groups


class ConservativeCriterion(BaseCriterion):

    def __init__(self,
        estimator,
        calibrate=True,
        conservativeness=0.0,
        batch_size=h.default.batch_size,
        gamma=25.0,
        logits=False):
        super(ConservativeCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            logits=logits)
        self._beta = conservativeness
        self._calibrate = calibrate
        self._gamma = gamma

    @property
    def conservativeness(self):
        return self._beta

    @conservativeness.setter
    def conservativeness(self, value):
        assert value >= 0.0
        self._beta = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        assert value >= 0
        self._gamma = value

    def _forward_without_logits(self, **kwargs):
        # Forward passes
        y_joint, log_r_joint = self._estimator(**kwargs)
        ## Shuffle to make necessary variables independent.
        for group in self._independent_random_variables:
            random_indices = torch.randperm(self._batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make variable independent.
        y_marginals, _ = self._estimator(**kwargs)
        # Compute losses
        loss_joint_1 = self._criterion(y_joint, self._ones)
        loss_marginals_0 = self._criterion(y_marginals, self._zeros)
        # Learn mixture of the joint vs. marginals
        loss = loss_joint_1 + loss_marginals_0
        if self._beta < 1.0:
            loss = loss + self._beta * log_r_joint.mean()  # Conservativeness regularizer
        # Check if calibration term needs to be added.
        if self._calibrate:
            calibration_term = (1.0 - y_joint - y_marginals).mean().pow(2)
            loss = loss + self._gamma * calibration_term  # Calibration

        return loss

    def _forward_with_logits(self, **kwargs):
        # Forward passes
        y_joint, log_r_joint = self._estimator(**kwargs)
        ## Shuffle to make necessary variables independent.
        for group in self._independent_random_variables:
            random_indices = torch.randperm(self._batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make variable independent.
        y_marginals, log_r_marginals = self._estimator(**kwargs)
        # Compute losses
        loss_joint_1 = self._criterion(log_r_joint, self._ones)
        loss_marginals_0 = self._criterion(log_r_marginals, self._zeros)
        # Learn mixture of the joint vs. marginals
        loss = loss_joint_1 + loss_marginals_0
        if self._beta < 1.0:
            loss = loss + self._beta * log_r_joint.mean()  # Conservativeness regularizer
        # Check if calibration term needs to be added.
        if self._calibrate:
            calibration_term = (1.0 - y_joint - y_marginals).mean().pow(2)
            loss = loss + self._gamma * calibration_term  # Calibration

        return loss

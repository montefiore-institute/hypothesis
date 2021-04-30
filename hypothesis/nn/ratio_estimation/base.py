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
    REDUCE_DISCRIMINATOR_MEAN = "discriminator_mean"
    REDUCE_RATIO_MEAN = "ratio_mean"

    def __init__(self, estimators, reduce="discriminator_mean"):
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
            RatioEstimatorEnsemble.REDUCE_MEDIAN: RatioEstimatorEnsemble._reduce_median,
            RatioEstimatorEnsemble.REDUCE_DISCRIMINATOR_MEAN: RatioEstimatorEnsemble._reduce_discriminator_mean,
            RatioEstimatorEnsemble.REDUCE_RATIO_MEAN: RatioEstimatorEnsemble._reduce_ratio_mean}
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

    @staticmethod
    def _reduce_discriminator_mean(log_ratios):
        return torch.logit(torch.sigmoid(log_ratios).mean(dim=1), eps=1e-6)

    @staticmethod
    def _reduce_ratio_mean(log_ratios):
        return torch.log(torch.exp(log_ratios).mean(dim=1))


class BaseCriterion(torch.nn.Module):

    def __init__(self,
        estimator,
        batch_size=h.default.batch_size,
        logits=False,
        **kwargs):
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
    def estimator(self):
        return self._estimator

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        assert batch_size >= 1
        self._batch_size = batch_size

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
        balance=True,
        conservativeness=0.0,
        batch_size=h.default.batch_size,
        gamma=10.0,
        logits=False, **kwargs):
        super(ConservativeCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            logits=logits,
            **kwargs)
        self._balance = balance
        self._beta = conservativeness
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

    def _balance_ratio_estimator(self, loss, log_r_marginals=None, log_r_joint=None, y_joint=None, y_marginals=None):
        if self._balance:
            term = (1.0 - y_joint.mean() - y_marginals.mean()).pow(2)
            loss = loss + self._gamma * term

        return loss

    def _forward_without_logits(self, **kwargs):
        # Forward passes
        y_joint, log_r_joint = self._estimator(**kwargs)
        ## Shuffle to make necessary variables independent.
        for group in self._independent_random_variables:
            random_indices = torch.randperm(self._batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make variable independent.
        y_marginals, log_r_marginals = self._estimator(**kwargs)
        # Compute losses
        loss_joint_1 = self._criterion(y_joint, self._ones)
        loss_marginals_0 = self._criterion(y_marginals, self._zeros)
        # Learn mixture of the joint vs. marginals
        loss = loss_joint_1 + loss_marginals_0
        if self._beta > 0.0:
            loss = loss + self._beta * log_r_joint.mean()  # Conservativeness regularizer
        loss = self._balance_ratio_estimator(loss,
            log_r_joint=log_r_joint,
            log_r_marginals=log_r_marginals,
            y_marginals=y_marginals,
            y_joint=y_joint)

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
        if self._beta > 0.0:
            loss = loss + self._beta * log_r_joint.mean()  # Conservativeness regularizer
        loss = self._balance_ratio_estimator(loss,
            log_r_joint=log_r_joint,
            log_r_marginals=log_r_marginals,
            y_marginals=y_marginals,
            y_joint=y_joint)

        return loss


class DualConservativeCriterion(ConservativeCriterion):

    def __init__(self,
        estimator,
        balance=True,
        conservativeness=0.0,
        batch_size=h.default.batch_size,
        gamma=10.0,
        logits=False, **kwargs):
        super(DualConservativeCriterion, self).__init__(
            balance=balance,
            batch_size=batch_size,
            conservativeness=conservativeness,
            estimator=estimator,
            gamma=gamma,
            logits=logits,
            **kwargs)

    def _balance_ratio_estimator(self, loss, log_r_marginals=None, log_r_joint=None, y_joint=None, y_marginals=None):
        if self._balance:
            term_a = (1.0 - log_r_marginals.exp()).mean().pow(2)
            term_b = (1.0 - y_joint.mean() - y_marginals.mean()).pow(2)
            loss = loss + self._gamma * (term_a + term_b)

        return loss


class ConservativeNewCriterion(ConservativeCriterion):

    def __init__(self,
        estimator,
        balance=True,
        conservativeness=0.0,
        batch_size=h.default.batch_size,
        gamma=10.0,
        logits=False, **kwargs):
        super(ConservativeNewCriterion, self).__init__(
            balance=balance,
            batch_size=batch_size,
            conservativeness=conservativeness,
            estimator=estimator,
            gamma=gamma,
            logits=logits,
            **kwargs)

    def _balance_ratio_estimator(self, loss, log_r_marginals=None, log_r_joint=None, y_joint=None, y_marginals=None):
        if self._balance:
            term = (1.0 - log_r_marginals.exp()).mean().pow(2)
            loss = loss + self._gamma * term

        return loss

class ConservativeNewAbsCriterion(ConservativeCriterion):

    def __init__(self,
        estimator,
        balance=True,
        conservativeness=0.0,
        batch_size=h.default.batch_size,
        gamma=10.0,
        logits=False, **kwargs):
        super(ConservativeNewAbsCriterion, self).__init__(
            balance=balance,
            batch_size=batch_size,
            conservativeness=conservativeness,
            estimator=estimator,
            gamma=gamma,
            logits=logits,
            **kwargs)

    def _balance_ratio_estimator(self, loss, log_r_marginals=None, log_r_joint=None, y_joint=None, y_marginals=None):
        if self._balance:
            term = torch.abs((1.0 - log_r_marginals.exp()).mean())
            loss = loss + self._gamma * term

        return loss

class ConservativeNew2Criterion(ConservativeCriterion):

    def __init__(self,
        estimator,
        balance=True,
        conservativeness=0.0,
        batch_size=h.default.batch_size,
        gamma=10.0,
        logits=False, **kwargs):
        super(ConservativeNew2Criterion, self).__init__(
            balance=balance,
            batch_size=batch_size,
            conservativeness=conservativeness,
            estimator=estimator,
            gamma=gamma,
            logits=logits,
            **kwargs)

    def _balance_ratio_estimator(self, loss, log_r_marginals=None, log_r_joint=None, y_joint=None, y_marginals=None):
        if self._balance:
            term = (1.0 - (1/(2*(1-y_marginals)))).mean().pow(2)
            loss = loss + self._gamma * term

        return loss

class ConservativeNew2AbsCriterion(ConservativeCriterion):

    def __init__(self,
        estimator,
        balance=True,
        conservativeness=0.0,
        batch_size=h.default.batch_size,
        gamma=10.0,
        logits=False, **kwargs):
        super(ConservativeNew2Criterion, self).__init__(
            balance=balance,
            batch_size=batch_size,
            conservativeness=conservativeness,
            estimator=estimator,
            gamma=gamma,
            logits=logits,
            **kwargs)

    def _balance_ratio_estimator(self, loss, log_r_marginals=None, log_r_joint=None, y_joint=None, y_marginals=None):
        if self._balance:
            term = torch.abs((1.0 - (1/(2*(1-y_marginals)))).mean())
            loss = loss + self._gamma * term

        return loss

import hypothesis as h
import numpy as np
import torch

from hypothesis.nn.ratio_estimation import BaseCriterion


class RegularizedCriterion(BaseCriterion):

    def __init__(self,
        estimator,
        batch_size=h.default.batch_size,
        gamma=100.0,
        logits=False, **kwargs):
        super(RegularizedCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            logits=logits,
            **kwargs)
        self._gamma = gamma

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        assert value >= 0
        self._gamma = value


class BalancedCriterion(RegularizedCriterion):

    def __init__(self,
        estimator,
        batch_size=h.default.batch_size,
        gamma=100.0,
        logits=False, **kwargs):
        super(BalancedCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            gamma=gamma,
            logits=logits,
            **kwargs)

    def _forward_without_logits(self, **kwargs):
        effective_batch_size = len(kwargs[self._independent_random_variables[0][0]])
        y_dependent, _ = self._estimator(**kwargs)
        for group in self._independent_random_variables:
            random_indices = torch.randperm(effective_batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make group independent.
        y_independent, log_r_independent = self._estimator(**kwargs)
        loss = self._criterion(y_dependent, self._ones[:effective_batch_size]) + self._criterion(y_independent, self._zeros[:effective_batch_size])
        # Balacing condition
        regularizer = (1.0 - y_dependent - y_independent).mean().pow(2)
        loss = loss + self._gamma * regularizer

        return loss

    def _forward_with_logits(self, **kwargs):
        effective_batch_size = len(kwargs[self._independent_random_variables[0][0]])
        y_dependent, log_r_dependent = self._estimator(**kwargs)
        for group in self._independent_random_variables:
            random_indices = torch.randperm(effective_batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make group independent.
        y_independent, log_r_independent = self._estimator.log_ratio(**kwargs)
        loss = self._criterion(log_r_dependent, self._ones[:effective_batch_size]) + self._criterion(log_r_independent, self._zeros[:effective_batch_size])
        # Balacing condition
        regularizer = (1.0 - y_dependent - y_independent).mean().pow(2)
        loss = loss + self._gamma * regularizer

        return loss


class ConservativeEqualityCriterion(RegularizedCriterion):

    def __init__(self,
        estimator,
        batch_size=h.default.batch_size,
        gamma=1.0,
        logits=False, **kwargs):
        super(ConservativeEqualityCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            gamma=gamma,
            logits=logits,
            **kwargs)

    def _forward_without_logits(self, **kwargs):
        # Forward passes
        y_joint, _ = self._estimator(**kwargs)
        ## Shuffle to make necessary variables independent.
        for group in self._independent_random_variables:
            random_indices = torch.randperm(self._batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make group independent.
        y_marginals, log_r_marginals = self._estimator(**kwargs)
        # Compute losses
        loss_joint_1 = self._criterion(y_joint, self._ones)
        loss_marginals_0 = self._criterion(y_marginals, self._zeros)
        # Learn mixture of the joint vs. marginals
        loss = loss_joint_1 + loss_marginals_0
        regularizer = (1 - log_r_marginals.exp().mean()).pow(2)
        loss = loss + self._gamma * regularizer

        return loss

    def _forward_with_logits(self, **kwargs):
        # Forward passes
        _, log_r_joint = self._estimator(**kwargs)
        ## Shuffle to make necessary variables independent.
        for group in self._independent_random_variables:
            random_indices = torch.randperm(self._batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make group independent.
        _, log_r_marginals = self._estimator(**kwargs)
        # Compute losses
        loss_joint_1 = self._criterion(log_r_joint, self._ones)
        loss_marginals_0 = self._criterion(log_r_marginals, self._zeros)
        # Learn mixture of the joint vs. marginals
        loss = loss_joint_1 + loss_marginals_0
        regularizer = (1 - log_r_marginals.exp().mean()).pow(2)
        loss = loss + self._gamma * regularizer

        return loss


class VariationalInferenceCriterion(BaseCriterion):

    def __init__(self,
        estimator,
        batch_size=h.default.batch_size,
        logits=False,
        dataset_train_size=None,
        **kwargs):
        super(VariationalInferenceCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            logits=logits,
            **kwargs)

        self._dataset_train_size = dataset_train_size

    def _forward_without_logits(self, **kwargs):
        # Forward passes
        y_joint, log_r_joint = self._estimator(**kwargs)
        effective_batch_size = len(kwargs[self._independent_random_variables[0][0]])
        ## Shuffle to make necessary variables independent.
        for group in self._independent_random_variables:
            random_indices = torch.randperm(effective_batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make group independent.
        y_marginals, log_r_marginals = self._estimator(**kwargs)

        data_log_likelihood = (torch.log(y_joint).mean() + torch.log(1 - y_marginals).mean()) * self._dataset_train_size
        kl_weight_prior = self._estimator.kl_loss()
        loss = kl_weight_prior - data_log_likelihood

        return loss

    def _forward_with_logits(self, **kwargs):
        raise NotImplementedError()


class VariationalInferenceCriterionNoKL(BaseCriterion):

    def __init__(self,
        estimator,
        batch_size=h.default.batch_size,
        logits=False,
        dataset_train_size=None,
        **kwargs):
        super(VariationalInferenceCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            logits=logits,
            **kwargs)

        self._dataset_train_size = dataset_train_size

    def _forward_without_logits(self, **kwargs):
        # Forward passes
        y_joint, log_r_joint = self._estimator(**kwargs)
        effective_batch_size = len(kwargs[self._independent_random_variables[0][0]])
        ## Shuffle to make necessary variables independent.
        for group in self._independent_random_variables:
            random_indices = torch.randperm(effective_batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make group independent.
        y_marginals, log_r_marginals = self._estimator(**kwargs)

        data_log_likelihood = (torch.log(y_joint).mean() + torch.log(1 - y_marginals).mean()) * self._dataset_train_size
        loss = -data_log_likelihood

        return loss

    def _forward_with_logits(self, **kwargs):
        raise NotImplementedError()

class KLCriterion(BaseCriterion):
    def __init__(self,
        estimator,
        batch_size=h.default.batch_size,
        logits=False, **kwargs):
        super(KLCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            logits=logits,
            **kwargs)

    def _forward_without_logits(self, **kwargs):
        log_posterior_joint = self._estimator.log_posterior(**kwargs)
        loss = -log_posterior_joint.mean()

        return loss

class KLBalancedCriterion(BaseCriterion):
    def __init__(self,
        estimator,
        batch_size=h.default.batch_size,
        gamma=100.0,
        logits=False, **kwargs):
        super(KLBalancedCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            logits=logits,
            **kwargs)
        self._gamma = gamma

    def _forward_without_logits(self, **kwargs):
        log_posterior_joint, y_joint = self._estimator.log_posterior_with_classifier(**kwargs)

        effective_batch_size = len(kwargs[self._independent_random_variables[0][0]])
        ## Shuffle to make necessary variables independent.
        for group in self._independent_random_variables:
            random_indices = torch.randperm(effective_batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make group independent.

        log_posterior_marginal, y_marginal = self._estimator.log_posterior_with_classifier(**kwargs)
        loss = -log_posterior_joint.mean()

        # Balacing condition
        regularizer = (1.0 - y_joint - y_marginal).mean().pow(2)
        loss = loss + self._gamma * regularizer

        return loss

    def _forward_with_logits(self, **kwargs):
        raise NotImplementedError()
import hypothesis as h
import numpy as np
import torch

from hypothesis.nn.ratio_estimation import BaseCriterion


class RegularizedCriterion(BaseCriterion):

    def __init__(self,
        estimator,
        batch_size=h.default.batch_size,
        gamma=10.0,
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
        gamma=10.0,
        logits=False, **kwargs):
        super(BalancedCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            gamma=gamma,
            logits=logits,
            **kwargs)

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
        regularizer = (1.0 - y_joint.mean() - y_marginals.mean()).pow(2)
        loss = loss + self._gamma * regularizer

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
        regularizer = (1.0 - y_joint.mean() - y_marginals.mean()).pow(2)
        loss = loss + self._gamma * regularizer

        return loss


class ConservativeRectifiedCriterion(RegularizedCriterion):

    def __init__(self,
        estimator,
        batch_size=h.default.batch_size,
        gamma=10.0,
        logits=False, **kwargs):
        super(ConservativeRectifiedCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            gamma=gamma,
            logits=logits,
            **kwargs)

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
        regularizer = torch.clamp((1.0 - log_r_marginals.exp()).mean(), max=0.0).pow(2)
        loss = loss + self._gamma * regularizer

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
        regularizer = torch.clamp((1.0 - log_r_marginals.exp()).mean(), max=0.0).pow(2)
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
                kwargs[variable] = kwargs[variable][random_indices]  # Make variable independent.
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
                kwargs[variable] = kwargs[variable][random_indices]  # Make variable independent.
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
        logits=False, **kwargs):
        super(VariationalInferenceCriterion, self).__init__(
            estimator=estimator,
            batch_size=batch_size,
            logits=logits,
            **kwargs)

    def _forward_without_logits(self, **kwargs):
        # Forward passes
        y_joint, log_r_joint = self._estimator(**kwargs)
        ## Shuffle to make necessary variables independent.
        for group in self._independent_random_variables:
            random_indices = torch.randperm(self._batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices]  # Make variable independent.
        y_marginals, log_r_marginals = self._estimator(**kwargs)

        data_log_likelihood = torch.log(y_joint).sum() + torch.log(1 - y_marginals).sum()
        kl_weight_prior = self._estimator.kl_loss()
        loss = kl_weight_prior - data_log_likelihood

        return loss
        

    def _forward_with_logits(self, **kwargs):
        raise NotImplementedError()

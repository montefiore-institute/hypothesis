import hypothesis
import hypothesis.nn
import torch

from .base import BaseCriterion
from .base import BaseRatioEstimator



class LikelihoodToEvidenceCriterion(BaseCriterion):

    DENOMINATOR = "inputs|outputs"

    def __init__(self,
        estimator,
        batch_size=hypothesis.default.batch_size,
        logits=False):
        super(LikelihoodToEvidenceCriterion, self).__init__(
            batch_size=batch_size,
            denominator=LikelihoodToEvidenceCriterion.DENOMINATOR,
            estimator=estimator,
            logits=logits)



class ConservativeLikelihoodToEvidenceCriterion(LikelihoodToEvidenceCriterion):

    def __init__(self,
        estimator,
        alpha=1.0,
        batch_size=hypothesis.default.batch_size,
        logits=False):
        super(ConservativeLikelihoodToEvidenceCriterion, self).__init__(
            batch_size=batch_size,
            estimator=estimator,
            logits=logits)
        self.alpha = alpha

    def _forward_without_logits(self, **kwargs):
        alpha = self.alpha
        y_dependent, log_ratios_dependent = self.estimator(**kwargs)
        for group in self.independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices] # Make variable independent.
        y_independent, _ = self.estimator(**kwargs)
        loss = (alpha * self.criterion(y_dependent, self.ones) + (1 - alpha) * self.criterion(y_independent, self.ones)) + self.criterion(y_independent, self.zeros)

        return loss

    def _forward_with_logits(self, **kwargs):
        alpha = self.alpha
        y_dependent = self.estimator.log_ratio(**kwargs)
        for group in self.independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices] # Make variable independent.
        y_independent = self.estimator.log_ratio(**kwargs)
        loss = (alpha * self.criterion(y_dependent, self.ones) + (1 - alpha) * self.criterion(y_independent, self.ones)) + self.criterion(y_independent, self.zeros)

        return loss



class BaseLikelihoodToEvidenceRatioEstimator(BaseRatioEstimator):

    def __init__(self):
        super(BaseLikelihoodToEvidenceRatioEstimator, self).__init__()

    def forward(self, inputs, outputs):
        log_ratios = self.log_ratio(inputs=inputs, outputs=outputs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, inputs, outputs):
        raise NotImplementedError

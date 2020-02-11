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



class LikelihoodToEvidenceSwappingCriterion(BaseCriterion):

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

    def _forward_without_logits(self, inputs, outputs):
        # A pass
        y_dependent, _ = self.ratio_estimator(inputs, outputs) # Dependent samples
        outputs_random = outputs[torch.randperm(self.batch_size)].detach()
        y_independent, _ = self.ratio_estimator(inputs, outputs_random) # Independent samples
        loss_a = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)
        # B pass
        y_dependent, _ = self.ratio_estimator(inputs, outputs) # Dependent samples
        inputs_random = inputs[torch.randperm(self.batch_size)].detach()
        y_independent, _ = self.ratio_estimator(inputs_random, outputs) # Independent samples
        loss_b = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)
        # Combined
        loss = (loss_a + loss_b) / 2

        return loss

    def _forward_with_logits(self, inputs, outputs):
        # A pass
        _, y_dependent = self.ratio_estimator(inputs, outputs) # Dependent samples
        outputs_random = outputs[torch.randperm(self.batch_size)].detach()
        _, y_independent = self.ratio_estimator(inputs, outputs_random) # Independent samples
        loss_a = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)
        # B pass
        _, y_dependent = self.ratio_estimator(inputs, outputs) # Dependent samples
        inputs_random = inputs[torch.randperm(self.batch_size)].detach()
        _, y_independent = self.ratio_estimator(inputs_random, outputs) # Independent samples
        loss_b = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)
        # Combined
        loss = (loss_a + loss_b) / 2

        return loss



class BaseLikelihoodToEvidenceRatioEstimator(BaseRatioEstimator):

    def __init__(self):
        super(BaseLikelihoodToEvidenceRatioEstimator, self).__init__()

    def forward(self, inputs, outputs):
        log_ratios = self.log_ratio(inputs=inputs, outputs=outputs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, inputs, outputs):
        raise NotImplementedError

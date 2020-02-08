import hypothesis
import hypothesis.nn
import torch

from .base import BaseCriterion
from .base import BaseAmortizedRatioEstimator



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



class BaseLikelihoodToEvidenceAmortizedRatioEstimator(BaseAmortizedRatioEstimator):

    def __init__(self):
        super(LikelihoodToEvidenceAmortizedRatioEstimator, self).__init__()

    def forward(self, inputs, outputs):
        log_ratios = self.log_ratio(inputs=inputs, outputs=outputs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, inputs, outputs):
        raise NotImplementedError

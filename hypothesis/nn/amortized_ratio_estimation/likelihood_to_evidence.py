import hypothesis
import hypothesis.nn
import torch

from .base import BaseCriterion
from .base import BaseConservativeCriterion
from .base import BaseRatioEstimator



DENOMINATOR = "inputs|outputs"



class LikelihoodToEvidenceCriterion(BaseCriterion):

    def __init__(self,
        estimator,
        batch_size=hypothesis.default.batch_size,
        logits=False):
        super(LikelihoodToEvidenceCriterion, self).__init__(
            batch_size=batch_size,
            denominator=DENOMINATOR,
            estimator=estimator,
            logits=logits)



class ConservativeLikelihoodToEvidenceCriterion(BaseConservativeCriterion):

    def __init__(self,
        estimator,
        beta=0.001,
        batch_size=hypothesis.default.batch_size,
        logits=False):
        super(ConservativeLikelihoodToEvidenceCriterion, self).__init__(
            batch_size=batch_size,
            denominator=DENOMINATOR,
            estimator=estimator,
            logits=logits)



class BaseLikelihoodToEvidenceRatioEstimator(BaseRatioEstimator):

    def __init__(self):
        super(BaseLikelihoodToEvidenceRatioEstimator, self).__init__()

    def forward(self, inputs, outputs):
        log_ratios = self.log_ratio(inputs=inputs, outputs=outputs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, inputs, outputs):
        raise NotImplementedError

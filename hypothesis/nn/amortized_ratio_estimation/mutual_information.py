import hypothesis
import hypothesis.nn
import torch

from .base import BaseCriterion
from .base import BaseRatioEstimator



class MutualInformationCriterion(BaseCriterion):

    DENOMINATOR = "x|y"

    def __init__(self,
        estimator,
        batch_size=hypothesis.default.batch_size,
        logits=False):
        super(LikelihoodToEvidenceCriterion, self).__init__(
            batch_size=batch_size,
            denominator=LikelihoodToEvidenceCriterion.DENOMINATOR,
            estimator=estimator,
            logits=logits)



class BaseMutualInformationRatioEstimator(BaseRatioEstimator):

    def __init__(self):
        super(LikelihoodToEvidenceRatioEstimator, self).__init__()

    def forward(self, x, y):
        log_ratios = self.log_ratio(x=x, y=y)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, x, y):
        raise NotImplementedError

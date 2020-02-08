import hypothesis
import hypothesis.nn
import torch

from .base import BaseCriterion



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

import hypothesis
import torch

from hypothesis.exception import NotDivisibleByTwoException



class ConditionalRatioEstimator(object):

    def forward(self, inputs, outputs):
        r""""""
        raise NotImplementedError

    def log_ratio(self, inputs, outputs):
        r""""""
        raise NotImplementedError



class BaseConditionalRatioEstimator(torch.nn.Module, ConditionalRatioEstimator):
    r""""""

    def __init__(self):
        ConditionalRatioEstimator.__init__(self)
        torch.nn.Module.__init__(self)

    def forward(self, inputs, outputs):
        r""""""
        raise NotImplementedError

    def log_ratio(self, inputs, outputs):
        r""""""
        raise NotImplementedError



class ConditionalRatioEstimatorEnsemble(BaseConditionalRatioEstimator):
    r""""""

    def __init__(self, ratio_estimators, reduce="mean"):
        super(ConditionalRatioEstimatorEnsemble, self).__init__()
        self.ratio_estimators = ratio_estimators
        self.reduce = self._allocate_reduce(reduce)

    def forward(self, inputs, outputs):
        log_ratios = self.log_ratio(inputs, outputs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, inputs, outputs, reduce=True):
        log_ratios = []
        for ratio_estimator in self.ratio_estimators:
            log_ratios.append(ratio_estimator.log_ratio(inputs, outputs))
        log_ratios = torch.cat(log_ratios, dim=1)
        if reduce:
            log_ratios = self.reduce(log_ratios)

        return log_ratios

    @staticmethod
    def _allocate_reduce(f):
        reductions = {
            "mean": ConditionalRatioEstimatorEnsemble._reduce_mean,
            "median": ConditionalRatioEstimatorEnsemble._reduce_median}
        reduce = None
        if hasattr(f, "__call__"):
            return f
        else:
            return reductions[f]

    @staticmethod
    def _reduce_mean(ratios):
        return ratios.mean(dim=1)

    @staticmethod
    def _reduce_median(ratios):
        return ratios.median(dim=1).values



class ConditionalRatioEstimatorCriterion(torch.nn.Module):
    r""""""

    def __init__(self, ratio_estimator, batch_size):
        super(ConditionalRatioEstimatorCriterion, self).__init__()
        # Check if a valid batch size has been supplied.
        if batch_size % 2 != 0:
            raise NotDivisibleByTwoException(
                "Only batch sizes divisible by two are permitted.")
        assert(batch_size % 2 == 0)
        self.chunked_batch_size = batch_size // 2
        self.criterion = torch.nn.BCELoss()
        self.ratio_estimator = ratio_estimator
        self.ones = torch.ones(self.chunked_batch_size, 1)
        self.zeros = torch.zeros(self.chunked_batch_size, 1)

    def _compute_loss(self, inputs, outputs):
        inputs = inputs.chunk(2)
        outputs = outputs.chunk(2)
        inputs_a = inputs[0]
        inputs_b = inputs[1]
        outputs_a = outputs[0]
        outputs_b = outputs[1]
        y_dependent_a, _ = self.ratio_estimator(inputs_a, outputs_a)
        y_independent_a, _ = self.ratio_estimator(inputs_a, outputs_b)
        y_dependent_b, _ = self.ratio_estimator(inputs_b, outputs_b)
        y_independent_b, _ = self.ratio_estimator(inputs_b, outputs_a)
        loss_a = self.criterion(y_dependent_a, self.ones) + \
                 self.criterion(y_independent_a, self.zeros)
        loss_b = self.criterion(y_dependent_b, self.ones) + \
                 self.criterion(y_independent_b, self.zeros)
        loss = loss_a + loss_b

        return loss

    def to(self, device):
        self.criterion = self.criterion.to(device)
        self.ones = self.ones.to(device)
        self.zeros = self.zeros.to(device)

        return self

    def forward(self, inputs, outputs):
        return self._compute_loss(inputs, outputs)



class ConditionalRatioEstimatorLogitsCriterion(ConditionalRatioEstimatorCriterion):

    def __init__(self, ratio_estimator, batch_size):
        super(ConditionalRatioEstimatorLogitsCriterion, self).__init__(ratio_estimator, batch_size)
        self.criterion = torch.nn.BCEWtihLogitsLoss()

    def _compute_loss(self, inputs, outputs):
        inputs = inputs.chunk(2)
        outputs = outputs.chunk(2)
        inputs_a = inputs[0]
        inputs_b = inputs[1]
        outputs_a = outputs[0]
        outputs_b = outputs[1]
        y_dependent_a = self.ratio_estimator.log_ratio(inputs_a, outputs_a)
        y_independent_a = self.ratio_estimator.log_ratio(inputs_a, outputs_b)
        y_dependent_b = self.ratio_estimator.log_ratio(inputs_b, outputs_b)
        y_independent_b = self.ratio_estimator.log_ratio(inputs_b, outputs_a)
        loss_a = self.criterion(y_dependent_a, self.ones) + \
                 self.criterion(y_independent_a, self.zeros)
        loss_b = self.criterion(y_dependent_b, self.ones) + \
                 self.criterion(y_independent_b, self.zeros)
        loss = loss_a + loss_b

        return loss

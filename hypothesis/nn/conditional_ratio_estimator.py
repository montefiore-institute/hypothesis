import hypothesis
import torch

from hypothesis.exception import NotDivisibleByTwoException
from hypothesis.nn.neuromodulation import BaseNeuromodulatedModule



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
        log_ratios = self.log_ratios(inputs, outputs)

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



class NeuromodulatedConditionalRatioEstimator(BaseConditionalRatioEstimator):

    def __init__(self, ratio_estimator):
        super(ModulatedConditionalRatioEstimator, self).__init__()
        self.ratio_estimator = ratio_estimator
        self.neuromodulated_modules = self._find_modulated_modules()
        if len(self.neuromodulated_modules) == 0:
            raise ValueError("No neuromodulated modules have been found!")

    def _find_neuromodulated_modules(self):
        modules = []

        for module in self.ratio_estimator.modules():
            if isinstance(module, BaseNeuromodulatedModule):
                modules.append(modules)

        return modules

    def forward(self, inputs, outputs):
        log_ratios = self.log_ratio(inputs, outputs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, inputs, outputs):
        # Update the context of the neuromodulated modules.
        for module in self.neuromodulated_modules:
            module.update(inputs)

        return self.ratio_estimator.log_ratio(inputs, outputs)



class ConditionalRatioEstimatorCriterion(torch.nn.Module):
    r""""""

    def __init__(self, ratio_estimator, batch_size, gamma=0.0):
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
        self.gamma = float(gamma)
        if self.gamma > 0:
            self.compute_loss = self._compute_loss_with_density_constraint
        else:
            self.compute_loss = self._compute_loss

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

    def _compute_loss_with_density_constraint(self, inputs, outputs):
        raise NotImplementedError

    def to(self, device):
        self.criterion = self.criterion.to(device)
        self.ones = self.ones.to(device)
        self.zeros = self.zeros.to(device)

        return self

    def forward(self, inputs, outputs):
        return self.compute_loss(inputs, outputs)



class ConditionalRatioEstimatorLogitsCriterion(ConditionalRatioEstimatorCriterion):

    def __init__(self, ratio_estimator, batch_size, gamma=0.0):
        super(ConditionalRatioEstimatorLogitsCriterion, self).__init__(ratio_estimator, batch_size, gamma)
        self.criterion = torch.nn.BCEWtihLogitsLoss()

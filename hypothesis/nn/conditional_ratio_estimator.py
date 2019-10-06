import torch

from hypothesis.exception import NotDivisibleByTwoException



class ConditionalRatioEstimator(object):

    def forward(self, xs, ys):
        r""""""
        raise NotImplementedError

    def log_ratio(self, xs, ys):
        r""""""
        raise NotImplementedError



class BaseConditionalRatioEstimator(torch.nn.Module, ConditionalRatioEstimator):
    r""""""

    def __init__(self):
        ConditionalRatioEstimator.__init__(self)
        torch.nn.Module.__init__(self)

    def forward(self, xs, ys):
        r""""""
        raise NotImplementedError

    def log_ratio(self, xs, ys):
        r""""""
        raise NotImplementedError



class ConditionalRatioEstimatorCriterion(torch.nn.Module):

    def __init__(self, ratio_estimator, batch_size):
        super(ConditionalRatioEstimatorCriterion, self).__init__()
        # Check if a valid batch size has been supplied.
        if batch_size % 2 != 0:
            raise NotDivisibleByTwoException("Only batch sizes divisible by two are permitted.")
        assert(batch_size % 2 == 0)
        self.chunked_batch_size = batch_size // 2
        self.criterion = torch.nn.BCELoss()
        self.ratio_estimator = ratio_estimator
        self.ones = torch.ones(self.chunked_batch_size, 1)
        self.zeros = torch.zeros(self.chunked_batch_size, 1)

    def forward(self, inputs, outputs):
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



class ConditionalRatioEstimatorLogitsCriterion(torch.nn.Module):

    def __init__(self, ratio_estimator, batch_size):
        super(ConditionalRatioEstimatorLogitsCriterion, self).__init__()
        # Check if a valid batch size has been supplied.
        if batch_size % 2 != 0:
            raise NotDivisibleByTwoException("Only batch sizes divisible by two are permitted.")
        self.chunked_batch_size = batch_size // 2
        self.ratio_estimator = ratio_estimator
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.ratio_estimator = ratio_estimator
        self.ones = torch.ones(self.chunked_batch_size, 1)
        self.zeros = torch.zeros(self.chunked_batch_size, 1)

    def forward(self, inputs, outputs):
        inputs = inputs.chunk(2)
        outputs = outputs.chunk(2)
        inputs_a = inputs[0]
        inputs_b = inputs[1]
        outputs_a = outputs[0]
        outputs_b = outputs[1]
        _, y_dependent_a = self.ratio_estimator(inputs_a, outputs_a)
        _, y_independent_a = self.ratio_estimator(inputs_a, outputs_b)
        _, y_dependent_b = self.ratio_estimator(inputs_b, outputs_b)
        _, y_independent_b = self.ratio_estimator(inputs_b, outputs_a)
        loss_a = self.criterion(y_dependent_a, self.ones) + \
                 self.criterion(y_independent_a, self.zeros)
        loss_b = self.criterion(y_dependent_b, self.ones) + \
                 self.criterion(y_independent_b, self.zeros)
        loss = loss_a + loss_b

        return loss

import torch



class ConditionalRatioEstimator(torch.nn.Module):
    r""""""

    def forward(self, inputs, outputs):
        r""""""
        raise NotImplementedError

    def log_ratio(self, inputs, outputs):
        r""""""
        raise NotImplementedError

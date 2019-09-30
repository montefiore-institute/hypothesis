import torch



class ConditionalRatioEstimator(torch.nn.Module):
    r""""""

    def __init__(self):
        super(ConditionalRatioEstimator, self).__init__()

    def forward(self, xs, ys):
        r""""""
        raise NotImplementedError

    def log_ratio(self, xs, ys):
        r""""""
        raise NotImplementedError

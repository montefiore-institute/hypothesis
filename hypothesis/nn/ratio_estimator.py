import torch



class RatioEstimator(object):

    def forward(self, xs):
        r""""""
        raise NotImplementedError

    def log_ratio(self, xs):
        r""""""
        raise NotImplementedError



class BaseRatioEstimator(torch.nn.Module, RatioEstimator):
    r""""""

    def __init__(self):
        super(RatioEstimator, self).__init__()

    def forward(self, xs):
        r""""""
        raise NotImplementedError

    def log_ratio(self, xs):
        r""""""
        raise NotImplementedError

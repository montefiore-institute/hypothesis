"""
Sampler base.
"""

from cag.engine import Module



class Sampler(Module):

    def __init__(self):
        super(Sampler, self).__init__()

    def sample(self):
        raise NotImplementedError

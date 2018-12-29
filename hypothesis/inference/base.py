import numpy as np
import torch



class Method:

    def infer(self, observations, **kwargs):
        raise NotImplementedError

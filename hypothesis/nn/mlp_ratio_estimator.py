import torch

from hypothesis.nn import BaseRatioEstimator
from hypothesis.nn import MultiLayerPerceptron as MLP



class MLPRatioEstimator(BaseRatioEstimator):
    r""""""

    def __init__(self, shape_xs, layers=(128, 128), activation=torch.nn.ELU):
        super(MLPRatioEstimator, self).__init__()
        self.dimensionality = 1
        for shape_element in shape_xs:
            self.dimensionality *= shape_element
        self.mlp = MLP(shape_xs=(self.dimensionality,), ys=(1,),
            layers=layers, activation=activation, normalize=False)

    def forward(self, xs):
        log_ratio = self.log_ratio(xs)

        return log_ratio.sigmoid(), log_ratio


    def log_ratio(self, xs):
        xs = xs.view(-1, self.dimensionality)

        return self.mlp(xs)

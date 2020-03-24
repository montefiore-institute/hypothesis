import hypothesis
import torch

from .simulator import NormalSimulator as Simulator
from .util import Prior
from .util import PriorExperiment
from hypothesis.benchmark import BenchmarkEnvironment



class Environment(BenchmarkEnvironment):

    def __init__(self, entropy_estimator,
        max_experiments=10,
        truth=None):
        super(Environment, self).__init__(
            entropy_estimator=entropy_estimator,
            max_experiments=max_experiments,
            prior=Prior(),
            prior_experiment=PriorExperiment(),
            simulator=Simulator(),
            truth=truth)

import hypothesis
import torch

from .simulator import NormalSimulator as Simulator
from .util import Prior
from .util import PriorExperiment
from hypothesis.simulation import Environment as BaseEnvironment



class Environment(BaseEnvironment):

    def __init__(self, entropy_estimator, max_experiments=10, truth=None):
        super(Environment, self).__init__()
        # Check if an entropy estimator has been specified.
        if entropy_estimator is None:
            raise ValueError("No entropy-estimator has been specified.")
        # Environment properties
        self.conducted_experiments = 0
        self.entropy_estimator = entropy_estimator
        self.max_experiments = 10
        self.predefined_truth = truth
        self.prior = Prior()
        self.prior_experiment = PriorExperiment()
        self.simulator = Simulator()
        # State
        self.actions = None
        self.observations = None
        self.rewards = None
        self.truth = None
        self.reset()

    def _perform_experiment(self, experiment):
        inputs = self.truth
        designs = experiment.view(-1, 1)
        observations = self.simulator(inputs=inputs, experimental_configurations=designs).view(-1, 1)

        return observations

    def _reward(self):
        return self.entropy_estimator(self.actions, self.observations)

    def summary(self):
        return {
            "experiments": self.actions,
            "observations": self.observations,
            "rewards": self.rewards,
            "truth": self.truth.item()}

    def step(self, action):
        assert(self.conducted_experiments < self.max_experiments)
        observation = self._perform_experiment(action)
        self.conducted_experiments += 1
        self.observations.append(observation.item())
        self.actions.append(action.item())
        reward = self._reward()
        self.rewards.append(reward)
        done = (self.conducted_experiments >= self.max_experiments)

        return observation, reward, done, self.summary()

    @torch.no_grad()
    def reset(self):
        self.actions = []
        self.conducted_experiments = 0
        self.observations = []
        self.rewards = []
        if self.predefined_truth is None:
            self.truth = self.prior.sample().view(-1, 1)
        else:
            self.truth = torch.tensor(self.predefined_truth).view(-1, 1)

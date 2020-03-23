import hypothesis
import torch

from .simulator import Simulator
from .util import Prior
from .util import PriorExperiment
from hypothesis.simulation import Environment as BaseEnvironment



class Environment(BaseEnvironment):

    def __init__(self, entropy_estimator, max_experiments=10):
        super(Environment, self).__init__()
        # Check if an entropy estimator has been specified.
        if entropy_estimator is None:
            raise ValueError("No entropy-estimator has been specified.")
        # Environment properties
        self.conducted_experiments = 0
        self.entropy_estimator = entropy_estimator
        self.max_experiments = 10
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
        observations = self.simulator(inputs, designs).view(-1, 1)

        return observations

    def _reward(self):
        return self.entropy_estimator(self.actions, self.observations)

    def _summary(self):
        return {
            "actions": self.actions,
            "observations": self.observations,
            "rewards": self.rewards,
            "truth": self.truth.item()}

    def step(self, action):
        assert(self.conducted_experiments < self.max_experiments)
        observation = self._perform_experiment(action)
        self.conducted_experiments += 1
        self.observations.append(observation)
        self.actions.append(action)
        reward = self._reward()
        self.rewards.append(reward)
        done = (self.conducted_experiments >= self.max_experiments)

        return observation, reward, done, self._summary()

    @torch.no_grad()
    def reset(self):
        self.actions = []
        self.conducted_experiments = 0
        self.observations = []
        self.rewards = []
        self.truth = self.prior.sample().view(-1, 1)

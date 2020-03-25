import hypothesis
import torch

from hypothesis.rl import Environment as BaseEnvironment



class BenchmarkEnvironment(BaseEnvironment):

    def __init__(self, simulator,
            prior,
            prior_experiment,
            entropy_estimator,
            max_experiments=10,
            truth=None):
        super(BenchmarkEnvironment, self).__init__()
        # Check if a simulation model has been specified
        if simulator is None:
            raise ValueError("A simulation model is required.")
        # Check if an entropy estimator has been specified.
        if entropy_estimator is None:
            raise ValueError("An entropy-estimator is required.")
        # Environment properties
        self.conducted_experiments = 0
        self.entropy_estimator = entropy_estimator
        self.max_experiments = max_experiments
        self.predefined_truth = truth
        self.prior = prior
        self.prior_experiment = prior_experiment
        self.simulator = simulator
        # Environment state
        self.reset()

    @torch.no_grad()
    def _perform_experiment(self, experiment):
        inputs = self.truth.view(1, -1)
        designs = experiment.view(1, -1)
        outputs = self.simulator(inputs=inputs, designs=designs)

        return outputs

    def _reward(self):
        return self.entropy_estimator(self.actions, self.observations)

    @torch.no_grad()
    def summary(self):
        return {
            "experiments": self.actions,
            "observations": self.observations,
            "rewards": self.rewards,
            "truth": self.truth.squeeze().numpy()}

    def step(self, action):
        assert(self.conducted_experiments < self.max_experiments)
        observation = self._perform_experiment(action)
        self.conducted_experiments += 1
        self.observations.append(observation.detach())
        self.actions.append(action.detach())
        reward = self._reward()
        self.rewards.append(reward.detach())
        done = (self.conducted_experiments >= self.max_experiments)

        return observation, reward, done, self.summary()

    @torch.no_grad()
    def reset(self):
        self.actions = []
        self.conducted_experiments = 0
        self.observations = []
        self.rewards = []
        if self.predefined_truth is None:
            self.truth = self.prior.sample().view(1, -1)
        else:
            self.truth = self.predefined_truth.view(1, -1)

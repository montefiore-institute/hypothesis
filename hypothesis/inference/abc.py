r"""Approximate Bayesian Computation"""

import hypothesis

from hypothesis.engine import Procedure



class ApproximateBayesianComputation(Procedure):
    r""""""

    def __init__(self, simulator, prior, summary, acceptor):
        super(ApproximateBayesianComputation, self).__init__()
        # Main classical ABC properties.
        self.acceptor = acceptor
        self.prior = prior
        self.simulator = simulator
        self.summary = summary
        # Sampler properties.
        self.summary_observation = None
        self.reset()

    def _register_events(self):
        # TODO Implement.
        pass

    def _draw_posterior_sample(self):
        sample = None

        while sample is None:
            prior_sample = self.prior.sample()
            x = self.simulator(prior_sample)
            s = self.summary(x)
            if self.acceptor(s, self.summary_observation):
                sample = prior_sample

        return sample

    def reset(self):
        self.summary_observation = None

    def sample(self, observation, num_samples):
        samples = []

        self.summary_observation = self.summary(observation)
        for _ in range(num_samples):
            samples.append(self._draw_posterior_sample())

        return samples

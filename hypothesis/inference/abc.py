r"""Approximate Bayesian Computation"""

import hypothesis
import torch

from hypothesis.engine import Procedure
from torch.multiprocessing import Pool



class ApproximateBayesianComputation(Procedure):
    r""""""

    def __init__(self, simulator, prior, summary, acceptor):
        super(ApproximateBayesianComputation, self).__init__()
        # Main classical ABC properties.
        self.acceptor = acceptor
        self.prior = prior
        self.simulator = simulator
        self.summary = summary

    def _draw_posterior_sample(self, summary_observation):
        sample = None

        while sample is None:
            prior_sample = self.prior.sample()
            x = self.simulator(prior_sample)
            s = self.summary(x)
            if self.acceptor(s, summary_observation):
                sample = prior_sample.unsqueeze(0)

        return sample

    def sample(self, observation, num_samples=1):
        samples = []

        summary_observation = self.summary(observation)
        for _ in range(num_samples):
            samples.append(self._draw_posterior_sample(summary_observation))
        samples = torch.cat(samples, dim=0)

        return samples



class ParallelApproximateBayesianComputation:

    def __init__(self, abc, workers=2):
        super(ParallelApproximateBayesianComputation, self).__init__()
        self.abc = abc
        self.pool = Pool(processes=workers)
        self.workers = workers

    def _prepare_arguments(self, observation, num_samples):
        arguments = []

        inputs = torch.arange(num_samples)
        num_chunks = num_samples // self.workers
        if num_chunks == 0:
            num_chunks = 1
        chunks = inputs.split(num_chunks, dim=0)
        for chunk in chunks:
            a = (self.abc, observation, len(chunk))
            arguments.append(a)

        return arguments

    def sample(self, observation, num_samples=1):
        self.abc.reset()
        arguments = self._prepare_arguments(observation, num_samples)
        outputs = self.pool.map(self._sample, arguments)
        outputs = torch.cat(outputs, dim=0)

        return outputs

    def __del__(self):
        self.pool.close()
        del self.pool
        self.pool = None

    @staticmethod
    def _sample(arguments):
        abc, observation, n = arguments

        return abc.sample(observation, num_samples=n)

r"""Markov chain Monte Carlo methods for inference.
"""

import hypothesis
import numpy as np
import torch

from hypothesis.engine import Procedure
from hypothesis.summary.mcmc import Chain
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.multiprocessing import Pool



class ParallelSampler:

    def __init__(self, sampler, chains=2, workers=torch.multiprocessing.cpu_count()):
        self.chains = chains
        self.sampler = sampler
        self.workers = workers

    def _prepare_arguments(self, observations, thetas, num_samples):
        arguments = []
        for input in inputs:
            arguments.append(self.sampler, observations, input, num_samples)

        return arguments

    def _prepare_inputs(self):
        inputs = []
        prior = self.sampler.prior
        for _ in range(self.chains):
            inputs.append(prior.sample())

        return inputs

    @torch.no_grad()
    def sample(self, observations, num_samples, thetas=None):
        assert(thetas is None or len(thetas) is self.chains)
        self.sampler.reset()
        if thetas is None:
            inputs = self._prepare_inputs()
        pool = Pool(processes=self.workers)
        arguments = self._prepare_arguments(observations, inputs, num_samples)
        chains = pool.map(self.sample_chain, arguments)
        del pool

        return chains

    @staticmethod
    def sample_chain(arguments):
        sampler, observations, input, num_samples = arguments
        chain = sampler.sample(observations, input, num_samples)

        return chain



class MarkovChainMonteCarlo(Procedure):
    r""""""

    def __init__(self, prior):
        super(MarkovChainMonteCarlo, self).__init__()
        self.prior = prior

    def _register_events(self):
        pass # No events to register.

    def _step(self, theta, observations):
        raise NotImplementedError

    def reset(self):
        pass

    @torch.no_grad()
    def sample(self, observations, input, num_samples):
        r""""""
        acceptance_probabilities = []
        acceptances = []
        samples = []
        self.reset()
        input = input.view(1, -1)
        for sample_index in range(num_samples):
            theta, acceptance_probability, acceptance = self._step(input, observations)
            input = input.view(1, -1)
            samples.append(input)
            acceptance_probabilities.append(acceptance_probability)
            acceptances.append(acceptance)
        samples = torch.cat(samples, dim=0)
        chain = Chain(samples, acceptance_probabilities, acceptances)

        return chain



class MetropolisHastings(MarkovChainMonteCarlo):
    r""""""

    def __init__(self, prior, log_likelihood, transition):
        super(MetropolisHastings, self).__init__(prior)
        self.denominator = None
        self.log_likelihood = log_likelihood
        self.transition = transition

    def _step(self, input, observations):
        accepted = False

        input_next = self.transition.sample(input)
        lnl_input_next = self.log_likelihood(input_next, observations)
        numerator = self.prior.log_prob(input_next) + lnl_input_next
        if self.denominator is None:
            lnl_input = self.log_likelihood(input, observations)
            self.denominator = self.prior.log_prob(input) + lnl_input
        acceptance_ratio = (numerator - self.denominator)
        if not self.transition.is_symmetrical():
            raise NotImplementedError
        acceptance_probability = min([1, acceptance_ratio.exp().item()])
        u = np.random.uniform()
        if u <= acceptance_probability:
            accepted = True
            input = input_next
            self.denominator = numerator

        return input, acceptance_probability, accepted

    def reset(self):
        self.denominator = None



class AALRMetropolisHastings(MarkovChainMonteCarlo):
    r"""Ammortized Approximate Likelihood Ratio Metropolis Hastings

    https://arxiv.org/abs/1903.04057
    """

    def __init__(self, prior, ratio_estimator, transition):
        super(AALRMetropolisHastings, self).__init__(prior)
        self.denominator = None
        self.prior = prior
        self.ratio_estimator = ratio_estimator
        self.transition = transition

    def _compute_ratio(self, input, outputs):
        num_observations = outputs.shape[0]
        inputs = input.repeat(num_observations, 1)
        inputs = inputs.to(hypothesis.accelerator)
        _, log_ratios = self.ratio_estimator(inputs=inputs, outputs=outputs)

        return log_ratios.sum().cpu()

    def _step(self, input, observations):
        accepted = False

        with torch.no_grad():
            input_next = self.transition.sample(input)
            lnl_input_next = self._compute_ratio(input_next, observations)
            numerator = self.prior.log_prob(input_next) + lnl_input_next
            if self.denominator is None:
                lnl_input = self._compute_ratio(input, observations)
                self.denominator = self.prior.log_prob(input) + lnl_input
            acceptance_ratio = (numerator - self.denominator)
            if not self.transition.is_symmetrical():
                raise NotImplementedError
            acceptance_probability = min([1, acceptance_ratio.exp().item()])
            u = np.random.uniform()
            if u <= acceptance_probability:
                accepted = True
                input = input_next
                self.denominator = numerator

        return input, acceptance_probability, accepted

    def reset(self):
        self.denominator = None

    @torch.no_grad()
    def sample(self, outputs, input, num_samples):
        assert(not self.ratio_estimator.training)
        outputs = outputs.to(hypothesis.accelerator)
        chain = super(AALRMetropolisHastings, self).sample(outputs, input, num_samples)

        return chain

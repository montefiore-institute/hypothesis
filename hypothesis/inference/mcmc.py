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
        for theta in thetas:
            t = (self.sampler, observations, theta, num_samples)
            arguments.append(t)

        return arguments

    def _prepare_thetas(self):
        thetas = []
        prior = self.sampler.prior
        for _ in range(self.chains):
            thetas.append(prior.sample())

        return thetas

    def sample(self, observations, num_samples, thetas=None):
        assert(thetas is None or len(thetas) is self.chains)
        self.sampler.reset()
        if thetas is None:
            thetas = self._prepare_thetas()
        pool = Pool(processes=self.workers)
        arguments = self._prepare_arguments(observations, thetas, num_samples)
        chains = pool.map(self.sample_chain, arguments)
        del pool

        return chains

    @staticmethod
    def sample_chain(arguments):
        sampler, observations, theta, num_samples = arguments
        chain = sampler.sample(observations, theta, num_samples)

        return chain



class MarkovChainMonteCarlo:
    r""""""

    def __init__(self, prior):
        super(MarkovChainMonteCarlo, self).__init__()
        self.prior = prior

    def _step(self, theta, observations):
        raise NotImplementedError

    def reset(self):
        pass

    def sample(self, observations, theta, num_samples):
        r""""""
        acceptance_probabilities = []
        acceptances = []
        samples = []
        self.reset()
        theta = theta.view(1, -1)
        for sample_index in range(num_samples):
            theta, acceptance_probability, acceptance = self._step(theta, observations)
            theta = theta.view(1, -1)
            samples.append(theta)
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

    def _step(self, theta, observations):
        accepted = False

        theta_next = self.transition.sample(theta)
        lnl_theta_next = self.log_likelihood(theta_next, observations)
        numerator = self.prior.log_prob(theta_next) + lnl_theta_next
        if self.denominator is None:
            lnl_theta = self.log_likelihood(theta, observations)
            self.denominator = self.prior.log_prob(theta) + lnl_theta
        acceptance_ratio = (numerator - self.denominator)
        if not self.transition.is_symmetrical():
            raise NotImplementedError
        acceptance_probability = min([1, acceptance_ratio.exp().item()])
        u = np.random.uniform()
        if u <= acceptance_probability:
            accepted = True
            theta = theta_next
            self.denominator = numerator

        return theta, acceptance_probability, accepted

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

    def _compute_ratio(self, theta, observations):
        num_observations = observations.shape[0]
        thetas = theta.repeat(num_observations, 1)
        _, log_ratios = self.ratio_estimator(thetas, observations)

        return log_ratios.sum().cpu()

    def _step(self, theta, observations):
        accepted = False

        with torch.no_grad():
            theta_next = self.transition.sample(theta)
            lnl_theta_next = self._compute_ratio(theta_next, observations)
            numerator = self.prior.log_prob(theta_next).sum() + lnl_theta_next
            if self.denominator is None:
                lnl_theta = self._compute_ratio(theta, observations)
                self.denominator = self.prior.log_prob(theta).sum() + lnl_theta
            acceptance_ratio = (numerator - self.denominator)
            if not self.transition.is_symmetrical():
                raise NotImplementedError
            acceptance_probability = min([1, acceptance_ratio.exp().item()])
            u = np.random.uniform()
            if u <= acceptance_probability:
                accepted = True
                theta = theta_next
                self.denominator = numerator

        return theta, acceptance_probability, accepted

    def reset(self):
        self.denominator = None

    def sample(self, observations, theta, num_samples):
        assert(not self.ratio_estimator.training)
        chain = super(AALRMetropolisHastings, self).sample(observations, theta, num_samples)

        return chain

import hypothesis
import numpy as np
import torch

from hypothesis.engine import Procedure
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal



class ApproximateBayesianComputationSequentialMonteCarlo(Procedure):
    r""""""

    def __init__(self, simulator, prior, summary, acceptor, particles=1000):
        super(ApproximateBayesianComputationSequentialMonteCarlo, self).__init__()
        # Main ABC SMC properties.
        self.acceptor = acceptor
        self.prior = prior
        self.simulator = simulator
        self.summary = summary
        self.num_particles = particles
        # Sampler state properties.
        self._reset()

    def _register_events(self):
        # TODO Implement.
        pass

    def _reset(self):
        self.covariance = None
        self.previous_covariance = None
        self.particles = []
        self.previous_particles = None
        self.weights = torch.ones(self.num_particles) / self.num_particles
        self.pertubator = None

    def _update_covariance(self):
        self.previous_covariance = self.covariance
        self.covariance = 2 * torch.from_numpy(np.cov(self.particles.numpy().T)).float()

    def _kernel_likelihood(self, particle_index):
        kernel = self.pertubator.__class__
        current_particle = self.particles[particle_index]
        previous_particle = self.previous_particles[particle_index]
        kernel = kernel(previous_particle, self.previous_covariance)
        likelihood = kernel.log_prob(current_particle).exp()

        return likelihood

    def _update_weights(self):
        # Compute the evidence.
        evidence = 0.0
        for index in range(self.num_particles):
            evidence += self.weights[index] * self._kernel_likelihood(index)
        # Compute the new weights.
        for index, particle in enumerate(self.particles):
            self.weights[index] = self.prior.log_prob(particle).exp() / evidence
        self.weights /= self.weights.sum()

    def _sample_from_prior(self, summary_observation):
        for particle_index in range(self.num_particles):
            sample = None
            while sample is None:
                prior_sample = self.prior.sample()
                x = self.simulator(prior_sample)
                s = self.summary(x)
                if self.acceptor(summary_observation, s):
                    sample = prior_sample.unsqueeze(0)
                    self.particles.append(sample)
        self.particles = torch.cat(self.particles, dim=0)
        self._update_covariance()

    def _sample_particle(self):
        indices = np.arange(self.num_particles)
        sampled_index = np.random.choice(indices, 1, p=self.weights.numpy())

        return self.particles[sampled_index]

    def _allocate_pertubator(self):
        dimensionality = self.covariance.dim()
        if dimensionality <= 1:
            pertubator = Normal(0, self.covariance)
        else:
            zeros = torch.zeros(dimensionality)
            pertubator = MultivariateNormal(zeros, covariance_matrix=self.covariance)
        self.pertubator = pertubator

    def _resample_particles(self, summary_observation):
        pertubator = self.pertubator
        self.previous_particles = self.particles.clone()
        for particle_index in range(self.num_particles):
            new_particle = None
            while new_particle is None:
                proposal = self._sample_particle()
                proposal = proposal + pertubator.sample()
                x = self.simulator(proposal)
                s = self.summary(x)
                if self.acceptor(s, summary_observation):
                    new_particle = proposal
            self.particles[particle_index, :] = proposal
        self._update_covariance()
        self._update_weights()

    def sample(self, observation, num_samples=1):
        samples = []

        # Summarize the observation.
        summary_observation = self.summary(observation)
        # Initialize the particles and set initial weights.
        self._sample_from_prior(summary_observation)
        self._allocate_pertubator()
        samples.append(self.particles)
        num_samples -= self.num_particles
        while num_samples > 0:
            self._resample_particles(summary_observation)
            num_samples -= self.num_particles
            samples.append(self.particles.clone().view(-1, 1))
        samples = torch.cat(samples, dim=0)[num_samples:]

        return samples

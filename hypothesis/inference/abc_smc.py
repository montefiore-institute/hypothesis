import hypothesis
import numpy as np
import torch

from hypothesis.engine import Procedure



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
        self.particles = []
        self.weights = []
        self.covariance = None
        self._reset()

    def _register_events(self):
        # TODO Implement.
        pass

    def _reset(self):
        self.covariance = None
        self.particles = []
        self.weights = []

    def _sample_from_prior(self, summary_observation):
        initial_weight = 1 / self.num_particles
        # Sample
        for particle_index in range(self.num_particles):
            sample = None
            while sample is None:
                prior_sample = self.prior.sample()
                x = self.simulator(prior_sample)
                s = self.summary(x)
                if self.acceptor(summary_observation, s):
                    sample = prior_sample.unsqueeze(0)
                    self.particles.append(sample)
                    self.weights.append(initial_weight)
        self.particles = torch.cat(self.particles, dim=0).numpy()
        self.covariance = np.cov(self.particles)

    def sample(self, observation, num_samples=1):
        samples = []

        # Summarize the observation.
        summary_observation = self.summary(observation)
        # Initialize the particles and set initial weights.
        self._sample_from_prior(summary_observation)
        samples.extend(self.particles)
        num_samples -= self.num_particles
        samples = torch.cat(samples, dim=0)

        return samples

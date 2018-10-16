"""
Markov Chain Monte Carlo methods for inference.
"""

import numpy as np
import torch


from cag.inference import Method



class MetropolisHastings(Method):

    def __init__(self, simulator,
                 likelihood,
                 transition,
                 summary,
                 warmup_steps=10):
        super(MetropolisHastings, self).__init__(simulator)
        self.transition = transition
        self.likelihood = likelihood
        self.summary = summary
        self._warmup_steps = int(warmup_steps)

    def _warmup(self, x, p_x):
        raise NotImplementedError

    def step(self, x, p_x):
        raise NotImplementedError

    def infer(self, x_o, initializer, num_samples):
        raise NotImplementedError


class ClassifierMetropolisHastings(Method):

    def __init__(self, simulator,
                 discriminator,
                 transition,
                 warmup_steps=10,
                 simulations=10000):
        super(ClassifierMetropolisHastings, self).__init__(simulator)
        self.discriminator = discriminator
        self.transition = transition
        self._warmup_steps = warmup_steps
        self._simulations = simulations
        self._epsilon = 10e-7
        self._o_discriminator = None

    def _warmup(self, theta):
        raise NotImplementedError

    def _reset_discriminator(self):
        # Reinitialize the weights of the discriminator.
        for p in self.discriminator.parameters():
            p.set_(.5 * torch.randn_like(p))
        # Allocate a new optimizer for the discriminator.
        self._o_discriminator = torch.optim.Adam(
            self.discriminator.parameters()
        )

    def _train_classifier(self, theta_0, theta_1):
        # Reset the current discriminator.
        self._reset_discriminator()
        raise NotImplementedError

    def step(self, x_o, theta):
        accepted = False

        while not accepted:
            theta_next = self.transition.sample(theta)
            self._train_classifier(theta, theta_next)
            likelihood_ratio = self._likelihood_ratio(x_o)
            if not self.transition.is_symmetric():
                p *= (self.transition.log_prob(theta_next, theta) / (self.transition.log_prob(theta, theta_next) + self._epsilon))
            alpha = min([1, p])
            u = np.random.uniform()
            if u <= alpha:
                theta = theta_next
                accepted = True

        return theta

    def infer(self, x_o, initializer, num_samples):
        samples = []

        # Draw a random initial sample from the initializer.
        theta = initializer.sample().detach()
        # TODO Implement.
        samples.append(theta)
        # Start the sampling procedure.
        for step in range(num_samples - 1):
            theta = self.step(x_o, theta)
            samples.append(theta)

        return torch.cat(samples, dim=0)

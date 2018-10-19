"""
Markov Chain Monte Carlo methods for inference.
"""

import numpy as np
import torch


from cag.inference import Method
from cag.util import sample
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



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


class LikelihoodFreeMetropolisHastings(Method):

    def __init__(self, simulator,
                 classifier,
                 transition,
                 criterion=torch.nn.MSELoss(),
                 epochs=10,
                 batch_size=32,
                 warmup_steps=10,
                 simulations=10000):
        super(LikelihoodFreeMetropolisHastings, self).__init__(simulator)
        self.batch_size = batch_size
        self.classifier = classifier
        self.transition = transition
        self._epochs = 1
        self._warmup_steps = warmup_steps
        self._simulations = simulations
        self._epsilon = 10e-7
        self._criterion = criterion
        self._o_classifier = None

    def _simulate(self, theta):
        theta = torch.cat([theta] * self._simulations, dim=0)
        _, x_theta = self.simulator(theta)

        return x_theta

    def _warmup(self, x_o, theta, x_theta):
        for step in range(self._warmup_steps):
            theta, x_theta = self.step(x_o, theta, x_theta)

        return theta, x_theta

    def _reset_classifier(self):
        # Reset the weights of the classifier.
        for p in self.classifier.parameters():
            torch.nn.init.xavier_uniform_(p)
        # Allocate a new optimizer.
        self._o_classifier = torch.optim.Adam(self.classifier.parameters())

    def _likelihood_ratio(x_o, t_next, x_t_next, t, x_t):
        # Reset the state of the classifier.
        self._reset_classifier()
        # Training of density ratio.
        num_batches = int(self._simulations / self.batch_size)
        real = torch.ones(self.batch_size, 1)
        fake = torch.zeros(self.batch_size, 1)
        for batch_index in range(num_batches):
            x_real = sample(x_t_next, self.batch_size)
            x_fake = sample(x_t, self.batch_size)
            y_real = self.classifier(x_real)
            y_fake = self.classifier(x_fake)
            loss = (self._criterion(y_real, real) + mse(y_fake, fake)) / 2.
            gradients = torch.autograd.grad(loss, self.classifier.parameters(), create_graph=True)
            gradient_norm = 0
            for gradient in gradients:
                gradient_norm = gradient_norm + (gradient ** 2).norm(p=1)
            gradient_norm /= len(gradients)
            self._o_classifier.zero_grad()
            loss.backward()
            self._o_classifier.step()
        # Obtaini the likelihood ratio.
        s_x = self.classifier(x_o).mean()
        lr = s_x / (1 - s_x)

        return lr

    def step(self, theta, x_theta):
        accepted = False

        while not accpeted:
            # Sample the next theta from the proposal.
            theta_next = self.transition.sample(theta)
            x_theta_next = self._simulate(theta_next)
            likelihood_ratio = self._likelihood_ratio(x_o, theta_next, x_theta_next, theta, x_theta)
            if not self.transition.is_symmetric():
                t_theta_next = self.transition.log_prob(theta_next, theta)
                t_theta = self.transition.log_prob(theta, theta_next)
                p *= (t_theta_next / (t_theta + self._epsilon))
            alpha = min([1, p])
            u = np.random.uniform()
            if u <= alpha:
                theta = theta_next
                accpeted = True

        return theta, x_theta

    def infer(self, x_o, initializer, num_samples):
        samples = []

        # Draw a random initial sample from the initializer.
        theta = initializer.sample().detach().view(-1)
        x_theta = self._simulate(theta)
        # MCMC burn-in.
        theta, x_theta = self._warmup(x_o, theta, x_theta)
        samples.append(theta)
        # Start the sampling procedure.
        for step in range(num_samples - 1):
            theta, x_theta = self.step(x_o, theta, x_theta)
            samples.append(theta)

        return torch.cat(samples, dim=0)

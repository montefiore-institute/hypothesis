"""
Markov Chain Monte Carlo methods for inference.
"""

import copy
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


class ClassifierMetropolisHastings(Method):

    def __init__(self, simulator,
                 transition,
                 classifier=torch.nn.BCELoss(),
                 warmup_steps=10):
        super(TestMetropolisHastings, self).__init__(simulator)
        self.classifier = classifier
        self.transition = transition
        self._warmup_steps = warmup_steps
        self._epsilon = 10e-7
        self._criterion = criterion

    def _warmup(self, x_o, theta):
        for step in range(self._warmup_steps):
            theta = self.step(x_o, theta)

        return theta

    def _create_test(self, theta, x_o):
        with torch.no_grad():
            thetas = torch.cat([theta.view(-1, 1)] * x_o.size(0), dim=0)
            x_in = torch.cat([thetas, x_o], dim=1)

        return x_in

    def _likelihood_ratio(self, x_o, theta_next, theta):
        x_next = torch.cat([x_o, ])
        x_current = self._create_test(x_o, theta)
        x_next = self._create_test(x_o, theta_next)
        s_x_current = self.classifier(x_current).mean()
        s_x_next = self.classifier(x_next).mean()
        r_current = s_x_current / (1 - s_x_current)
        r_next = s_x_next / (1 - s_x_next)
        r = r_next / r_current

        return r.item()

    def step(self, x_o, theta):
        # Sample the next theta from the proposal.
        theta_next = self.transition.sample(theta)
        p = self._likelihood_ratio(x_o, theta_next, theta)
        if not self.transition.is_symmetric():
            t_theta_next = self.transition.log_prob(theta_next, theta)
            t_theta = self.transition.log_prob(theta, theta_next)
            p *= (t_theta_next / (t_theta + self._epsilon))
        alpha = min([1, p])
        u = np.random.uniform()
        if u <= alpha:
            theta = theta_next

        return theta

    def infer(self, x_o, initializer, num_samples):
        samples = []

        # Draw a random initial sample from the initializer.
        theta = initializer.sample().detach().view(-1)
        x_theta = self._simulate(theta)
        # MCMC burn-in.
        theta = self._warmup(x_o, theta, x_theta)
        samples.append(theta)
        # Start the sampling procedure.
        for step in range(num_samples - 1):
            theta = self.step(x_o, theta)
            samples.append(theta)

        return torch.cat(samples, dim=0)



class LikelihoodFreeMetropolisHastings(Method):

    def __init__(self, simulator,
                 transition,
                 classifier,
                 criterion=torch.nn.BCELoss(),
                 epochs=10,
                 batch_size=32,
                 warmup_steps=10,
                 simulations=10000):
        super(LikelihoodFreeMetropolisHastings, self).__init__(simulator)
        self.batch_size = batch_size
        self.classifier = classifier
        self.transition = transition
        self._initial_classifier = copy.deepcopy(classifier)
        self._epochs = epochs
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
        # Allocate a new optimizer.
        self.classifier = copy.deepcopy(self._initial_classifier)
        self._o_classifier = torch.optim.Adam(self.classifier.parameters())

    def _likelihood_ratio(self, x_o, t_next, x_t_next, t, x_t):
        # Reset the state of the classifier.
        self._reset_classifier()
        num_batches = int(self._simulations / self.batch_size) * self._epochs
        real = torch.ones(self.batch_size, 1)
        fake = torch.zeros(self.batch_size, 1)
        # Training of density ratio of x_o and x_t_next.
        # for batch_index in range(num_batches):
        #     x_real = sample(x_o, self.batch_size)
        #     x_fake = sample(x_t_next, self.batch_size)
        #     y_real = self.classifier(x_real)
        #     y_fake = self.classifier(x_fake)
        #     loss = (self._criterion(y_real, real) + self._criterion(y_fake, fake)) / 2.
        #     self._o_classifier.zero_grad()
        #     loss.backward()
        #     self._o_classifier.step()
        # # Obtain the likelihood ratio.
        # lr_a = (self.classifier(x_o) - .5).abs()
        # self._reset_classifier()
        # # Training of density ratio of x_o and x_t.
        # for batch_index in range(num_batches):
        #     x_real = sample(x_o, self.batch_size)
        #     x_fake = sample(x_t, self.batch_size)
        #     y_real = self.classifier(x_real)
        #     y_fake = self.classifier(x_fake)
        #     loss = (self._criterion(y_real, real) + self._criterion(y_fake, fake)) / 2.
        #     self._o_classifier.zero_grad()
        #     loss.backward()
        #     self._o_classifier.step()
        # # Obtain the likelihood ratio.
        # lr_b = (self.classifier(x_o) - .5).abs()
        # lr = (lr_b.mean() / (lr_a.mean() + self._epsilon)).item()
        for batch_index in range(num_batches):
            x_real = sample(x_t_next, self.batch_size)
            x_fake = sample(x_t, self.batch_size)
            y_real = self.classifier(x_real)
            y_fake = self.classifier(x_fake)
            loss = (self._criterion(y_real, real) + self._criterion(y_fake, fake)) / 2.)
            self._o_classifier.zero_grad()
            loss.backward()
            self._o_classifier.step()
        s_x = self.classifier(x_o).mean()
        lr = s_x / (1 - s_x)

        return lr

    def step(self, x_o, theta, x_theta):
        # Sample the next theta from the proposal.
        theta_next = self.transition.sample(theta)
        x_theta_next = self._simulate(theta_next)
        p = self._likelihood_ratio(x_o, theta_next, x_theta_next, theta, x_theta)
        if not self.transition.is_symmetric():
            t_theta_next = self.transition.log_prob(theta_next, theta)
            t_theta = self.transition.log_prob(theta, theta_next)
            p *= (t_theta_next / (t_theta + self._epsilon))
        alpha = min([1, p])
        u = np.random.uniform()
        if u <= alpha:
            theta = theta_next
            x_theta = x_theta_next

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

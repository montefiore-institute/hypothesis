import hypothesis
import torch

from hypothesis.inference import Method
from hypothesis.util import epsilon
from hypothesis.util import sample



class AdversarialVariationalOptimization(Method):
    r"""Adversarial Variational Optimization

    An implementation of arxiv.org/abs/1707.07113
    """

    DEFAULT_BATCH_SIZE = 32
    KEY_STEPS = "steps"
    KEY_PROPOSAL = "proposal"
    KEY_BATCH_SIZE = "batch_size"

    def __init__(self, model,
                 discriminator,
                 gamma=10,
                 lr_discriminator=.001,
                 lr_proposal=.001,
                 baseline=None):
        super(AdversarialVariationalOptimization, self).__init__()
        self.discriminator = discriminator
        self.model = model
        if not baseline:
            baseline = AVOBaseline(discriminator)
        self.baseline = baseline
        self.lr_discriminator = lr_discriminator
        self.lr_proposal = lr_proposal
        self.gamma = float(gamma)
        self.proposal = None
        self.batch_size = self.DEFAULT_BATCH_SIZE
        self.optimizer_discriminator = None
        self.optimizer_proposal = None
        self.ones = torch.ones(self.batch_size, 1)
        self.zeros = torch.zeros(self.batch_size, 1)
        self.criterion = torch.nn.BCELoss()

    def set_criterion(self, criterion):
        self.criterion = criterion

    def reset(self):
        # Allocate the optimizers.
        self.allocate_optimizers()
        # Allocate the targets wrt the batch-size.
        self.ones = torch.ones(self.batch_size, 1)
        self.zeros = torch.zeros(self.batch_size, 1)

    def allocate_optimizers(self):
        # Discriminator optimizer.
        self.optimizer_discriminator = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=self.lr_discriminator)
        # Proposal optimizer.
        self.optimizer_proposal = torch.optim.RMSprop(
            self.proposal.parameters(), lr=self.lr_proposal)
        # Call the reset hook.
        hypothesis.call_hooks(hypothesis.hooks.reset, self)

    def update_discriminator(self, observations, thetas, x_thetas):
        x_real = sample(observations, self.batch_size)
        x_real.requires_grad = True
        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_thetas)
        loss = (self.criterion(y_real, self.ones) + self.criterion(y_fake, self.zeros)) / 2
        loss = loss + self.gamma * r1_regularization(y_real, x_real).mean()
        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()
        x_real.requires_grad = False

    def update_proposal(self, observations, thetas, x_thetas):
        # Compute the gradients of the log probabilities.
        gradients = []
        log_probabilities = self.proposal.log_prob(thetas)
        for log_p in log_probabilities:
            gradient = torch.autograd.grad(log_p, self.proposal.parameters(), create_graph=True)
            gradients.append(gradient)
        # Compute the REINFORCE gradient wrt the model parameters.
        gradient_U = []
        with torch.no_grad():
            # Allocate a buffer for all parameters in the proposal.
            for p in self.proposal.parameters():
                gradient_U.append(torch.zeros_like(p))
            # Apply a baseline for variance reduction in the theta grads.
            p_thetas = self.baseline.apply(inputs=x_thetas, gradients=gradients)
            # Compute the REINFORCE gradient.
            for index, gradient in enumerate(gradients):
                p_theta = p_thetas[index]
                for p_index, p_gradient in enumerate(gradient):
                    pg_theta = p_theta[p_index].squeeze()
                    gradient_U[p_index] += -pg_theta * p_gradient
            # Average out the REINFORCE gradient.
            for g in gradient_U:
                g /= self.batch_size
            # Set the REINFORCE gradient for the optimizer.
            for index, p in enumerate(self.proposal.parameters()):
                p.grad = gradient_U[index].expand(p.size())
        # Apply an optimization step.
        self.optimizer_proposal.step()
        self.proposal.fix()


    def sample_and_simulate(self):
        inputs = self.proposal.sample(self.batch_size)
        hypothesis.call_hooks(hypothesis.hooks.pre_simulation, self, inputs=inputs)
        outputs = self.model(inputs)
        hypothesis.call_hooks(hypothesis.hooks.post_simulation, self, inputs=inputs, outputs=outputs)

        return inputs, outputs

    def step(self, observations):
        hypothesis.call_hooks(hypothesis.hooks.pre_step, self)
        inputs, outputs = self.sample_and_simulate()
        self.update_discriminator(observations, inputs, outputs)
        self.update_proposal(observations, inputs, outputs)
        hypothesis.call_hooks(hypothesis.hooks.post_step, self)

    def infer(self, observations, **kwargs):
        # Fetch the desired number of inference steps.
        steps = int(kwargs[self.KEY_STEPS])
        # Clone the specified proposal.
        self.proposal = kwargs[self.KEY_PROPOSAL].clone()
        # Check if a custom batch-size has been specified.
        if self.KEY_BATCH_SIZE in kwargs.keys():
            self.batch_size = int(kwargs[self.KEY_BATCH_SIZE])
        self.reset()
        hypothesis.call_hooks(hypothesis.hooks.pre_inference, self)
        for step in range(steps):
            self.step(observations)
        hypothesis.call_hooks(hypothesis.hooks.post_inference, self)

        return self.proposal.clone()



class Baseline:
    r"""Abstract base class for variance reduction in AVO."""

    def apply(self, inputs, gradients):
        raise NotImplementedError



class NashBaseline(Baseline):

    def __init__(self, discriminator):
        self.discriminator = discriminator
        self.equilibrium = torch.tensor(.5).log().detach()

    def apply(self, inputs, gradients):
        baselines = []

        with torch.no_grad():
            y = (1 - self.discriminator(inputs)).log()
            b = (self.equilibrium - y)
            for g in gradients[0]:
                baselines.append(b)
            baselines = torch.cat(baselines, dim=1)

        return baselines



class MeanBaseline(Baseline):

    def __init__(self, discriminator):
        self.discriminator = discriminator

    def apply(self, inputs, gradients):
        baselines = []

        with torch.no_grad():
            y = (1 - self.discriminator(inputs)).log()
            b = (y.mean() - y)
            for g in gradients[0]:
                baselines.append(b)
            baselines = torch.cat(baselines, dim=1)

        return baselines



class AVOBaseline(Baseline):

    def __init__(self, discriminator):
        self.discriminator = discriminator

    def apply(self, inputs, gradients):
        numerators = []
        denominators = []
        num_parameters = len(gradients[0])
        batch_size = inputs.size(0)

        with torch.no_grad():
            # Initialize a buffer for every parameter.
            for g in gradients[0]:
                numerators.append(torch.zeros_like(g))
                denominators.append(torch.zeros_like(g))
            y = (1 - self.discriminator(inputs)).log()
            for index, gradient in enumerate(gradients):
                for p_index, p_gradient in enumerate(gradient):
                    p_gradient2 = p_gradient.pow(2)
                    y_theta = y[index].squeeze()
                    numerators[p_index] += p_gradient2 * y_theta
                    denominators[p_index] += p_gradient2
            b = []
            for index in range(num_parameters):
                numerators[index] /= batch_size
                denominators[index] /= batch_size
                b.append(numerators[index] / (denominators[index] + epsilon))
            baselines = []
            for index in range(batch_size):
                parameters = []
                for p_index in range(num_parameters):
                    p = (b[p_index] - y[index])
                    parameters.append(p)
                baselines.append(parameters)

        return baselines



def r1_regularization(y_hat, x):
    """R1 regularization from Mesheder et al, 2017."""
    batch_size = x.size(0)
    grad_y_hat = torch.autograd.grad(
        outputs=y_hat.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    regularizer = grad_y_hat.pow(2).view(batch_size, -1).sum()

    return regularizer

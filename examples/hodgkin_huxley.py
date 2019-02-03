import hypothesis
import torch

from hypothesis.benchmark.neuron import allocate_observations
from torch.distributions.uniform import Uniform


lower = torch.tensor([0] * 12).float()
upper = torch.tensor([1] * 12).float()
uniform = Uniform(lower, upper)
theta = uniform.sample()

allocate_observations(theta)

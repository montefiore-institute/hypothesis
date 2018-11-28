"""
Recursive Likelihood-free MCMC
"""

import argparse
import hypothesis
import numpy as np
import os
import matplotlib.pyplot as plt
import torch

from hypothesis.inference import MetropolisHastings
from hypothesis.inference import RatioMetropolisHastings
from hypothesis.io.dataset import ReferenceSimulationDataset
from hypothesis.io.dataset import SimulationDataset
from hypothesis.simulation.simulator import NormalSimulator
from hypothesis.transition import NormalTransitionDistribution
from hypothesis.util import epsilon
from hypothesis.visualization.mcmc import plot_density
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.utils.data import DataLoader
from tqdm import tqdm



def main(arguments):
    iterations = arguments.iterations
    for iteration in tqdm(range(iterations), desc="Recursions"):
        classifier, result_lf, result_analytical = approximate_posterior(arguments)
        #plot_density(result_lf, show_mean=True, truth=arguments.truth)
        #plt.show()
        #plot_density(result_analytical, show_mean=True, truth=arguments.truth)
        #plt.show()
        arguments.lower = results_lf.min() - .1
        arguments.upper = results_lf.max() + .1
        arguments.reference = (arguments.lower + arguments.upper) / 2.
        arguments.theta0 = arguments.reference
    plot_density(result_lf, show_mean=True, truth=result_analytical.mean().item())
    print(result_analytical.mean(), "---", result_lf.mean())


def approximate_posterior(arguments):
    observations = allocate_observations(arguments)
    classifier = optimize_classifier(arguments)

    def ratio(observations, theta_next, theta):
        if theta_next.item() <= arguments.lower or theta_next.item() >= arguments.upper:
            lr = 0.
        else:
            n = observations.size(0)
            theta_next = theta_next.repeat(n).view(-1, 1)
            theta = theta.repeat(n).view(-1, 1)
            x_in = torch.cat([theta, observations], dim=1)
            x_in_next = torch.cat([theta_next, observations], dim=1)
            s = classifier(x_in)
            s_next = classifier(x_in_next)
            lr = (s / (1 - s + epsilon)).log().sum()
            lr_next = (s_next / (1 - s_next + epsilon)).log().sum()
            lr = (lr_next - lr).exp().item()

        return lr

    theta_0 = torch.tensor(arguments.theta0).view(-1).float()
    transition = NormalTransitionDistribution(1.)
    # Allocate the LFI MCMC sampler.
    sampler = RatioMetropolisHastings(ratio, transition)
    result_lf = sampler.infer(
        observations,
        theta_0=theta_0,
        samples=arguments.samples,
        burnin_steps=arguments.burnin)
    sampler.terminate()

    def log_likelihood(observations, theta):
        with torch.no_grad():
            N = Normal(theta, 1.)
            log_p = N.log_prob(observations).sum()

        return log_p

    # Allocate the analytical sampler.
    sampler = MetropolisHastings(log_likelihood, transition)
    result_analytical = sampler.infer(
        observations,
        theta_0=theta_0,
        samples=arguments.samples,
        burnin_steps=arguments.burnin)
    sampler.terminate()


    return classifier, result_lf, result_analytical


def optimize_classifier(arguments):
    classifier = allocate_classifier(arguments)
    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters())
    U = Uniform(arguments.lower, arguments.upper)
    simulator = NormalSimulator()
    simulation_dataset = SimulationDataset(U, simulator, size=arguments.size)
    reference_dataset = ReferenceSimulationDataset(arguments.reference, simulator, size=arguments.size)
    iterations = int(arguments.size / arguments.batch_size)
    ones = torch.ones(arguments.batch_size, 1)
    zeros = torch.zeros(arguments.batch_size, 1)
    criterion = torch.nn.BCELoss()

    # Start the training procedure.
    for epoch in tqdm(range(arguments.epochs), desc="Epochs"):
        # Allocate the data-loader for the simulations.
        simulation_loader = DataLoader(simulation_dataset,
                                       num_workers=arguments.data_workers,
                                       batch_size=arguments.batch_size)
        # Allocate the data-loader for the reference simulations.
        reference_loader = DataLoader(reference_dataset,
                                      num_workers=arguments.data_workers,
                                      batch_size=arguments.batch_size)
        # Convert to iterators.
        simulation_loader = iter(simulation_loader)
        reference_loader = iter(reference_loader)
        # Start the training epoch.
        for iteration in tqdm(range(iterations), desc="Samples"):
            theta, x_theta = next(simulation_loader)
            _, x_theta_ref = next(reference_loader)
            in_real = torch.cat([theta, x_theta], dim=1).squeeze().detach()
            in_fake = torch.cat([theta, x_theta_ref], dim=1).squeeze().detach()
            y_real = classifier(in_real)
            y_fake = classifier(in_fake)
            loss = (criterion(y_real, ones) + criterion(y_fake, zeros)) / 2.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # End of the training procedure.
    classifier.eval()

    return classifier


def allocate_classifier(arguments):
    hidden = arguments.hidden
    classifier = torch.nn.Sequential(
        torch.nn.Linear(2, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
        torch.nn.Sigmoid())

    return classifier


def allocate_observations(arguments):
    path = "observations.th"
    if not os.path.exists(path):
        normal = Normal(arguments.truth, 1.)
        observations = normal.sample(torch.Size([arguments.observations]))
        observations = observations.view(-1, 1)
        torch.save(observations, path)
    else:
        observations = torch.load(path)

    return observations


def parse_arguments():
    parser = argparse.ArgumentParser("Recursive Likelihood-free MCMC.")
    parser.add_argument("--observations", type=int, default=100, help="Number of observations sampled from the observed distribution.")
    parser.add_argument("--truth", type=float, default=1, help="True mean of the observed distribution.")
    parser.add_argument("--reference", type=float, default=0, help="Initial reference model parameter.")
    parser.add_argument("--hidden", type=int, default=50, help="Number of hidden units in the neural units.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of recursive iterations.")
    parser.add_argument("--samples", type=int, default=100000, help="Number of MCMC samples.")
    parser.add_argument("--burnin", type=int, default=1000, help="Number of burnin steps.")
    parser.add_argument("--lower", type=float, default=-5, help="Initial lower-bound of the parameter space.")
    parser.add_argument("--upper", type=float, default=5, help="Initial upper-bound of the parameter space.")
    parser.add_argument("--size", type=int, default=256000, help="Dataset size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch-size of the optimization procedure.")
    parser.add_argument("--data-workers", type=int, default=0, help="Number of asynchronous data workers.")
    parser.add_argument("--theta0", type=float, default=5, help="Initial theta to start sampling from.")
    arguments, _ = parser.parse_known_args()

    return arguments


def create_directories():
    path = "models"
    if not os.path.exists(path):
        os.makedirs(path)
    path = "data"
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    arguments = parse_arguments()
    create_directories()
    main(arguments)

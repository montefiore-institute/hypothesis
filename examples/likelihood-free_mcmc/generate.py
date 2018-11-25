"""
MCMC posterior sample generation.
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from hypothesis.inference import RatioMetropolisHastings as MetropolisHastings
from hypothesis.util import epsilon
from torch.distributions.normal import Normal
from hypothesis.visualization.mcmc import plot_trace
from hypothesis.visualization.mcmc import plot_autocorrelation
from hypothesis.visualization.mcmc import plot_density



def main(arguments):
    method = select_method(arguments)
    if not method:
        raise ValueError("Specified method not available.")
    method(arguments)


def select_method(arguments):
    methods = {
        "mh": metropolis_hastings,
        "hmc": hamiltonian_monte_carlo
    }
    if arguments.method in methods.keys():
        method = methods[arguments.method]
    else:
        method = None

    return method


def metropolis_hastings(arguments):
    from hypothesis.transition import NormalTransitionDistribution

    classifier = load_classifier(arguments.classifier)
    observations = generate_observations(arguments)

    def ratio(observations, theta_next, theta):
        with torch.no_grad():
            n = observations.size(0)
            theta_next = theta_next.repeat(n).view(-1, 1)
            theta = theta.repeat(n).view(-1, 1)
            x_in = torch.cat([theta, observations], dim=1)
            x_in_next = torch.cat([theta_next, observations], dim=1)
            s = classifier(x_in)
            s_next = classifier(x_in_next)
            lr = s / (1 - s + epsilon)
            lr_next = s_next / (1 - s_next + epsilon)
            log_lr = lr_next.log().sum() - lr.log().sum()

        return log_lr.exp()

    theta_0 = torch.tensor(arguments.theta0).view(-1)
    transition = NormalTransitionDistribution(.1)
    sampler = MetropolisHastings(ratio, transition)
    result = sampler.infer(
        observations,
        theta_0=theta_0,
        samples=arguments.steps,
        burnin_steps=arguments.burnin)
    plot_density(result, show_mean=True, truth=arguments.truth, bins=50)
    plt.show()
    plot_trace(result, show_mean=True, truth=arguments.truth, show_burnin=True, aspect=1.)
    plt.show()
    plot_autocorrelation(result, max_lag=100, interval=1)
    plt.show()
    sampler.terminate()


def generate_observations(arguments):
    truth = arguments.truth
    n = arguments.observations
    normal = Normal(truth, 1.)
    observations = normal.sample(torch.Size([n])).view(-1, 1)

    return observations


def hamiltonian_monte_carlo(arguments):
    raise NotImplementedError


def load_classifier(path):
    classifier = torch.load(path)

    return classifier


def parse_arguments():
    parser = argparse.ArgumentParser("Likelihood-free MCMC. Posterior sampling.")
    parser.add_argument("--method", type=str, default=None, help="Sampling method (mh, hmc).")
    parser.add_argument("--classifier", type=str, default=None, help="Path to the classifier.")
    parser.add_argument("--truth", type=float, default=1., help="Defines the true theta.")
    parser.add_argument("--observations", type=int, default=100, help="Defines the number of observations.")
    parser.add_argument("--burnin", type=int, default=1000, help="Number of burnin steps of the Markov Chain.")
    parser.add_argument("--steps", type=int, default=5000, help="Number of MC steps.")
    parser.add_argument("--theta0", type=float, default=5., help="Initial theta of the Markov Chain.")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)

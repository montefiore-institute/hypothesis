import argparse
import hypothesis
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from hypothesis.inference import MetropolisHastings
from hypothesis.inference import RatioMetropolisHastings
from hypothesis.transition import NormalTransitionDistribution
from hypothesis.util import epsilon
from hypothesis.inference import ApproximateBayesianComputation as ABC
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal



def main(arguments):
    # Analytical Metropolis-Hastings.
    ensemble_size = arguments.ensemble_size
    name = "analytical-" + str(arguments.observations)
    path = "results/ensemble_" + str(ensemble_size) + "/" + name
    if not os.path.exists(path) or arguments.force:
        result_analytical = metropolis_hastings_analytical(arguments)
        save_result(ensemble_size, result_analytical, name=name)
    else:
        result_analytical = hypothesis.load(path)
    # Classifier (likelihood-free) Metropolis-Hastings.
    name = "lf-" + str(arguments.observations)
    path = "results/ensemble_" + str(ensemble_size) + "/" + name
    if not os.path.exists(path) or arguments.force:
        result_lf = metropolis_hastings_classifier(arguments)
        save_result(ensemble_size, result_lf, name=name)
    else:
        result_lf = hypothesis.load(path)

    # Plotting.
    bins = 50
    minimum = min([result_analytical.min(), result_lf.min()])
    maximum = max([result_analytical.max(), result_lf.max()])
    binwidth = abs(maximum - minimum) / bins
    bins = np.arange(minimum - binwidth, maximum + binwidth, binwidth)
    chain_analytical = result_analytical.chain(0)
    chain_lf = result_lf.chain(0)
    plt.hist(chain_analytical.numpy(), histtype="step", bins=bins, density=True, alpha=.8, label="Analytical", color="blue")
    plt.hist(chain_lf.numpy(), histtype="step", bins=bins, density=True, alpha=.8, label="Likelihood-free", color="orange")
    plt.axvline(chain_analytical.mean().item(), c="blue", lw=2, linestyle="-.", alpha=.9)
    plt.axvline(chain_lf.mean().item(), c="orange", lw=2, linestyle="-.", alpha=.9)
    plt.axvline(arguments.truth, c='r', lw=2, linestyle='-', alpha=.95, label="Truth")
    plt.minorticks_on()
    plt.legend()
    plt.savefig("plots/ensemble_" + str(ensemble_size) + "/" +str(arguments.observations) + ".pdf", bbox_inches="tight", pad_inches=0)
    plt.clf()

def save_result(ensemble_size, result, name):
    path = "results/ensemble_" + str(ensemble_size) + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + name
    hypothesis.save(result, path)


def get_observations(arguments):
    path = "observations/"
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + str(arguments.observations) + ".th"
    if not os.path.exists(path):
        N = Normal(arguments.truth, 1.)
        observations = N.sample(torch.Size([arguments.observations])).view(-1, 1)
        torch.save(observations, path)
    else:
        observations = torch.load(path)

    return observations


def metropolis_hastings_analytical(arguments):
    # Define the log-likelihood of the observations wrt theta.
    def log_likelihood(theta, observations):
        N = Normal(theta.item(), 1.)
        likelihood = N.log_prob(observations).sum().detach()
        return likelihood

    observations = get_observations(arguments)
    transition = get_transition(arguments)
    theta0 = get_theta0(arguments)
    sampler = MetropolisHastings(log_likelihood, transition)
    result = sampler.infer(observations,
                           theta_0=theta0,
                           burnin_steps=arguments.burnin,
                           samples=arguments.samples)
    sampler.terminate()

    return result


def metropolis_hastings_classifier(arguments):
    classifiers = get_classifiers(arguments)

    # Extract the approximate likelihood-ratio from the classifier.
    def ratio(observations, theta_next, theta):
        n = observations.size(0)
        theta_next = theta_next.repeat(n).view(-1, 1)
        theta = theta.repeat(n).view(-1, 1)
        x = torch.cat([theta, observations], dim=1)
        x_next = torch.cat([theta_next, observations], dim=1)

        s = 0
        s_next = 0
        for classifier in classifiers:
            s += classifier(x)
            s_next += classifier(x_next)

        s /= len(classifiers)
        s_next /= len(classifiers)

        lr = ((1 - s) / s).log().sum()
        lr_next = ((1 - s_next) / s_next).log().sum()
        lr = (lr_next - lr).exp().item()

        return lr

    observations = get_observations(arguments)
    transition = get_transition(arguments)
    theta0 = get_theta0(arguments)
    sampler = RatioMetropolisHastings(ratio, transition)
    result = sampler.infer(observations,
                           theta_0=theta0,
                           burnin_steps=arguments.burnin,
                           samples=arguments.samples)

    return result


def get_classifiers(arguments):
    classifiers = []
    for i in range(arguments.ensemble_size):
        path = "model/classifier_final_{}.th".format(i+1)
        classifier = torch.load(path)
        classifier.eval()
        classifiers.append(classifier)

    return classifiers


def get_transition(arguments):
    transition = NormalTransitionDistribution(1.)

    return transition


def get_theta0(arguments):
    theta0 = torch.tensor(arguments.theta0).float().view(-1)

    return theta0


def parse_arguments():
    parser = argparse.ArgumentParser("Likelihood-free Posterior Sampling. Demonstration 1 - Sampling.")
    parser.add_argument("--samples", type=int, default=100000, help="Number of MCMC samples.")
    parser.add_argument("--burnin", type=int, default=5000, help="Number of burnin samples.")
    parser.add_argument("--observations", type=int, default=50, help="Number of observations.")
    parser.add_argument("--upper", type=float, default=5, help="Upper-limit of the parameter space.")
    parser.add_argument("--lower", type=float, default=-5, help="Lower-limit of the parameter space.")
    parser.add_argument("--truth", type=float, default=1, help="True model parameter (theta).")
    parser.add_argument("--theta0", type=float, default=5, help="Initial theta of the Markov chain.")
    parser.add_argument("--force", type=bool, default=False, nargs='?', const=True, help="Force sampling.")
    parser.add_argument("--ensemble_size", type=int, default=2, help="The size of the ensemble.")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
    os._exit(0)

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
from hypothesis.distribution import MixtureOfNormals
from hypothesis.inference import ApproximateBayesianComputation as ABC
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform



def main(arguments):
    # Analytical Metropolis-Hastings.
    name = "analytical-" + str(arguments.observations)
    path = "results/" + name
    if not os.path.exists(path) or arguments.force:
        result_analytical = metropolis_hastings_analytical(arguments)
        save_result(result_analytical, name=name)
    else:
        result_analytical = hypothesis.load(path)
    # Classifier (likelihood-free) Metropolis-Hastings.
    name = "lf-" + str(arguments.observations)
    path = "results/" + name
    if not os.path.exists(path) or arguments.force:
        result_lf = metropolis_hastings_classifier(arguments)
        save_result(result_lf, name=name)
    else:
        result_lf = hypothesis.load(path)
    #ABC
    for epsilon in [0.05, 0.1, 0.2, 0.3]:

        name = "abc-" + str(arguments.observations) + "-epsilon-" + str(epsilon)
        path = "results/" + name
        if not os.path.exists(path) or arguments.force:
            result_abc = abc(arguments, epsilon)
            save_result(result_abc, name=name)
        else:
            result_abc = hypothesis.load(path)

        # Plotting.
        bins = 50
        minimum = min([result_analytical.min(), result_lf.min(), result_abc.min()])
        maximum = max([result_analytical.max(), result_lf.max(), result_abc.max()])
        binwidth = abs(maximum - minimum) / bins
        bins = np.arange(minimum - binwidth, maximum + binwidth, binwidth)
        chain_analytical = result_analytical.chain(0)
        chain_lf = result_lf.chain(0)
        plt.hist(chain_analytical.numpy(), histtype="step", bins=bins, density=True, alpha=.8, label="Analytical", color="blue")
        plt.hist(chain_lf.numpy(), histtype="step", bins=bins, density=True, alpha=.8, label="Likelihood-free", color="orange")
        plt.hist(result_abc.numpy(), histtype="step", bins=bins, density=True, alpha=.8, label="ABC", color="green")
        plt.axvline(chain_analytical.mean().item(), c="blue", lw=2, linestyle="-.", alpha=.9)
        plt.axvline(chain_lf.mean().item(), c="orange", lw=2, linestyle="-.", alpha=.9)
        plt.axvline(result_abc.mean().item(), c="green", lw=2, linestyle="-.", alpha=.9)
        plt.axvline(arguments.truth, c='r', lw=2, linestyle='-', alpha=.95, label="Truth")
        plt.minorticks_on()
        plt.legend()
        plt.savefig("plots/"+str(arguments.observations) + "-epsilon-" + str(epsilon) + ".pdf", bbox_inches="tight", pad_inches=0)
        plt.clf()


def save_result(result, name):
    path = "results/"
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
        modes = arguments.modes.split(",")
        modes = [Normal(float(x), arguments.truth) for x in modes]
        mixing_coefficients = arguments.mixing_coefficients.split(",")
        mixing_coefficients = [float(x) for x in mixing_coefficients]

        dist = MixtureOfNormals(modes, mixing_coefficients)
        observations = []
        for i in range(arguments.observations):
            observations.append(dist.sample())
        observations = torch.FloatTensor(observations).view(-1, 1)

        torch.save(observations, path)
    else:
        observations = torch.load(path)

    return observations


def metropolis_hastings_analytical(arguments):
    # Define the log-likelihood of the observations wrt theta.

    def log_likelihood(theta, observations):
        modes = arguments.modes.split(",")
        modes = [Normal(float(x), theta.item()) for x in modes]
        mixing_coefficients = arguments.mixing_coefficients.split(",")
        mixing_coefficients = [float(x) for x in mixing_coefficients]

        dist = MixtureOfNormals(modes, mixing_coefficients)
        likelihood = dist.log_prob(observations).sum()
        if torch.isnan(likelihood) or (theta > arguments.upper).any() or (theta < arguments.lower).any():
            likelihood = torch.FloatTensor([float("-Inf")])
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
    classifier = get_classifier(arguments)

    # Extract the approximate likelihood-ratio from the classifier.
    def ratio(observations, theta_next, theta):
        n = observations.size(0)
        if (theta_next > arguments.upper).any() or (theta_next < arguments.lower).any():
            return 0
        else:
            theta_next = theta_next.repeat(n).view(-1, 1)
            theta = theta.repeat(n).view(-1, 1)
            x = torch.cat([theta, observations], dim=1)
            x_next = torch.cat([theta_next, observations], dim=1)
            s = classifier(x)
            s_next = classifier(x_next)
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


def abc(arguments, epsilon):

    def forward_model(theta, arguments=arguments):
        with torch.no_grad():
            modes = arguments.modes.split(",")
            modes = [Normal(float(x), theta.item()) for x in modes]
            mixing_coefficients = arguments.mixing_coefficients.split(",")
            mixing_coefficients = [float(x) for x in mixing_coefficients]

            dist = MixtureOfNormals(modes, mixing_coefficients)
            samples = []
            for i in range(arguments.observations):
                samples.append(dist.sample())
            samples = torch.FloatTensor(samples).view(-1, 1)

        return samples

    def summary(x):
        return x.mean().detach()

    def distance(x_a, x_b):
        d = (x_a - x_b).abs()
        return d

    prior = Uniform(arguments.lower, arguments.upper)
    abc = ABC(prior, forward_model, summary, distance, epsilon=epsilon)
    samples = abc.infer(get_observations(arguments), samples=arguments.samples)
    samples = torch.tensor(samples)
    return samples


def get_classifier(arguments):
    path = arguments.classifier
    classifier = torch.load(path)
    classifier.eval()

    return classifier


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
    parser.add_argument("--truth", type=float, default=1, help="True model parameter (theta).")
    parser.add_argument("--theta0", type=float, default=5, help="Initial theta of the Markov chain.")
    parser.add_argument("--classifier", type=str, default=None, help="Path to the classifier.")
    parser.add_argument("--force", type=bool, default=False, nargs='?', const=True, help="Force sampling.")
    parser.add_argument("--modes", type=str, default="-5, 0, 5", help="Modes of the mixture.")
    parser.add_argument("--mixing-coefficients", type=str, default="0.4, 0.2, 0.4", help="Mixing coefficients for the modes.")
    parser.add_argument("--lower", type=float, default=0, help="Lower-limit of the parameter space.")
    parser.add_argument("--upper", type=float, default=10, help="Upper-limit of the parameter space.")
    arguments, _ = parser.parse_known_args()
    if arguments.classifier is None:
        raise ValueError("No classifier has been specified.")

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
    os._exit(0)

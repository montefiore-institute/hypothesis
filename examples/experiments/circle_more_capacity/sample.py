import argparse
import hypothesis
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from hypothesis.inference import MetropolisHastings
from hypothesis.inference import RatioMetropolisHastings
from hypothesis.transition import MultivariateNormalTransitionDistribution
from hypothesis.util import epsilon
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from hypothesis.benchmark import circle

def main(arguments):
    # Analytical Metropolis-Hastings.
    #name = "analytical-" + str(arguments.observations)
    #path = "results/" + name
    #if not os.path.exists(path) or arguments.force:
    #    result_analytical = metropolis_hastings_analytical(arguments)
    #    save_result(result_analytical, name=name)
    #else:
    #    result_analytical = hypothesis.load(path)
    # Classifier (likelihood-free) Metropolis-Hastings.
    name = "lf-" + str(arguments.observations)
    path = "results/" + name
    if not os.path.exists(path) or arguments.force:
        result_lf = metropolis_hastings_classifier(arguments)
        save_result(result_lf, name=name)
    else:
        result_lf = hypothesis.load(path)

    #Bounds for next iteration
    for i in range(result_lf.size()):
        print("New lower bound: {}".format(result_lf.chain(i).min()))
        print("New upper bound: {}".format(result_lf.chain(i).max()))
        print("###################")

    # Plotting.
    true_thetas = [float(x) for x in arguments.truth.split(",")]
    fig, axes  = plt.subplots(1, 3, sharey=False, figsize=(10, 2.7))
    for i in range(result_lf.size()):
        bins = 50
        ax = axes[i]
        #minimum = min([result_analytical.chain(i).min(), result_lf.chain(i).min()])
        #maximum = max([result_analytical.chain(i).max(), result_lf.chain(i).max()])
        minimum = result_lf.chain(i).min()
        maximum = result_lf.chain(i).max()
        binwidth = abs(maximum - minimum) / bins
        bins = np.arange(minimum - binwidth, maximum + binwidth, binwidth)
        #chain_analytical = result_analytical.chain(i)
        chain_lf = result_lf.chain(i)

        #ax.hist(chain_analytical.numpy(), histtype="step", bins=bins, density=True, alpha=.8, label="Analytical")
        #ax.axvline(chain_analytical.mean().item(), c="gray", lw=2, linestyle="-.", alpha=.9)
        # Likelihood-free
        ax.hist(chain_lf.numpy(), histtype="step", bins=bins, density=True, alpha=.8, label="Likelihood-free")
        ax.axvline(chain_lf.mean().item(), c="gray", lw=2, linestyle="-.", alpha=.9)
        # Truth
        ax.axvline(true_thetas[i], c='r', lw=2, linestyle='-', alpha=.95, label="Truth")
        ax.minorticks_on()
    fig.savefig(str(arguments.observations) + ".pdf", bbox_inches="tight", pad_inches=0)


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
        theta = arguments.truth.split(",")
        theta = torch.Tensor([float(x) for x in theta])
        observations = circle.allocate_observations(theta).view(-1)
        observations = observations.repeat(arguments.observations, 1)

        torch.save(observations, path)
    else:
        observations = torch.load(path)

    return observations


def metropolis_hastings_analytical(arguments):

    lower_bounds = [float(x) for x in arguments.lower.split(",")]
    upper_bounds = [float(x) for x in arguments.upper.split(",")]

    # Define the log-likelihood of the observations wrt theta.
    def log_likelihood(theta, observations):
        dist = MultivariateNormal(theta, torch.eye(arguments.dimensionality))
        likelihood = dist.log_prob(observations).sum()
        if torch.isnan(likelihood):
            likelihood = torch.FloatTensor([float("-Inf")])

        for i in range(len(lower_bounds)):
            if(theta[i] < lower_bounds[i] or theta[i] > upper_bounds[i]):
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

    lower_bounds = [float(x) for x in arguments.lower.split(",")]
    upper_bounds = [float(x) for x in arguments.upper.split(",")]
    # Extract the approximate likelihood-ratio from the classifier.
    def ratio(observations, theta_next, theta):
        n = observations.size(0)

        for i in range(len(lower_bounds)):
            if(theta_next[i] < lower_bounds[i] or theta_next[i] > upper_bounds[i]):
                return 0

        theta_next = theta_next.repeat(n).view(n, -1)
        theta = theta.repeat(n).view(n, -1)
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


def get_classifier(arguments):
    path = arguments.classifier
    classifier = torch.load(path)
    classifier.eval()

    return classifier


def get_transition(arguments):
    transition = MultivariateNormalTransitionDistribution(0.05*torch.eye(arguments.dimensionality))

    return transition


def get_theta0(arguments):
    theta0 = arguments.theta0.split(",")
    theta0 = [float(x) for x in theta0]
    theta0 = torch.tensor(theta0).float().view(-1)

    return theta0


def parse_arguments():
    parser = argparse.ArgumentParser("Likelihood-free Posterior Sampling. Demonstration 1 - Sampling.")
    parser.add_argument("--samples", type=int, default=1000000, help="Number of MCMC samples.")
    parser.add_argument("--burnin", type=int, default=10000, help="Number of burnin samples.")
    parser.add_argument("--observations", type=int, default=1, help="Number of observations.")
    parser.add_argument("--truth", type=str, default="0.5, 0, 0", help="True model parameters (theta).")
    parser.add_argument("--theta0", type=str, default="-0.75, 0, 0", help="Initial theta of the Markov chain.")
    parser.add_argument("--classifier", type=str, default="models/iteration1/classifier_final.th", help="Path to the classifier.")
    parser.add_argument("--force", type=bool, default=False, nargs='?', const=True, help="Force sampling.")
    parser.add_argument("--lower", type=str, default="-1,-1,-1", help="Lower-limit of the parameter space.")
    parser.add_argument("--upper", type=str, default="1,1,1", help="Upper-limit of the parameter space.")
    parser.add_argument("--dimensionality", type=int, default=3, help="Dimensionality of the multivariate normal.")
    arguments, _ = parser.parse_known_args()
    if arguments.classifier is None:
        raise ValueError("No classifier has been specified.")

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
    os._exit(0)

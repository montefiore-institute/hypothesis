r"""A minimal example of LFIRE (Likelihood-Free Inference By Ratio Estimation).

"""

import argparse
import hypothesis
import matplotlib.pyplot as plt
import numpy as np
import torch

from hypothesis.benchmark.normal import Simulator
from hypothesis.benchmark.normal import allocate_prior
from hypothesis.inference.lfire import LFIRE
from hypothesis.visualization.util import make_square



def main(arguments):
    simulator = Simulator()
    prior = allocate_prior()
    truth = torch.tensor(arguments.truth)
    observation = simulator(truth)
    lfire = LFIRE(
        simulator=simulator,
        prior=prior,
        simulation_batch_size=arguments.simulations)
    inputs = torch.linspace(prior.low, prior.high, arguments.posterior_resolution).view(-1, 1)
    observations = observation.repeat(arguments.posterior_resolution).view(-1, 1)
    log_ratios = lfire.log_ratios(inputs, observations)
    # Check if the results have to be shown.
    if arguments.show:
        plt.axvline(truth.numpy(), lw=2, color="C0")
        plt.minorticks_on()
        plt.plot(inputs.numpy(), log_ratios.exp().numpy(), lw=2, color="black")
        plt.title("LFIRE approximate posterior")
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$p(\theta\vert x)$")
        make_square(plt.gca())
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser("LFIRE Posterior Inference: minimal example on a tractable problem.")
    parser.add_argument("--posterior-resolution", type=int, default=100, help="Grid-resolution of the posterior (default: 100).")
    parser.add_argument("--show", action="store_true", help="Show the obtained posterior (default: false).")
    parser.add_argument("--simulations", type=int, default=100000, help="Number of simulations to draw at every evaluation of the model parameter (default: 100000).")
    parser.add_argument("--truth", type=float, default=0.0, help="Default assume truth value (default: 0).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)

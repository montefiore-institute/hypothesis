import argparse
import hypothesis
import matplotlib.pyplot as plt
import numpy as np
import os
import torch



def main(arguments):

    name = "lf-" + str(arguments.observations) + "-positive-radius-mode"
    path = "results/32x32/iteration2/" + name
    result_lf = [hypothesis.load(path)]
    result_lf.append(hypothesis.load(path.replace("positive", "negative")))

    #Bounds for next iteration
    for i in range(result_lf[0].size()):
        print("New lower bound: {}".format(min([result_lf[0].chain(i).min(), result_lf[1].chain(i).min()])))
        print("New upper bound: {}".format(max([result_lf[0].chain(i).max(), result_lf[1].chain(i).max()])))
        print("###################")

    # Plotting.
    true_thetas = [float(x) for x in arguments.truth.split(",")]
    fig, axes  = plt.subplots(1, 3, sharey=False, figsize=(10, 2.7))
    for i in range(result_lf[0].size()):
        bins = 50
        ax = axes[i]
        #minimum = min([result_analytical.chain(i).min(), result_lf.chain(i).min()])
        #maximum = max([result_analytical.chain(i).max(), result_lf.chain(i).max()])
        minimum = min([result_lf[0].chain(i).min(), result_lf[1].chain(i).min()])
        maximum = max([result_lf[0].chain(i).max(), result_lf[1].chain(i).max()])
        binwidth = abs(maximum - minimum) / bins
        bins = np.arange(minimum - binwidth, maximum + binwidth, binwidth)
        #chain_analytical = result_analytical.chain(i)
        chain_lf = result_lf[0].chain(i).numpy()
        chain_lf = np.concatenate((chain_lf, result_lf[1].chain(i).numpy()), axis=0)

        #ax.hist(chain_analytical.numpy(), histtype="step", bins=bins, density=True, alpha=.8, label="Analytical")
        #ax.axvline(chain_analytical.mean().item(), c="gray", lw=2, linestyle="-.", alpha=.9)
        # Likelihood-free
        ax.hist(chain_lf, histtype="step", bins=bins, density=True, alpha=.8, label="Likelihood-free")
        if(i == 0):
            ax.axvline(result_lf[0].chain(i).numpy().mean(), c="gray", lw=2, linestyle="-.", alpha=.9)
            ax.axvline(result_lf[1].chain(i).numpy().mean(), c="gray", lw=2, linestyle="-.", alpha=.9)
            ax.axvline(-true_thetas[i], c='r', lw=2, linestyle='-', alpha=.95, label="Truth")
        else:
            ax.axvline(chain_lf.mean(), c="gray", lw=2, linestyle="-.", alpha=.9)

        ax.axvline(true_thetas[i], c='r', lw=2, linestyle='-', alpha=.95, label="Truth")
        ax.minorticks_on()
    fig.savefig(str(arguments.observations) + ".pdf", bbox_inches="tight", pad_inches=0)


def parse_arguments():
    parser = argparse.ArgumentParser("Likelihood-free Posterior Sampling. Demonstration 1 - Sampling.")
    parser.add_argument("--samples", type=int, default=1000000, help="Number of MCMC samples.")
    parser.add_argument("--burnin", type=int, default=10000, help="Number of burnin samples.")
    parser.add_argument("--observations", type=int, default=1, help="Number of observations.")
    parser.add_argument("--truth", type=str, default="0.5, 0, 0", help="True model parameters (theta).")
    parser.add_argument("--theta0", type=str, default="0, 0, 0", help="Initial theta of the Markov chain.")
    parser.add_argument("--classifier", type=str, default=None, help="Path to the classifier.")
    parser.add_argument("--force", type=bool, default=False, nargs='?', const=True, help="Force sampling.")
    parser.add_argument("--lower", type=str, default="-0.9972356557846069, -0.06088715046644211, -0.03716924786567688", help="Lower-limit of the parameter space.")
    parser.add_argument("--upper", type=str, default="0.9964998960494995, 0.06596662104129791, 0.062344297766685486", help="Upper-limit of the parameter space.")
    parser.add_argument("--dimensionality", type=int, default=3, help="Dimensionality of the multivariate normal.")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
    os._exit(0)

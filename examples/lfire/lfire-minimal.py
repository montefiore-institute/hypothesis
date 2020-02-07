r"""A minimal example of LFIRE (Likelihood-Free Inference By Ratio Estimation).

"""

import hypothesis
import numpy
import torch

from hypothesis.benchmark.normal import Simulator
from hypothesis.benchmark.normal import allocate_prior
from hypothesis.benchmark.normal import allocate_truth
from hypothesis.inference.lfire import LFIRE



def main(arguments):
    raise NotImplementedError


def parse_arguments():
    parser = argparse.ArgumentParser("LFIRE Posterior Inference: minimal example on a tractable problem.")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)

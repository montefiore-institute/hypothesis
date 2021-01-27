r"""Utility program to train ratio estimators.

This program provides a whole range of utilities to
monitor and train ratio estimators in various ways.
All defined through command line arguments!

"""

import argparse
import hypothesis as h
import hypothesis.workflow as w
import numpy as np
import os


def main(arguments):
    pass


def load_class(full_classname):
    if full_classname is None:
        raise ValueError("The specified classname cannot be `None`.")
    module_name, class_name = full_classname.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])

    return getattr(module, class_name)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument("--data-parallel", action="store_true", help="Enable data-parallel training whenever multiple GPU's are available (default: false).")
    parser.add_argument("--disable-gpu", action="store_true", help="Disable the usage of GPU's (default: false).")
    parser.add_argument("--dont-shuffle", action="store_true", help="Do not shuffle the datasets (default: false).")
    parser.add_argument("--out", type=str, default='.', help="Output directory of the generated files (default: '.').")
    parser.add_argument("--show", action="store_true", help="Show progress of the training to stdout (default: false).")
    # Optimization settings
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256).")
    parser.add_argument("--conservativeness", type=float, default=0.0, help="Conservative term (default: 0.0).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1).")
    parser.add_argument("--logits", action="store_true", help="Use the logit-trick for the minimization criterion (default: false).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (default: 0.0).")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent data loaders (default: 4).")
    # Data settings
    parser.add_argument("--data-test", type=str, default=None, help="Full classname of the testing dataset (default: none, optional).")
    parser.add_argument("--data-test", type=str, default=None, help="Full classname of the validation dataset (default: none, optional).")
    parser.add_argument("--data-train", type=str, default=None, help="Full classname of the training dataset (default: none).")
    # Ratio estimator settings
    parser.add_argument("--estimator", type=str, default=None, help="Full classname of the ratio estimator (default: none).")
    # Learning rate scheduling
    ## Learning rate scheduling on a plateau
    parser.add_argument("--lrsched-on-plateau", action="store_true", help="Enables learning rate scheduling whenever a plateau has been detected (default: false).")
    ## Cyclic learning rate scheduling
    parser.add_argument("--lrsched-cyclic", action="store_true", help="Enables cyclic learning rate scheduling")
    # Parse the supplied arguments
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)

"""
Trains a parameterized classifier to use with the likelihood-ratio trick.
"""

import argparse
import torch
import numpy as np
import os
import re

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


data_directory = "data/"

def main(arguments):
    # Data-source preperation.
    simulation_dataset = SimulationDataset(arguments.iterations)
    reference_dataset = ReferenceDataset(arguments.iterations)
    simulation_loader = DataLoader(simulation_dataset, num_workers=1, batch_size=arguments.batch_size)
    reference_loader = DataLoader(reference_dataset, num_workers=1, batch_size=arguments.batch_size)
    # Training preperation.
    real = torch.ones(arguments.batch_size, 1)
    fake = torch.zeros(arguments.batch_size, 1)
    bce = torch.nn.BCELoss()
    iterations = int(arguments.iterations / arguments.batch_size)

    model_files = os.listdir("models_{}/".format(arguments.run))
    model_files = [x for x in model_files if x.startswith("{}_".format(arguments.hidden))]
    file_epochs = [re.search('{}_(.+?).th'.format(arguments.hidden), x).group(1) for x in model_files]
    if 'final' in file_epochs:
        print("Model already trained")
        exit()
    elif len(file_epochs) == 0:
        epoch_lower_bound = 0
        classifier = allocate_classifier(arguments.hidden)
    else:
        file_epochs = [int(x) for x in file_epochs]
        epoch_lower_bound = max(file_epochs)+1
        classifier = torch.load("models_{}/{}_{}.th".format(arguments.run, arguments.hidden, epoch_lower_bound-1))

    optimizer = torch.optim.Adam(classifier.parameters())

    for epoch in range(epoch_lower_bound, arguments.epochs):
        simulation_loader = iter(DataLoader(simulation_dataset, num_workers=0, batch_size=arguments.batch_size))
        reference_loader = iter(DataLoader(reference_dataset, num_workers=0, batch_size=arguments.batch_size))
        for iteration in range(iterations):
            theta, x_theta, y_theta = next(simulation_loader)
            x_theta_ref, y_theta_ref = next(reference_loader)
            in_real = torch.cat([theta, x_theta], dim=1).detach()
            in_fake = torch.cat([theta, x_theta_ref], dim=1).detach()
            y_real = classifier(in_real)
            y_fake = classifier(in_fake)
            loss = (bce(y_real, real) + bce(y_fake, fake)) / 2.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration % 100 == 0:
                print(epoch, '/', arguments.epochs, " @ ", iteration, '/', iterations)
        torch.save(classifier, "models_" + str(arguments.run) + "/" + str(arguments.hidden) + '_' + str(epoch) + ".th")
    torch.save(classifier, "models_" + str(arguments.run) + "/" + str(arguments.hidden) + "_final.th")


def allocate_classifier(hidden):
    classifier = torch.nn.Sequential(
        torch.nn.Linear(2, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
        torch.nn.Sigmoid())

    return classifier


class SimulationDataset(Dataset):

    def __init__(self, iterations, lower=-5., upper=5.):
        super(SimulationDataset, self).__init__()
        self.iterations = iterations
        self.uniform = Uniform(lower, upper)

    def __getitem__(self, index):
        return self.sample()

    def __len__(self):
        return self.iterations

    def sample(self):
        theta = self.uniform.sample().view(-1)
        x = Normal(theta, 1.).sample().view(-1)
        y = torch.tensor(1).float().view(-1)

        return theta, x, y


class ReferenceDataset(Dataset):

    def __init__(self, iterations, reference=0.):
        super(ReferenceDataset, self).__init__()
        self.iterations = iterations
        self.normal = Normal(reference, 1.)

    def __getitem__(self, index):
        x = self.normal.sample().view(-1)
        y = torch.tensor(0).float().view(-1)

        return x, y

    def __len__(self):
        return self.iterations


def parse_arguments():
    parser = argparse.ArgumentParser("Likelihood-free MCMC. Training.")
    parser.add_argument("--num-thetas", type=int, default=1000, help="Number of thetas to generate.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples for every theta.")
    parser.add_argument("--reference", type=float, default=0., help="Reference model parameter.")
    parser.add_argument("--lower", type=float, default=-5, help="Lower-bound of the parameter space.")
    parser.add_argument("--upper", type=float, default=5, help="Upper-bound of the parameter space.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations within a single epoch.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch-size")
    parser.add_argument("--hidden", type=int, default=500, help="Number of hidden units.")
    parser.add_argument("--run", type=int, default=1, help="Experiment run.")
    arguments, _ = parser.parse_known_args()
    if arguments.iterations is None:
        arguments.iterations = arguments.num_thetas * arguments.num_samples

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)

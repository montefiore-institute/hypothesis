import hypothesis
import numpy as np
import torch

from hypothesis.simulation import Simulator as BaseSimulator



class SpatialSIRSimulator(BaseSimulator):

    step_size = 0.01
    N = 100
    M = 100

    def __init__(self, default_measurement_time=1.0):
        super(SpatialSIRSimulator, self).__init__()
        self.default_measurement_time = default_measurement_time

    def simulate(self, theta, psi):
        # theta = [beta, gamma]
        # psi = tau
        # sample = [S(tau), I(tau), R(tau)]
        beta = theta[0].item()
        gamma = theta[1].item()
        # initialize the population with susceptible individuals only
        # first dimension indexes grid rows
        # second dimension indexes grid columns
        # third dimension indexes the one-hot encoded vector for the state of the indivitual
        population_grid = self.N * [self.M * [[1, 0, 0]]]
        # infect a single random individual
        population_grid[np.random.randint(0, self.N)][np.random.randint(0, self.M)] = [0, 1, 0]
        # evolve the population
        n_steps = int(psi / self.step_size)
        for _ in range(n_steps):
            for i in range(self.N):
                for j in range(self.M):
                    # if susceptible
                    if population_grid[i][j][0]:
                        I_neighbors = 0.0
                        N_neighbors = 0.0
                        if i > 0:
                            N_neighbors += 1
                            if population_grid[i - 1][j][1]:
                                I_neighbors += 1
                        if i < self.N - 1:
                            N_neighbors += 1
                            if population_grid[i + 1][j][1]:
                                I_neighbors += 1
                        if j > 0:
                            N_neighbors += 1
                            if population_grid[i][j - 1][1]:
                                I_neighbors += 1
                        if j < self.M - 1:
                            N_neighbors += 1
                            if population_grid[i][j + 1][1]:
                                I_neighbors += 1
                        if i > 0 and j > 0:
                            N_neighbors += 1
                            if population_grid[i - 1][j - 1][1]:
                                I_neighbors += 1
                        if i > 0 and j < self.M - 1:
                            N_neighbors += 1
                            if population_grid[i - 1][j + 1][1]:
                                I_neighbors += 1
                        if i < self.N - 1 and j > 0:
                            N_neighbors += 1
                            if population_grid[i + 1][j - 1][1]:
                                I_neighbors += 1
                        if i < self.N - 1 and j < self.M - 1:
                            N_neighbors += 1
                            if population_grid[i + 1][j + 1][1]:
                                I_neighbors += 1
                        p_infect = beta * I_neighbors / N_neighbors
                        if np.random.uniform() < p_infect:
                            population_grid[i][j] = [0, 1, 0]
                    # if infected
                    elif population_grid[i][j][1]:
                        if np.random.uniform() < gamma:
                            population_grid[i][j] = [0, 0, 1]

        return population_grid

    @torch.no_grad()
    def forward(self, inputs, experimental_configurations=None):
        outputs = []

        n = len(inputs)
        for index in range(n):
            theta = inputs[index]
            if experimental_configurations is not None:
                psi = experimental_configurations[index]
                x = self.simulate(theta, psi.item())
            else:
                x = self.simulate(theta, self.default_measurement_time)
            outputs.append(x)

        return torch.tensor(outputs).float()

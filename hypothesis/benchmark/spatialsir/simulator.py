import hypothesis
import numpy as np
import torch

from hypothesis.simulation import Simulator as BaseSimulator
from scipy import signal
from torch.distributions.poisson import Poisson



class SpatialSIRSimulator(BaseSimulator):

    def __init__(self, initial_infections_rate=3, shape=(100, 100), default_measurement_time=1.0, step_size=0.01):
        super(SpatialSIRSimulator, self).__init__()
        self.default_measurement_time = default_measurement_time
        self.lattice_shape = shape
        self.p_initial_infections = Poisson(float(initial_infections_rate))
        self.simulation_step_size = step_size

    def _sample_num_initial_infections(self):
        return int(1 + self.p_initial_infections.sample().item())

    def simulate(self, theta, psi):
        # Extract the simulation parameters.
        beta = theta[0].item()  # Infection rate
        gamma = theta[1].item() # Recovery rate
        # Allocate the data grids.
        infected = np.zeros(self.lattice_shape, dtype=np.int)
        recovered = np.zeros(self.lattice_shape, dtype=np.int)
        kernel = np.ones((3, 3), dtype=np.int)
        # Seed the grid with the initial infections.
        num_initial_infections = self._sample_num_initial_infections()
        for _ in range(num_initial_infections):
            index_height = np.random.randint(0, self.lattice_shape[0])
            index_width = np.random.randint(0, self.lattice_shape[1])
            infected[index_height][index_width] = 1
        # Derrive the maximum number of simulation steps.
        simulation_steps = int(psi / self.simulation_step_size)
        susceptible = (1 - recovered) * (1 - infected)
        for _ in range(simulation_steps):
            if infected.sum() == 0:
                break
            # Infection
            potential = signal.convolve2d(infected, kernel, mode="same")
            potential *= susceptible
            potential = potential * beta / 8
            next_infected = ((potential > np.random.uniform(size=self.lattice_shape)).astype(np.int) + infected) * (1 - recovered)
            next_infected = (next_infected >= 1).astype(np.int)
            # Recover
            potential = infected * gamma
            next_recovered = ((potential > np.random.uniform(size=self.lattice_shape)).astype(np.int) + recovered)
            # Next parameters
            recovered = next_recovered
            infected = next_infected
            susceptible = (1 - recovered) * (1 - infected)
        # Convert to tensors
        susceptible = torch.from_numpy(susceptible).float().view(1, 1, self.lattice_shape[0], self.lattice_shape[1])
        infected = torch.from_numpy(infected).float().view(1, 1, self.lattice_shape[0], self.lattice_shape[1])
        recovered = torch.from_numpy(recovered).float().view(1, 1, self.lattice_shape[0], self.lattice_shape[1])
        image = torch.cat([susceptible, infected, recovered], dim=1)

        return image

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

        return torch.cat(outputs, dim=0).float()

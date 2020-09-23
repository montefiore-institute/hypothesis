import hypothesis
import numpy as np
import torch

from hypothesis.simulation import Simulator as BaseSimulator
from .util import PriorExperiment



class CatapultSimulator(BaseSimulator):

    LAUNCH_ANGLE_LIMIT_HIGH = 1.5707963267948965
    LAUNCH_ANGLE_LIMIT_LOW = 0.0

    def __init__(self, limit=100.0, step_size=0.01, record_wind=False):
        super(CatapultSimulator, self).__init__()
        self.dt = step_size
        # self.prior_experiment = PriorExperiment()
        self.limit = limit # Observational limit in meters
        self.record_wind = record_wind
        self.planet_mass = 5.972 * 10**24 # Kilogram
        self.planet_radius = 6371000 # Meters
        self.air_density = 1.2

    def _get_projectile(self, psi):
        area = psi[0].item()
        mass = psi[1].item()
        drag_coefficient = psi[2].item()

        return Projectile(area=area, mass=mass, drag_coefficient=drag_coefficient)

    def _get_launch_angle(self, psi):
        nominal_launch_angle = psi[3].item()
        launch_angle = nominal_launch_angle + np.random.normal() * 0.1
        # Check if the launch angle is valid (in radians).
        if launch_angle < self.LAUNCH_ANGLE_LIMIT_LOW:
            launch_angle = self.LAUNCH_ANGLE_LIMIT_LOW
        elif launch_angle > self.LAUNCH_ANGLE_LIMIT_HIGH:
            launch_angle = self.LAUNCH_ANGLE_LIMIT_HIGH

        return launch_angle

    def _get_launch_force(self, psi):
        launch_force = psi[4].item()
        launch_force = launch_force + (np.random.normal() * 5) # Newton
        if launch_force < 10:
            launch_force = 10

        return launch_force

    def _get_wind(self):
        return np.random.normal() * 5 # Meters per second

    def simulate(self, theta, psi, trajectory=False):
        # Setup the initial conditions and simulator state
        positions = []
        G = theta.item() * (10 ** -11)
        v_nominal_wind = self._get_wind()
        launch_angle = self._get_launch_angle(psi)
        launch_force = self._get_launch_force(psi)
        projectile = self._get_projectile(psi)

        # Compute the initial launch force.
        force = np.zeros(2)
        force[0] = np.cos(launch_angle) * launch_force
        force[1] = np.sin(launch_angle) * launch_force

        # Compute the force due to acceleration
        force_gravitational = np.zeros(2)
        force_gravitational[1] = -projectile.mass * ((G * self.planet_mass) / self.planet_radius ** 2)
        positions.append(np.copy(projectile.position).reshape(1, 2))

        # Apply the launching force for a 0.1 second.
        n = int(0.1 / self.dt)
        for _ in range(n):
            projectile.apply(force, self.dt)
            positions.append(np.copy(projectile.position).reshape(1, 2))

        # Integrate until the projectile hits the ground.
        while not projectile.stopped() and np.abs(projectile.position[0]) <= self.limit:
            v_wind = v_nominal_wind + 0.01 * np.random.normal()

            # Force of the wind component.
            force_wind = np.zeros(2)
            force_wind[0] = np.sign(v_wind) * 0.5 * self.air_density * (projectile.area / projectile.mass) * (v_wind ** 2)

            # Force of the drag
            force_drag = np.zeros(2)
            force_drag[0] = -np.sign(projectile.velocity[0]) * 0.5 * projectile.drag_coefficient * self.air_density * projectile.area * (projectile.velocity[0] ** 2)
            force_drag[1] = -np.sign(projectile.velocity[1]) * 0.5 * projectile.drag_coefficient * self.air_density * projectile.area * (projectile.velocity[1] ** 2)

            # Compute net drag
            force = force_gravitational + force_wind + force_drag
            projectile.apply(force, self.dt)

            # Check if projectile is within limits
            x_position = projectile.position[0]
            if np.abs(x_position) > self.limit:
                x_position = np.sign(x_position) * self.limit
                positions.append(np.array([[x_position, 0]]))
            else:
                positions.append(np.copy(projectile.position).reshape(1, 2))

        positions = np.vstack(positions)
        if trajectory:
            return positions
        else:
            if self.record_wind:
                return np.array([v_nominal_wind, positions[-1][0]])
            else:
                return positions[-1][0]

    @torch.no_grad()
    def forward(self, inputs, experimental_configurations=None):
        outputs = []

        n = len(inputs)
        for index in range(n):
            theta = inputs[index]
            if experimental_configurations is not None:
                psi = experimental_configurations[index]
            else:
                psi = self.prior_experiment.sample()
            outputs.append(self._simulate(theta, psi).view(1, -1))

        return torch.cat(outputs, dim=0).float()



class SimpleCatapultSimulator(CatapultSimulator):

    def __init__(self, step_size=0.01, record_wind=False):
        super(SimpleCatapultSimulator, self).__init__(
            step_size=step_size,
            record_wind=record_wind)

    def _get_projectile(self, psi):
        area = 0.1 # Meters
        mass = 1.0 # Kilogram
        drag_coefficient = psi[0].item()

        return Projectile(area=area, mass=mass, drag_coefficient=drag_coefficient)

    def _get_launch_angle(self, psi):
        nominal_launch_angle = psi[1].item()
        launch_angle = nominal_launch_angle + np.random.normal() * 0.1
        # Check if the launch angle is valid (in radians).
        if launch_angle < self.LAUNCH_ANGLE_LIMIT_LOW:
            launch_angle = self.LAUNCH_ANGLE_LIMIT_LOW
        elif launch_angle > self.LAUNCH_ANGLE_LIMIT_HIGH:
            launch_angle = self.LAUNCH_ANGLE_LIMIT_HIGH

        return launch_angle

    def _get_launch_force(self, psi):
        launch_force = 250.0 # Newton
        launch_force = launch_force + (np.random.normal() * 5) # Newton
        if launch_force < 10:
            launch_force = 10

        return launch_force



class Projectile:
    r"""A spherical projectile."""

    def __init__(self, area=0.1, mass=1.0, drag_coefficient=0.0):
        self.position = np.zeros(2) # x -> distance, y -> height
        self.velocity = np.zeros(2)
        self.mass = mass # Kilogram
        self.drag_coefficient = drag_coefficient
        self.area = area # Meter

    def stopped(self):
        return self.position[1] < 0

    def apply(self, force, dt):
        impulse = force * dt
        dv = impulse / self.mass
        self.velocity += dv
        self.position += self.velocity * dt

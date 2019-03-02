from __future__ import division

import numpy as np
import hypothesis.benchmark.util.math as util_math


class SimTooLongException(Exception):
    """
    Exception to be thrown when a simulation runs for too long.
    """

    def __init__(self, max_n_steps):
        self.max_n_steps = max_n_steps

    def __str__(self):
        return 'Simulation exceeded the maximum of {} steps.'.format(self.max_n_steps)


class MarkovJumpProcess:
    """
    Implements a generic Markov Jump Process. It's an abstract class and must be implemented by a subclass.
    """

    def __init__(self, init, params):
        """
        :param init: initial state
        :param params: parameters
        """

        self.state = None
        self.params = None
        self.time = None
        self.reset(init, params)

    def reset(self, init, params):
        """
        Resets the simulator.
        :param init: initial state
        :param params: parameters
        """

        self.state = np.asarray(init, dtype=float)
        self.params = np.asarray(params, dtype=float)
        self.time = 0.0

    def _calc_propensities(self):
        raise NotImplementedError('This is an abstract method and should be implemented in a subclass.')

    def _do_reaction(self, reaction):
        raise NotImplementedError('This is an abstract method and should be implemented in a subclass.')

    def sim_steps(self, num_steps, include_init_state=True, rng=np.random):
        """
        Runs the simulator for a given number of steps.
        :param num_steps: number of steps
        :param include_init_state: if True, include the initial state in the output
        :param rng: random number generator to use
        :return: times, states
        """

        times = [self.time]
        states = [self.state.copy()]

        for _ in range(num_steps):

            rates = self.params * self._calc_propensities()
            total_rate = rates.sum()

            if total_rate == 0:
                self.time = float('inf')
                break

            self.time += rng.exponential(scale=1./total_rate)

            reaction = util_math.discrete_sample(rates / total_rate, rng=rng)
            self._do_reaction(reaction)

            times.append(self.time)
            states.append(self.state.copy())

        if not include_init_state:
            times, states = times[1:], states[1:]

        return np.array(times), np.array(states)

    def sim_time(self, dt, duration, include_init_state=True, max_n_steps=float('inf'), rng=np.random):
        """
        Runs the simulator for a given amount of time.
        :param dt: time step
        :param duration: total amount of time
        :param include_init_state: if True, include the initial state in the output
        :param max_n_steps: maximum number of simulator steps allowed. If exceeded, an exception is thrown.
        :param rng: random number generator to use
        :return: states
        """

        num_rec = int(duration / dt) + 1
        states = np.empty([num_rec, self.state.size], float)
        cur_time = self.time
        n_steps = 0

        for i in range(num_rec):

            while cur_time > self.time:

                rates = self.params * self._calc_propensities()
                total_rate = rates.sum()

                if total_rate == 0:
                    self.time = float('inf')
                    break

                self.time += rng.exponential(scale=1./total_rate)

                reaction = util_math.discrete_sample(rates / total_rate, rng=rng)
                self._do_reaction(reaction)

                n_steps += 1
                if n_steps > max_n_steps:
                    raise SimTooLongException(max_n_steps)

            states[i] = self.state.copy()
            cur_time += dt

        return states if include_init_state else states[1:]


class LotkaVolterra(MarkovJumpProcess):
    """
    The Lotka-Volterra implementation of the Markov Jump Process.
    """

    def _calc_propensities(self):

        x, y = self.state
        xy = x * y
        return np.array([xy, x, y, xy])

    def _do_reaction(self, reaction):

        if reaction == 0:
            self.state[0] += 1

        elif reaction == 1:
            self.state[0] -= 1

        elif reaction == 2:
            self.state[1] += 1

        elif reaction == 3:
            self.state[1] -= 1

        else:
            raise ValueError('Unknown reaction.')

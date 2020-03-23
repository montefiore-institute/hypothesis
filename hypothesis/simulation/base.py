import torch

from torch.multiprocessing import Pool



class Simulator(torch.nn.Module):
    r"""Base simulator class.

    A simulator defines the forward model.

    Example usage of a potential simulator implementation::

        simulator = MySimulator()
        inputs = prior.sample(torch.Size([10])) # Draw 10 samples from the prior.
        outputs = simulator(inputs)
    """

    def __init__(self):
        super(Simulator, self).__init__()

    def forward(self, inputs):
        r"""Defines the computation of the forward model at every call.

        Note:
            Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def __del__(self):
        self.terminate()

    def terminate(self):
        r"""Terminates the simulator and cleans up possible contexts.

        Note:
            Should be overridden by subclasses with a simulator state requiring graceful exits.
        Note:
            Subclasses should describe the expected format of ``inputs``.
        """
        pass



class Environment:

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError



class ParallelSimulator(Simulator):

    def __init__(self, simulator, workers=2):
        super(ParallelSimulator, self).__init__()
        self.pool = Pool(processes=workers)
        self.simulator = simulator
        self.workers = workers

    def _prepare_arguments(self, inputs):
        arguments = []

        chunks = inputs.shape[0] // self.workers
        if chunks == 0:
            chunks = 1
        chunks = inputs.split(chunks, dim=0)
        for chunk in chunks:
            a = (self.simulator, chunk)
            arguments.append(a)

        return arguments

    def forward(self, inputs):
        arguments = self._prepare_arguments(inputs)
        outputs = self.pool.map(self._simulate, arguments)
        outputs = torch.cat(outputs, dim=0)

        return outputs

    def terminate(self):
        self.pool.close()
        del self.pool
        self.pool = None
        self.simulator.terminate()

    @staticmethod
    def _simulate(arguments):
        simulator, inputs = arguments

        return simulator(inputs)

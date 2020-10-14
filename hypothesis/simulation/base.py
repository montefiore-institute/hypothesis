import torch

from multiprocessing import Pool



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



class ParallelSimulator(Simulator):

    def __init__(self, simulator, workers=2):
        super(ParallelSimulator, self).__init__()
        self.simulator = simulator
        self.workers = workers

    @torch.no_grad()
    def _prepare_arguments(self, **kwargs):
        arguments = []

        # Determine the number of chunks
        rows = kwargs[list(kwargs.keys())[0]].shape[0]
        chunk_size = rows // self.workers
        if chunk_size == 0:
            chunk_size = 1
        for base in range(0, rows, chunk_size):
            argument = {}
            for k, v in kwargs.items():
                argument[k] = v[base:base + chunk_size]
            arguments.append((self.simulator, argument))

        return arguments

    @torch.no_grad()
    def forward(self, **kwargs):
        pool = Pool(processes=self.workers)
        arguments = self._prepare_arguments(**kwargs)
        outputs = pool.map(self._simulate, arguments)
        pool.close()
        pool.join()
        del pool

        return torch.cat(outputs, dim=0)

    @staticmethod
    def _simulate(arguments):
        simulator, kwargs = arguments

        return simulator(**kwargs)

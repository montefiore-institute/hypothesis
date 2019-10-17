import torch



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
        self.simulator = simulator
        self.workers = workers
        self.pool = Pool(processes=self.workers)

    def forward(self, inputs):
        arguments = self._prepare_batch(inputs)
        outputs = pool.map(self._simulate, arguments)
        outputs = torch.cat(outputs, dim=0)

        return outputs

    def terminate(self):
        del self.pool
        self.pool = None

    @staticmethod
    def _simulate(simulator, inputs):
        return simulator(inputs)

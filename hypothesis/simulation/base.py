import torch


class BaseSimulator(torch.nn.Module):
    r"""Base simulator class.

    A simulator defines the forward model.

    Example usage of a potential simulator implementation:

    .. code-block:: python

        simulator = MySimulator()
        inputs = prior.sample((10,)) # Draw 10 samples from the prior.
        outputs = simulator(inputs)

    .. note::

        The ``inputs`` and ``outputs`` variable name in most simulator denote
        their position with respect to the simulation model. ``inputs`` are
        typically free parameters of the simulation model which sample
        (or produce deterministically) ``outputs``.

    .. note::

        Although it is possibly to supply a batch of inputs, it should be
        noted that these are currently `not` parallelized.

    """

    def __init__(self):
        super(BaseSimulator, self).__init__()
        self.eval()

    def forward(self, **kwargs):
        r"""Defines the computation of the forward model at every call.

        .. note::

            Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def __del__(self):
        self.terminate()

    def terminate(self):
        r"""Terminates the simulator and cleans up possible contexts.

        .. note::

            Should be overridden by subclasses with a simulator state requiring graceful exits.
        """
        pass

r"""Exceptions specific to `hypothesis.simulation`,
and simulators defined through that interface.

"""

class SimulationTimeError(Exception):
    r"""an exception to indicate an issue with the simulation time.

    This happens whenever the simulation time has exceeded or
    was to quick according to some prespecified threshold.
    """
    def __init__(self, message=None):
        super(SimulationTimeError, self).__init__(message)

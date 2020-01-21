class SimulationException(Exception):
    r""""""

    def __init__(self, message=None):
        if message is None:
            message = "Unspecified simulation error."
        super(SimulationException, self).__init__(message)

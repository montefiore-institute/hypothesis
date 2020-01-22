class SimulatorException(Exception):
    r""""""

    def __init__(self, message=None):
        if message is None:
            message = "Unspecified simulation error."
        super(SimulatorException, self).__init__(message)

r"""Exceptions related to Hypothesis events for hook registration and handling.

"""

class NoEventRegistrationException(Exception):
    r""""""

    def __init__(self, message=None):
        if message is None:
            message = "No events were registered by the procedure."
        super(NoEventRegistrationException, self).__init__(message)


class NoSuchEventException(Exception):
    r""""""

    def __init__(self, message=None):
        if message is None:
            message = "The specified event is not registered."
        super(NoSuchEventException, self).__init__(message)

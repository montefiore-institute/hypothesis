class NoEventRegistrationException(Exception):
    r""""""

    def __init__(self, message=None):
        if message is None:
            message = "No events were registered by the procedure."
        super(NoEventRegistrationException, self).__init__(message)

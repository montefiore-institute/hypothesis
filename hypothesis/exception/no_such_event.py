class NoSuchEventException(Exception):
    r""""""

    def __init__(self, message=None):
        if message is None:
            message = "The specified event is not registered."
        super(NoSuchEventException, self).__init__(message)

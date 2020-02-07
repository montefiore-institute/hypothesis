class IntractableException(Exception):
    r""""""

    def __init__(self, message=None):
        if message is None:
            message = "This evaluation is intractable!"
        super(IntractableException, self).__init__(message)

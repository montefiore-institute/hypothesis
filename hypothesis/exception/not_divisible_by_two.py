class NotDivisibleByTwoException(Exception):
    r""""""

    def __init__(self, message=None):
        # Check if a custom message has been specified.
        if message is None:
            message = "Not divisible by two!"
        super(DivisibleByTwoException, self).__init__(message)

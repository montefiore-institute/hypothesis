r"""Exceptions related to ratio estimation.

"""


default_error_message = r"""Unknown ratio estimator specified.

The following types are currently supported:
 - `mlp`
 - `resnet`
 - `resnet-18`
 - `resnet-34`
 - `resnet-50`
 - `resnet-101`
 - `resnet-152`
 - `densenet`
 - `densenet-121`
 - `densenet-161`
 - `densenet-169`
 - `densenet-201`
"""

class UnknownRatioEstimatorError(Exception):
    r"""An exception to indicate the absence of a ratio estimator definition.

    This happens whenever a ratio estimator identifier has been specified
    which is not known to hypothesis.
    """
    def __init__(self, message=default_error_message):
        super(NoWorkflowContextError, self).__init__(message)

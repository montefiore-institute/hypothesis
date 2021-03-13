class NoWorkflowContextError(Exception):
    r"""An exception to indicate the absence of a workflow context.

    This happens whenever no node has been added to a workflow.
    """
    def __init__(self, message="No node has been added to the workflow context."):
        super(NoWorkflowContextError, self).__init__(message)

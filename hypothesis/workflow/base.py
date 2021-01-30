import hypothesis as h
import hypothesis.workflow as w

from .graph import WorkflowGraph
from hypothesis.engine import Procedure
from hypothesis.util import is_iterable


class BaseWorkflow(Procedure):

    def __init__(self):
        super(BaseWorkflow, self).__init__()
        # Backup the current context
        original_context = h.workflow.context
        w.context = WorkflowGraph()
        # Build the workflow graph
        self._build_graph()
        self._graph = w.context
        w.context = original_context

    def _register_events(self):
        pass  # No new events to register.

    @property
    def graph(self):
        return self._graph

    def attach(self, workflows):
        if not is_iterable(workflows):
            workflows = [workflows]
        leaves = list(self._graph.leaves)
        for leaf in leaves:
            for workflow in workflows:
                leaf.add_child(workflow.graph.root)
            self._graph.rebuild()

        return self

    def build(self):
        self._graph.prune()
        w.context = self._graph

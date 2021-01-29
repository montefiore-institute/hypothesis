import hypothesis as h
import hypothesis.workflow as w

from hypothesis.engine import Procedure
from .graph import WorkflowGraph


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

    @property
    def graph(self):
        return self._graph

    def attach(self, workflow):
        leaves = self._graph.leaves
        for leaf in leaves:
            leaf.add_child(workflow._graph.root)

        return self

    def build(self):
        self._graph.prune()
        w.context = self._graph

    def _build_graph(self):
        raise NotImplementedError

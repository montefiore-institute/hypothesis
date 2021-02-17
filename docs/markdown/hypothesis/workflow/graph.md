Module hypothesis.workflow.graph
================================

Classes
-------

`WorkflowGraph()`
:   

    ### Instance variables

    `leaves`
    :

    `nodes`
    :

    `root`
    :

    ### Methods

    `add_node(self, node)`
    :

    `bfs(self)`
    :

    `debug(self)`
    :

    `delete_node(self, node)`
    :

    `find_node(self, f)`
    :

    `program(self)`
    :

    `prune(self)`
    :

    `rebuild(self)`
    :

`WorkflowNode(f)`
:   

    ### Instance variables

    `attributes`
    :

    `children`
    :

    `dependencies`
    :

    `disabled`
    :

    `f`
    :

    `name`
    :

    `parents`
    :

    `postconditions`
    :

    `siblings`
    :

    `tasks`
    :

    ### Methods

    `add_child(self, node)`
    :

    `add_parent(self, node)`
    :

    `add_postcondition(self, condition)`
    :

    `has_posconditions(self)`
    :

    `postconditions_satisfied(self)`
    :

    `remove_child(self, node)`
    :

    `remove_parent(self, node)`
    :
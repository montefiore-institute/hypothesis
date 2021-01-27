from queue import Queue


class WorkflowGraph:

    def __init__(self):
        super(WorkflowGraph, self).__init__()
        self._root = None
        self._nodes = {}

    def add_node(self, node):
        self._nodes[id(node.f)] = node

    def delete_node(self, node):
        del self._nodes[id(node.f)]

    def find_node(self, f):
        try:
            node = self._nodes[id(f)]
        except:
            node = None

        return node

    @property
    def nodes(self):
        return self._nodes.values()

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, node):
        if self._root is not None:
            self.delete_node(node)
        self._root = node
        self.add_node(node)

    @property
    def leafs(self):
        leafs = []

        for n in self.nodes:
            if len(n.children) == 0:
                leafs.append(n)

        return leafs

    def program(self):
        # Determine the order of execution
        execution_order = {}
        bfs_order = self.bfs()
        current_priority = 0
        for priority, node in enumerate(bfs_order):
            execution_order[id(node.f)] = node, priority
        # Sort subroutines by priority
        program = []
        for instruction_with_priority in sorted(execution_order.items(), key=lambda item: item[1][1]):
            subroutine = instruction_with_priority[1][0]
            program.append(subroutine)  # Fetch the instruction

        return program

    def bfs(self):
        queue = Queue()
        queue.put(self.root)
        yield from self._bfs(queue)

    def _bfs(self, queue):
        if not queue.empty():
            node = queue.get()
            yield node
            for c in node.children:
                queue.put(c)
            yield from self._bfs(queue)

    def prune(self):
        # Check if the postconditions of the root node have been satisfied.
        if len(self._root.postconditions) > 0 and self._root.postconditions_satisfied():
            self._root = None
            self._nodes = {}
        else:
            # Prune the computational graph
            for leaf in self.leafs:
                self._prune_node(leaf)
            all_nodes = set(self.nodes)
            in_graph = self._in_subgraph(self.root)
            to_delete = all_nodes.difference(in_graph)
            for node in to_delete:
                del self._nodes[id(node.f)]

    def _branches(self, node):
        branches = []

        for p in node.parents:
            if p == self.root:
                branches.append(node)
            else:
                branches.extend(self._branches(p))

        return list(set(branches))

    def _prune_node(self, node):
        pc = len(node.postconditions) > 0 and node.postconditions_satisfied()
        if pc or node.disabled:
            branches = self._branches(node)
            if len(branches) == 0:
                return
            for b in branches:
                self.root.remove_child(b)
            for c in node.children:
                self.root.add_child(c)
        else:
            for p in node.parents:
                self._prune_node(p)

    def _in_subgraph(self, node):
        attached = [node]

        for c in node.children:
            attached.append(c)
        for c in node.children:
            attached.extend(self._in_subgraph(c))

        return list(set(attached))

    def debug(self):
        self._debug_node(self.root)

    def _debug_node(self, node):
        children = node.children
        for c in children:
            print(c, "depends on", node)
        for c in children:
            self._debug_node(c)


class WorkflowNode:

    def __init__(self, f):
        super(WorkflowNode, self).__init__()
        self._attributes = {}
        self._disabled = False
        self._children = []
        self._parents = []
        self._f = f
        self._postconditions = []
        self._tasks = 1

    @property
    def f(self):
        return self._f

    @property
    def tasks(self):
        return self._tasks

    @tasks.setter
    def tasks(self, value):
        assert value >= 1
        self._tasks = value

    @property
    def attributes(self):
        return self._attributes

    @property
    def postconditions(self):
        return self._postconditions

    def add_postcondition(self, condition):
        self._postconditions.append(condition)

    def postconditions_satisfied(self):
        return all(c() for c in self.postconditions)

    @property
    def name(self):
        if "name" in self._attributes.keys():
            return self._attributes["name"]
        else:
            return self.f.__name__

    @name.setter
    def name(self, value):
        self._attributes["name"] = value

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, state):
        self._disabled = state

    def __setitem__(self, key, value):
        self._attributes[key] = value

    def __getitem__(self, key):
        return self._attributes[key]

    @property
    def parents(self):
        return self._parents

    @parents.setter
    def parents(self, parents):
        self._parents = parents

    def add_parent(self, node):
        assert node is not None
        self._parents.append(node)
        node._children.append(self)
        node._children = list(set(node._children))
        self._parents = list(set(self._parents))

    def remove_parent(self, node):
        try:
            index = self._parents.index(node)
            del self._parents[index]
            node.remove_child(self)
        except:
            pass

    def add_child(self, node):
        assert node is not None
        self._children.append(node)
        node._parents.append(self)
        self._children = list(set(self._children))
        node._parents = list(set(node._parents))

    def remove_child(self, node):
        try:
            index = self._children.index(node)
            del self._children[index]
            node.remove_parent(self)
        except:
            pass

    @property
    def dependencies(self):
        return self.parents

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        self._children = children

    def __str__(self):
        return self.name

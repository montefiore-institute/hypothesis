import hypothesis as h


def parameterized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


def get_and_add_node(f):
    # Check if a context has been allocated
    if h.workflow.context is None:
        h.workflow.context = Graph()
    node = h.workflow.context.find_node(f)
    if node is None:
        node = Node(f)
        h.workflow.context.add_node(node)

    return node


def main(f):
    if f is None:
        return
    node = get_and_add_node(f)
    if node.disabled:
        raise Exception("The entry point cannot be disabled.")
    # Check if a root node has already been set
    if h.workflow.context.root is not None:
        raise Exception("Duplicate decorator @hypothesis.workflow.main.")
    h.workflow.context.root = node

    return f


@parameterized
def dependency(f, dependency):
    if f == dependency:
        raise Exception("A function cannot depend on itself.")
    dependency_node = get_and_add_node(dependency)
    node = get_and_add_node(f)
    dependency_node.add_dependency(node)

    return f


@parameterized
def postcondition(f, condition):
    node = get_and_add_node(f)
    node.add_postcondition(condition)

    return f


def disable(f):
    node = get_and_add_node(f)
    node.disabled = True
    if node == h.workflow.context.root:
        raise Exception("The entry point cannot be disabled.")

    return f


@parameterized
def tasks(f, num_tasks):
    assert num_tasks >= 1
    node = get_and_add_node(f)
    node.tasks = num_tasks

    return f


@parameterized
def attribute(f, key, value):
    node = get_and_add_node(f)
    node.set(key, value)

    return f


@parameterized
def name(f, name):
    node = get_and_add_node(f)
    node.name = name

    return f


class Graph:

    def __init__(self):
        self._root = None
        self._nodes = {}

    @property
    def nodes(self):
        return self._nodes.values()

    def add_node(self, node):
        self._nodes[node.f] = node

    def delete_node(self, node):
        del self._nodes[node.f]

    def find_node(self, f):
        try:
            node = self._nodes[f]
        except:
            node = None

        return node

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, node):
        if self._root is not None:
            self._nodes.remove(self._root)
        self._root = node

    def compile(self):
        self._initialize_parent(self.root)

    @staticmethod
    def _initialize_parent(parent):
        for dependency in parent.dependencies:
            dependency.parent = parent
            Graph._initialize_parent(dependency)

    def prune(self):
        self.compile()
        self._prune_disabled_children(self.root)
        for leaf in self.leafs:
            self._prune_postcondition_parents(leaf)
        to_delete = []
        for node in self.nodes:
            if node.parent is None:
                to_delete.append(node)
        for node in to_delete:
            self.delete_node(node)

    def _prune_disabled_children(self, root):
        # Remove children which are disabled.
        disabled = []
        for c in root.dependencies:
            if c.disabled:
                disabled.append(c)
        for node in disabled:
            del self.nodes[node.f]
            index = root.dependencies.index(node)
            del root.dependencies[index]
        for c in root.dependencies:
            self._prune_disabled_children(c)

    def _clear_parent_until_root(self, node):
        if node != self.root and node != None:
            node.parent = None
            self._clear_parent_until_root(node.parent)

    def _prune_postcondition_parents(self, leaf):
        if len(leaf.postconditions) > 0 and leaf.postconditions_satisfied():
            self._clear_parent_until_root(leaf)
            leaf.parent = self.root
            self.root.add_dependency(leaf)

    @property
    def leafs(self):
        leafs = []
        for node in self.nodes:
            if len(node.dependencies) == 0:
                leafs.append(node)

        return leafs

    def debug(self):
        self.prune()
        print("Root node:", self.root)
        self._debug_children(self.root)

    def _debug_children(self, root):
        if root is not None:
            for c in root.dependencies:
                print(root, " -> ", c)
            for c in root.dependencies:
                self._debug_children(c)


class Node:

    def __init__(self, f):
        self._parent = None
        self._attributes = {}
        self._dependencies = []
        self._disabled = False
        self._postconditions = []
        self._tasks = 1
        self.f = f

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def tasks(self):
        return self._tasks

    @tasks.setter
    def tasks(self, value):
        assert value >= 1
        self._tasks = value

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

    def set(self, key, value):
        self._attributes[key] = value

    def get(self, key):
        return self._attributes[key]

    @property
    def attributes(self):
        return self._attributes

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, state):
        self._disabled = state

    def add_dependency(self, node):
        assert node is not None
        self._dependencies.append(node)

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, dependencies):
        self._dependencies = dependencies

    def __str__(self):
        return self.name

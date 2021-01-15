import hypothesis as h


from .graph import Graph
from .graph import Node


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
        return self._nodes

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
        self.add_node(node)

    def prune(self):
        self._prune_disabled_children(self.root)
        previous_num_nodes = 0
        num_nodes = len(self.nodes)
        while num_nodes != previous_num_nodes:
            self._prune_postcondition_children(self.root)
            previous_num_nodes = num_nodes
            num_nodes = len(self.nodes)

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

    def _prune_postcondition_children(self, root):
        new_dependencies = []
        delete_nodes = []
        for c in root.dependencies:
            if c.postconditions_satisfied():
                delete_nodes.append(c)
                new_dependencies.extend(c.dependencies)
            else:
                new_dependencies.append(c)
        for n in delete_nodes:
            self.delete_node(n)
        root.dependencies = new_dependencies
        for c in root.dependencies:
            self._prune_postcondition_children(c)

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
        self._attributes = {}
        self._dependencies = []
        self._disabled = False
        self._postconditions = []
        self.f = f

    @property
    def postconditions(self):
        return self._postconditions

    def add_postcondition(self, condition):
        self._postconditions.append(condition)

    def postconditions_satisfied(self):
        return len(self.postconditions) > 0 and all(c() for c in self.postconditions)

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

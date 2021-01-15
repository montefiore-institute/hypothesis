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

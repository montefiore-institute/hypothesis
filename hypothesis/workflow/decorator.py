import hypothesis as h

from .util import *


def root(f):
    node = add_and_get_node(f)
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
    dependency_node = add_and_get_node(dependency)
    node = add_and_get_node(f)
    node.add_parent(dependency_node)

    return f


@parameterized
def conda(f, environment):
    node = add_and_get_node(f)
    node["conda"] = str(environment)

    return f


@parameterized
def postcondition(f, condition):
    node = add_and_get_node(f)
    node.add_postcondition(condition)

    return f


def disable(f):
    node = add_and_get_node(f)
    node.disabled = True
    if node == h.workflow.context.root:
        raise Exception("The entry point cannot be disabled.")

    return f


@parameterized
def tasks(f, num_tasks):
    assert num_tasks >= 1
    node = add_and_get_node(f)
    node.tasks = num_tasks

    return f


@parameterized
def attribute(f, key, value):
    node = add_and_get_node(f)
    node.set(key, value)

    return f


@parameterized
def name(f, name):
    node = add_and_get_node(f)
    node.name = name

    return f

import os
import hypothesis as h
import hypothesis.workflow as w

from .graph import *


def add_and_get_node(f):
    if h.workflow.context is None:
        h.workflow.context = WorkflowGraph()
    node = h.workflow.context.find_node(f)
    if node is None:
        node = WorkflowNode(f)
        h.workflow.context.add_node(node)

    return node


def exists(f):
    def wrapper():
        return os.path.exists(f)

    return wrapper


def not_exists(f):
    def wrapper():
        return not os.path.exists(f)

    return wrapper


def parameterized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


def shell(command):
    return os.system(command)


def clear():
    w.context = None

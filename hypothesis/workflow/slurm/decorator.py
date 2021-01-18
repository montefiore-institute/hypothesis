import hypothesis.workflow as w

from hypothesis.workflow import get_and_add_node


@w.parameterize
def gpu(f, num_gpus):
    if num_gpus >= 1:
        node = get_and_add_node(f)
        node.set("--gres=gpu:", str(num_gpus))

    return f

@w.parameterize
def name(f, name):
    node = get_and_add_node(f)
    node.name = name
    node.set("--job-name", str(name))

    return f

@w.parameterize
def timelimit(f, time):
    node = get_and_add_node(f)
    node.set("--time", str(time))

    return f

@w.parameterize
def memory(f, memory):
    r"""Specify the real memory required per node.

    Default units are megabytes. Different units can be
    specified using the suffix [K|M|G|T].
    """
    node = get_and_add_node(f)
    node.set("--mem", str(memory))

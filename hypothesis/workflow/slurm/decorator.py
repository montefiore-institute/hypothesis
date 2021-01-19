import hypothesis.workflow as w


@w.parameterized
def gpu(f, num_gpus):
    if num_gpus >= 1:
        node = w.add_and_get_node(f)
        node["--gres=gpu:"] = str(num_gpus)

    return f

@w.parameterized
def name(f, name):
    node = w.add_and_get_node(f)
    node.name = name
    node["--job-name"] = str(name)

    return f

@w.parameterized
def timelimit(f, time):
    node = w.add_and_get_node(f)
    node["--time"] == str(time)

    return f

@w.parameterized
def memory(f, memory):
    r"""Specify the real memory required per node.

    Default units are megabytes. Different units can be
    specified using the suffix [K|M|G|T].
    """
    node = w.add_and_get_node(f)
    node["--mem"] = str(memory).upper()

    return f

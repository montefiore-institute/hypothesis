import hypothesis.workflow as w


@w.parameterized
def gpu(f, num_gpus):
    if num_gpus >= 1:
        node = w.add_and_get_node(f)
        node["--gres=gpu:"] = str(num_gpus)

    return f

@w.parameterized
def cpu(f, num_cpus):
    if num_cpus > 1:
        node = w.add_and_get_node(f)
        node["--cpus-per-task"] = str(num_cpus)

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
def cpu_and_memory(f, cores, memory):
    r"""Specify the minmum number of requires CPU cores and
    the TOTAL memory requirement of the job."""
    cpu(f, cores)
    # Check if custom memory has been specified.
    suffix = memory[-1]
    if suffix.isdigit():
        suffix = ""
        memory = int(memory)
    else:
        memory = int(memory[:-1])
    request = str(memory // cores) + suffix
    memory_per_cpu(f, request)

    return f


@w.parameterized
def memory_per_cpu(f, memory):
    r"""Specify the minimum memory requirement per cpu core.

    Default units are megabytes. Different units can be
    specified using the suffix [K|M|G|T].
    """
    node = w.add_and_get_node(f)
    node["--mem-per-cpu"] = str(memory).upper()

    return f

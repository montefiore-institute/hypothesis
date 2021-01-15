def execute(context):
    context.prune()
    root = context.root
    execute_node(root)
    execute_children(root)


def execute_node(root):
    num_tasks = root.tasks
    if num_tasks > 1:
        for id in range(num_tasks):
            root.f(id)
    else:
        root.f()
    assert root.postconditions_satisfied()


def execute_children(root):
    for d in root.dependencies:
        execute_node(d)
    for d in root.dependencies:
        execute_children(d)

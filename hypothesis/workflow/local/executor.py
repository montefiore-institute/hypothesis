import gc
import hypothesis.workflow as w
import logging
import sys


def execute(context=None):
    # Check if custom context has been specified
    if context is None:
        context = w.context
    # Prune the computational graph
    context.prune()
    # Check if a root node is present.
    if context.root is None:
        logging.critical("Postconditions of computational graph are met. Nothing to do.")
        sys.exit(0)
    # Determine the order of execution
    execution_order = {}
    bfs_order = list(context.bfs())
    current_priority = 0
    for priority, node in enumerate(bfs_order):
        execution_order[id(node.f)] = node, priority
    # Sort subroutines by priority
    program = []
    for instruction_with_priority in sorted(execution_order.items(), key=lambda item: item[1][1]):
        subroutine = instruction_with_priority[1][0]
        program.append(subroutine)  # Fetch the instruction
    # Execute the computational graph
    for instruction_index in range(len(program)):
        subroutine = program[instruction_index]
        num_tasks = subroutine.tasks
        if num_tasks > 1:
            for task_index in range(num_tasks):
                subroutine.f(task_index)
        else:
            subroutine.f()
        gc.collect()

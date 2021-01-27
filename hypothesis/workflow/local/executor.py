import gc
import hypothesis.workflow as w
import logging
import sys
import faulthandler
import torch

faulthandler.enable()


@torch.no_grad()
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
    # Fetch the program in execution order.
    program = context.program()
    # Execute the computational graph
    for instruction_index in range(len(program)):
        subroutine = program[instruction_index]
        num_tasks = subroutine.tasks
        if num_tasks > 1:
            for task_index in range(num_tasks):
                with torch.enable_grad():
                    subroutine.f(task_index)
        else:
            with torch.enable_grad():
                subroutine.f()
        gc.collect()

import logging
import os
import shutil
import tempfile


def execute(context, directory=None, cleanup=False):
    # Add default Slurm attributes to the nodes.
    add_default_attributes(context)
    # Create the generation directory.
    if directory is None:
        directory = tempfile.mkdtemp()
    else:
        os.makedirs(directory)
    # Cleanup the generated filed.
    if cleanup:
        shutil.rmtree(directory)


def add_default_attributes(context):
    for node in context.nodes:
        node.set("--export", "ALL")
        node.set("--parsable", "")
        node.set("--requeue", "")

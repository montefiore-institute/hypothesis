import hypothesis as h
import hypothesis.workflow as w
import logging
import numpy as np
import os
import torch

from hypothesis.workflow import BaseWorkflow
from hypothesis.workflow import WorkflowGraph


class SimulationWorkflow(BaseWorkflow):

    def __init__(self, prior, simulator, directory=".", n=1000000, blocksize=1000):
        self._blocksize = blocksize
        self._class_prior = prior
        self._class_simulator = simulator
        self._directory = directory
        self._n = n
        self._num_blocks = n // blocksize
        super(SimulationWorkflow, self).__init__()

    @property
    def n(self):
        return self._n

    @property
    def blocksize(self):
        return self._blocksize

    @property
    def num_blocks(self):
        return self._num_blocks

    def _register_events(self):
        pass  # No events to register for this workflow

    def _build_graph(self):
        ### Stage 1. Create the data directory
        @w.root
        @w.postcondition(w.exists(self._directory + "/inputs.npy"))
        @w.postcondition(w.exists(self._directory + "/outputs.npy"))
        def create_data_directory():
            logging.info("Allocating the required data directories.")
            if not os.path.exists(self._directory):
                os.mkdir(self._directory)
            if not os.path.exists(self._directory + "/blocks"):
                os.mkdir(self._directory + "/blocks")

        ### Stage 2. Simulate the data blocks
        @w.dependency(create_data_directory)
        @w.tasks(self.num_blocks)
        @torch.no_grad()
        def simulate_block(task_index):
            skip = False
            suffix = str(task_index).zfill(5)
            base = self._directory + "/blocks/block-" + suffix
            path_inputs = base + "/inputs.npy"
            path_outputs = base + "/outputs.npy"
            # Check if the simulated files are present
            if os.path.exists(path_inputs) and os.path.exists(path_outputs):
                inputs = np.load(path_inputs)
                skip = (inputs.shape[0] == self.blocksize)
            # Check if this block can be skipped
            if skip:
                logging.warning("Data block " + suffix + " already simulated. Skipping.")
                return
            # Check if the base path exists
            if not os.path.exists(base):
                os.mkdir(base)
            # Simulate the block
            percentage = ((task_index + 1) / self.num_blocks) * 100
            percentage = "{:.2f}%".format(percentage)
            logging.info("Simulating block " + str(task_index + 1) + " / " + str(self.num_blocks) + " (" + percentage + ")")
            # Simulate the current block
            prior = self._class_prior()
            simulator = self._class_simulator()
            inputs = prior.sample((self.blocksize,))
            outputs = simulator(inputs)
            # Convert the samples from the joint to numpy data files
            inputs = inputs.numpy()
            outputs = outputs.numpy()
            np.save(path_inputs, inputs)
            np.save(path_outputs, outputs)

        ### Stage 3. Merge the simulated data blocks
        @w.dependency(simulate_block)
        @w.postcondition(w.exists(self._directory + "/inputs.npy"))
        def merge_inputs():
            logging.info("Merging the simulated inputs.")
            query = self._directory + "/blocks/block-*/inputs.npy"
            out = self._directory + "/inputs.npy"
            os.system("hypothesis merge --dimension 0 --sort --extension numpy --out " + out + " --files '" + query + "'")
        @w.dependency(simulate_block)
        @w.postcondition(w.exists(self._directory + "/outputs.npy"))
        def merge_outputs():
            logging.info("Merging the simulated outputs.")
            query = self._directory + "/blocks/block-*/outputs.npy"
            out = self._directory + "/outputs.npy"
            os.system("hypothesis merge --dimension 0 --sort --extension numpy --out " + out + " --files '" + query + "'")

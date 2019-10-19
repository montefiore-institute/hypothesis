import numpy as np
import os
import torch

from torch.utils.data import Dataset



class NumpySimulationDataset(Dataset):
    r""""""

    def __init__(self, inputs, outputs):
        super(NumpySimulationDataset, self).__init__()
        if not os.path.exists(inputs) or not os.path.exists(outputs):
            raise ValueError("Please specify a proper inputs or outputs path.")
        # Inputs properties.
        self.inputs_path = inputs
        self.inputs_fd = open(self.inputs_path, "rb")
        self.inputs_header, self.inputs_header_offset = self._parse_header(self.inputs_fd)
        self.inputs_fd.close()
        self.inputs_data_shape = self.inputs_header["shape"][-2:]
        self.inputs_data_type = self.inputs_header["descr"]
        self.inputs_data_counts = self._compute_counts(self.inputs_data_shape)
        self.inputs_data_bytes = int(self.inputs_data_type[-1]) * self.inputs_data_counts
        self.inputs_fd = None
        # Outputs properties.
        self.outputs_path = outputs
        self.outputs_fd = open(self.outputs_path, "rb")
        self.outputs_header, self.outputs_header_offset = self._parse_header(self.outputs_fd)
        self.outputs_fd.close()
        self.outputs_data_shape = self.outputs_header["shape"][-2:]
        self.outputs_data_type = self.outputs_header["descr"]
        self.outputs_data_counts = self._compute_counts(self.outputs_data_shape)
        self.outputs_data_bytes = int(self.outputs_data_type[-1]) * self.outputs_data_counts
        self.outputs_fd = None

    def _parse_header(self, fd):
        r"""
        Parses the ``numpy`` header of the specified file descriptor.

        Note:
            * The first 6 bytes are a magic string: exactly \x93NUMPY.
            * The next 1 byte is an unsigned byte: the major version number of the file format, e.g. \x01.
            * The next 1 byte is an unsigned byte: the minor version number of the file format, e.g. \x00. Note: the version of the file format is not tied to the version of the numpy package.
            * The next 2 bytes form a little-endian unsigned short int: the length of the header data HEADER_LEN.
        """
        prefix = fd.read(10) # Read fixed header.
        header_offset = int.from_bytes(prefix[-2:], byteorder="little")
        header = eval(fd.read(header_offset)) # Not very secure but whatever.
        header_offset += 10

        return header, header_offset

    def _retrieve(self, index):
        r""""""
        if self.inputs_fd is None:
            self.inputs_fd = open(self.inputs_path, "rb")
            self.outputs_fd = open(self.outputs_path, "rb")
        self.inputs_fd.seek(self.inputs_header_offset + index * self.inputs_data_bytes)
        self.outputs_fd.seek(self.outputs_header_offset + index * self.outputs_data_bytes)
        inputs = np.fromfile(self.inputs_fd, dtype=self.inputs_data_type, count=self.inputs_data_counts)
        outputs = np.fromfile(self.outputs_fd, dtype=self.outputs_data_type, count=self.outputs_data_counts)

        return inputs.reshape(self.inputs_data_shape[1:]), outputs.reshape(self.outputs_data_shape[1:])

    def __len__(self):
        return self.inputs_data_shape[0]

    def __del__(self):
        r""""""
        if hasattr(self, "inputs_fd") and self.inputs_fd is not None:
            self.inputs_fd.close()
            self.outputs_fd.close()
            self.inputs_fd = None
            self.outputs_fd = None

    def __getitem__(self, index):
        r""""""
        inputs, outputs = self._retrieve(index)
        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)

        return inputs, outputs

    @staticmethod
    def _compute_counts(shape):
        counts = 1
        for dimensionality in shape[1:]:
            counts *= dimensionality

        return counts

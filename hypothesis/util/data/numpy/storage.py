import numpy as np
import os
import torch


class BaseStorage:

    def close(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def __del__(self):
        self.close()


class InMemoryStorage(BaseStorage):

    def __init__(self, path):
        super(InMemoryStorage, self).__init__()
        # Check if the specified path exists.
        if path is None or not os.path.exists(path):
            raise ValueError("The path", path, "does not exists.")
        # Storage properties.
        self.path = path
        self.data = np.load(path)

    def close(self):
        if hasattr(self, "data"):
            del self.data
            self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index])


class PersistentStorage(BaseStorage):

    def __init__(self, path):
        super(PersistentStorage, self).__init__()
        # Check if the specified path exists.
        if path is None or not os.path.exists(path):
            raise ValueError("The path", path, "does not exists.")
        # Storage properties.
        self.path = path
        self.fd = open(self.path, "rb")
        self.header, self.offset = self._parse_header(self.fd)
        self.fd.close()
        self.fd = None
        self.data_shape = self.header["shape"][1:]
        self.data_type = self.header["descr"]
        self.data_dimensionality = self._compute_dimensionality(self.data_shape)
        self.data_bytes = int(self.data_type[-1]) * self.data_dimensionality
        self.size = self.header["shape"][0]

    def _retrieve(self, index):
        if self.fd is None:
            self.fd = open(self.path, "rb")
        self.fd.seek(self.offset + index * self.data_bytes)
        data = np.fromfile(self.fd, dtype=self.data_type, count=self.data_dimensionality)

        return data.reshape(self.data_shape)

    def close(self):
        if hasattr(self, "fd") and self.fd is not None:
            self.fd.close()
        self.fd = None

    def __getitem__(self, index):
        return self._retrieve(index)
        item = torch.from_numpy(data)

        return item

    def __len__(self):
        return self.size

    @staticmethod
    def _compute_dimensionality(shape):
        dimensionality = 1
        for size in shape:
            dimensionality *= size

        return dimensionality

    @staticmethod
    def _parse_header(fd):
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

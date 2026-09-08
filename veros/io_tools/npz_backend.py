"""Minimal restart file backend based on numpy .npz archives.

Used by :mod:`veros.io_tools.hdf5` as a fallback when ``h5py`` cannot be
installed. Only implements the small subset of the ``h5py.File`` API used by
:mod:`veros.restart`, and only supports single-process runs.
"""

import numpy as np

_DATA_MARKER = "data"
_ATTR_MARKER = "attr"


class NpzGroup:
    def __init__(self):
        self.datasets = {}
        self.attrs = {}

    def require_dataset(self, name, shape, dtype, **kwargs):
        return self.datasets.setdefault(name, np.zeros(shape, dtype=dtype))

    def __getitem__(self, name):
        return self.datasets[name]

    def __setitem__(self, name, value):
        self.datasets[name] = np.asarray(value)

    def items(self):
        return self.datasets.items()


class NpzFile:
    """Emulates the subset of the h5py.File API used by veros.restart."""

    def __init__(self, filepath, mode):
        self.filepath = filepath
        self.mode = mode
        self.groups = {}

        if mode == "r":
            with open(filepath, "rb") as f:
                archive = np.load(f, allow_pickle=False)
                for key in archive.files:
                    groupname, kind, name = key.split("/", 2)
                    group = self.groups.setdefault(groupname, NpzGroup())
                    if kind == _ATTR_MARKER:
                        group.attrs[name] = archive[key]
                    else:
                        group.datasets[name] = archive[key]

    def require_group(self, name):
        return self.groups.setdefault(name, NpzGroup())

    def __getitem__(self, name):
        return self.groups[name]

    def close(self):
        if self.mode == "r":
            return

        payload = {}
        for groupname, group in self.groups.items():
            for name, arr in group.datasets.items():
                payload[f"{groupname}/{_DATA_MARKER}/{name}"] = arr
            for name, val in group.attrs.items():
                payload[f"{groupname}/{_ATTR_MARKER}/{name}"] = np.asarray(val)

        with open(self.filepath, "wb") as f:
            np.savez(f, **payload)

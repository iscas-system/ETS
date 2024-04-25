from abc import ABC
from abc import abstractmethod
from typing import Mapping, List

import numpy as np
import torch.nn


class MModule(torch.nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_transfer(self, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self, outputs, Y) -> torch.Tensor:
        pass


def nested_detach(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    if isinstance(tensors, np.ndarray):
        return tensors
    return tensors.detach()


def pad_np_vectors(v: List[np.ndarray] | np.ndarray, maxsize=None):
    if maxsize is None:
        if isinstance(v, np.ndarray):
            raise ValueError("maxsize must be specified is v is np.ndarray")
        if isinstance(v, list):
            maxsize = np.amax([f.size for f in v])

    nv = list()
    for l in v:
        if l.size < maxsize:
            nv.append(
                np.pad(l, (0, maxsize - l.size))
            )
        else:
            nv.append(l)
    return nv

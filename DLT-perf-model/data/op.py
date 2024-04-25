from enum import Enum
from functools import lru_cache
from typing import Tuple, Optional, Union, List, Dict

import numpy as np

from data.util import get_op_frequency


class OperatorMode(Enum):
    Forward = 0
    Backward = 1
    Update = 2

    @lru_cache(maxsize=None)
    def encode(self) -> List:
        op_modes = [mode for mode in OperatorMode]
        return [1 if self == op_mode_ else 0 for op_mode_ in op_modes]


class Operator:
    def __init__(self,
                 operator_type_id: int,
                 operator_mode: OperatorMode,
                 batch_size: int = 0,
                 FLOPS: int = 0,
                 bytes: int = 0,
                 hyper_parameters: Optional[Tuple[Union[float, int]]] = None
                 ):
        self.operator_type_id: int = operator_type_id
        self.operator_mode: OperatorMode = operator_mode
        self.batch_size: int = batch_size
        self.FLOPS: int = FLOPS
        self.bytes: int = bytes
        self.hyper_parameters: Optional[Tuple[Union[float, int]]
        ] = hyper_parameters

    @staticmethod
    def dummy_op():
        return Operator(0, OperatorMode.Forward)

    @lru_cache(maxsize=None)
    @staticmethod
    def encode_op_type_id(i: int) -> List:
        l = [0] * 238
        l[i - 1] = 1
        return l

    @lru_cache(maxsize=None)
    @staticmethod
    def encode_op_type_id_static(i: int) -> List:
        return [i]

    @lru_cache(maxsize=None)
    @staticmethod
    def encode_op_type_id_frequency(i: int) -> List:
        op_frequency = get_op_frequency()
        return [op_frequency[i]]

    def to_feature_array(self, mode):
        if mode == "complex":
            complex_feature_vector = [
                *Operator.encode_op_type_id_static(self.operator_type_id),
                *self.operator_mode.encode(),
                self.batch_size,
                self.FLOPS,
                self.bytes,
            ]
            if self.hyper_parameters is not None:
                complex_feature_vector.extend(self.hyper_parameters)
            return np.array(complex_feature_vector)
        elif mode == "simple":
            simple_feature_vector = [
                *self.encode_op_type_id_static(self.operator_type_id),
                *self.operator_mode.encode(),
                self.batch_size,
            ]
            return np.array(simple_feature_vector)
        else:
            raise ValueError(
                "Invalid mode. Mode must be 'complex' or 'simple'.")

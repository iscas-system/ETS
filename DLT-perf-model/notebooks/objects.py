import pathlib
from enum import Enum
from functools import lru_cache
from typing import List

from torch.optim import Adam, RMSprop, SGD

ckpts_dir = pathlib.Path(__file__).parent / 'ckpts'
logs_dir = pathlib.Path(__file__).parent / 'logs'


class ModelType(Enum):
    GBDT = 0
    GCNSubgraph = 1
    MLP = 2
    PerfNet = 4
    Transformer = 5
    LSTM = 6
    GCNGrouping = 7
    GRU = 8
    RNN = 9
    MLPTestGrouping = 100
    MLPTestSubgraph = 101


class Environment:
    def __init__(self, gpu_type: 'GPUType'):
        self.gpu_type: GPUType = gpu_type
        # self.framework: str = framework
        # self.cuda_version: str = cuda_version

    @staticmethod
    def from_str(environment_str: str) -> 'Environment':
        gpu_type = GPUType[environment_str]
        return Environment(gpu_type=gpu_type)

    def __str__(self):
        # return f"{self.gpu_type.name}_{self.framework}_{self.cuda_version}"
        # return f"{self.gpu_type.name}_CPU{self.cpu}"
        return self.gpu_type.name

    def __repr__(self):
        return self.__str__()
    
    def __hash__(self) -> int:
        return self.gpu_type.__hash__()
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Environment):
            return False
        return __value.gpu_type == self.gpu_type 

    def __ne__(self, __value: object) -> bool:
        return not self.__eq__(__value)
    
    def toJson(self):
        return self.__str__()


class GPUType(Enum):
    RTX2080Ti_CPUALL = 0
    RTX2080Ti_CPU100 = 1
    RTX2080Ti_CPU80 = 2
    RTX2080Ti_CPU60 = 3
    T4_CPUALL = 4
    T4_CPU100 = 5
    T4_CPU80 = 6
    T4_CPU60 = 7
    P4_CPUALL = 8
    P4_CPU100 = 9
    P4_CPU80 = 10
    P4_CPU60 = 11
    RTX3080Ti_CPUALL = 12
    RTX3080Ti_CPU100 = 13
    RTX3080Ti_CPU80 = 14
    RTX3080Ti_CPU60 = 15
    RTX4090 = 8
    P40 = 9
    K80 = 10
    TEST = 100


class OptimizerType(Enum):
    Adam = Adam
    SGD = SGD
    RMSProp = RMSprop

    @lru_cache(maxsize=None)
    def encode(self, method="one-hot") -> List:
        om_types = [om for om in OptimizerType]
        if method == "one-hot":
            return [1 if self == om_type_ else 0 for om_type_ in om_types]
        else:
            raise ValueError(
                "Invalid method. Must be 'one-hot'.")

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



class GPUType(Enum):
    RTX2080TiCPUALL = 0
    RTX2080TiCPU100 = 1
    RTX2080TiCPU80 = 2
    RTX2080TiCPU60 = 3
    T4CPUALL = 4
    T4CPU100 = 5
    T4CPU80 = 6
    T4CPU60 = 7
    P4CPUALL = 8
    P4CPU100 = 9
    P4CPU80 = 10
    P4CPU60 = 11
    RTX3080TiCPUALL = 12
    RTX3080TiCPU100 = 13
    RTX3080TiCPU80 = 14
    RTX3080TiCPU60 = 15
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

import pathlib
import os
import pandas as pds
from pandas import DataFrame
from functools import lru_cache
from typing import List, Dict
from torch.utils.data import Dataset
from objects import Environment
import math
import random
import string
from typing import List, Optional, Dict, Tuple

import numpy as np
import pickle
import multiprocessing as mp
from enum import Enum
from functools import lru_cache
from typing import Tuple, Optional, Union, List, Dict
from logger import logging

repo_root = pathlib.Path(__file__).parent.parent.absolute()
datasets_path = str(repo_root / "datasets")
pkl_path = str(repo_root / "pkl")
configs_path = str(pathlib.Path(__file__).parent / "configs")
print(f"datasets_path: {datasets_path}")
print(f"configs_path: {configs_path}")


op_freq = {}

with open(str(pathlib.Path(configs_path) / "op_frequency.dict"), "r") as f:
    op_freq = eval(f.read())


class OperatorMode(Enum):
    Forward = 0
    Backward = 1
    Update = 2

    @lru_cache(maxsize=None)
    def encode(self) -> List:
        op_modes = [mode for mode in OperatorMode]
        return [1 if self == op_mode_ else 0 for op_mode_ in op_modes]

class OperatorType(Enum):
    add = 0  # 12
    mul = 1  # 16
    conv2d = 2  # 203
    floordiv = 3  # 23
    sigmoid = 4  # 156
    batchnorm = 5  # 205
    relu = 6  # 152
    iadd = 7  # 35
    dropout = 7  # 212
    silu = 8  # 159
    linear = 9  # 201
    bernoulli = 10  # 234
    adaptive_avg_pool2d = 11  # 199
    layer_norm = 12  # 0
    normalize = 13  # 230
    sub = 14  # 21
    matmul = 15  # 207
    gelu = 16  # 166
    cat = 17  # 213
    clone = 18  # 239
    index = 19  # 233
    softmax = 20  # 224
    truediv = 21  # 25
    matmul_ = 22  # 206
    reshape = 23  # 214
    cross_entrpy = 24  # 181
    # 出现频率过低， 视为一类
    others = 25

    # pad = 25 # 229
    # clamp = 26 # 93
    # avg_pool2d = 27 # 189
    # hardswish = 28 # 243
    # roll = 29 # 231
    # max_pool3d = 30 # 183
    # hardtanh = 31 # 154
    # hardsigmoid = 32 # 244
    # newzeros = 33 # 232
    # mean = 34 # 210
    @lru_cache(maxsize=None)
    def encode(self) -> List:
        ops = [op for op in OperatorType]
        return [1 if self == op else 0 for op in ops]

    @staticmethod
    def get_encode(op_id: int) -> List:
        op_map = {
            12: OperatorType.add,
            16: OperatorType.mul,
            203: OperatorType.conv2d,
            23: OperatorType.floordiv,
            156: OperatorType.sigmoid,
            205: OperatorType.batchnorm,
            152: OperatorType.relu,
            35: OperatorType.iadd,
            212: OperatorType.dropout,
            159: OperatorType.silu,
            201: OperatorType.linear,
            234: OperatorType.bernoulli,
            199: OperatorType.adaptive_avg_pool2d,
            0: OperatorType.layer_norm,
            230: OperatorType.normalize,
            21: OperatorType.sub,
            207: OperatorType.matmul,
            166: OperatorType.gelu,
            213: OperatorType.cat,
            239: OperatorType.clone,
            233: OperatorType.index,
            224: OperatorType.softmax,
            25: OperatorType.truediv,
            206: OperatorType.matmul_,
            214: OperatorType.reshape,
            181: OperatorType.cross_entrpy,
        }
        if op_id in op_map:
            return op_map[op_id].encode()
        else:
            return OperatorType.others.encode()


class OperatorDtype(Enum):
    Float = 0
    Half = 1
    Double = 2

    @lru_cache(maxsize=None)
    def encode(self) -> List:
        op_dtypes = [dtype for dtype in OperatorDtype]
        return [1 if self == op_dtype_ else 0 for op_dtype_ in op_dtypes]




class Operator:
    def __init__(self,
                 operator_type_id: int,
                 operator_mode: OperatorMode,
                 operator_dtype: OperatorDtype = OperatorDtype.Float,
                 h: int = 0,
                 batch_size: int = 0,
                 FLOPS: int = 0,
                 bytes: int = 0,
                 hyper_parameters: Optional[Tuple[Union[float, int]]] = None
                 ):
        self.operator_type_id: int = operator_type_id
        self.operator_mode: OperatorMode = operator_mode
        self.batch_size: int = batch_size
        self.h: int = h
        self.dtype: OperatorDtype = operator_dtype

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
        pass

    def to_feature_array(self, mode):
        if mode == "complex":
            complex_feature_vector = [
                # *Operator.encode_op_type_id_static(self.operator_type_id),
                *OperatorType.get_encode(self.operator_type_id),
                *self.operator_mode.encode(),
                *self.dtype.encode(),
                self.batch_size,
                self.h,
                self.FLOPS,
                self.bytes,
            ]
            if self.hyper_parameters is not None:
                complex_feature_vector.extend(self.hyper_parameters)
            return np.array(complex_feature_vector)
        elif mode == "simple":
            simple_feature_vector = [
                # *self.encode_op_type_id_static(self.operator_type_id),
                *OperatorType.get_encode(self.operator_type_id),
                *self.operator_mode.encode(),
                *self.dtype.encode(),
                self.h,
                self.batch_size,
            ]
            return np.array(simple_feature_vector)
        else:
            raise ValueError(
                "Invalid mode. Mode must be 'complex' or 'simple'.")



class GraphNode:
    def __init__(self,
                 node_id: int,
                 op: Operator,
                 duration: int,
                 gap: int):
        self.node_id: int = node_id
        self.op: Operator = op
        self.duration: int = duration
        self.gap: int = gap

    @staticmethod
    def dummy_node():
        return GraphNode(node_id=random.randint(1000, 1e6), op=Operator.dummy_op(), duration=0, gap=0)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id

    def __ne__(self, other):
        return not self.__eq__(other)


class Graph:

    operator_modes_map = {
        1: OperatorMode.Forward,
        2: OperatorMode.Backward,
        3: OperatorMode.Update
    }
    def __init__(self,
                 ID: Optional[str],
                 environment: Optional[Environment],
                 batch_size: Optional[int],
                 nodes: Optional[List[GraphNode]],
                 root_node: Optional[GraphNode]):
        self.ID: Optional[str] = ID
        self.environment: Optional[Environment] = environment
        self.batch_size: Optional[int] = batch_size
        self.nodes: List[GraphNode] = list() if nodes is None else nodes
        self.root_node: Optional[GraphNode] = root_node
        self.graph_duration = self._init_graph_duration()

    def _init_graph_duration(self) -> float:
        graph_duration = 0
        for node in self.nodes:
            graph_duration += node.duration + node.gap
        return graph_duration

    @staticmethod
    def from_data(env: Environment,
                  filename: Optional[str] = None,
                  df: Optional[DataFrame] = None,
                  dummy: bool = False,
                  seed: int = 0) -> 'Graph':
        def generate_dummy():
            rand = random.Random(seed)
            random_graph_id = ''.join(rand.choices(
                string.ascii_letters + string.digits, k=10))
            env = Environment.from_str("RTX2080Ti_CPU100")
            batch_size = 64
            operator_modes = [OperatorMode.Forward,
                              OperatorMode.Backward, OperatorMode.Update]

            num_nodes = rand.randint(10, 100)
            nodes = list()
            for i in range(num_nodes):
                op_type = rand.choice([0, 1, 2, 3])
                op_mode = rand.choice(operator_modes)
                batch_size = rand.choice([32, 64, 128])
                hyper_param_cnt = rand.randint(0, 10)
                args = {
                    'FLOPS': rand.uniform(0, 1),
                    'batch_size': batch_size,
                    'hyper_parameters': tuple(rand.uniform(0, 1) for i in range(hyper_param_cnt))
                }
                op = Operator(op_type, op_mode, **args)
                duration, gap = rand.uniform(0, 1), rand.uniform(0, 1)
                current_node = GraphNode(i,
                                         op,
                                         duration=duration,
                                         gap=gap)
                nodes.append(current_node)
            root_node = nodes[0]
            return Graph(random_graph_id, env, batch_size, nodes, root_node)

        if dummy:
            return generate_dummy()

        # d = df.to_dict()
        total_op = len(df)
        nodes = list()
        for i in range(total_op):
            operator_type_id = int(df.iloc[i]["op"])
            operator_mode = OperatorMode(Graph.operator_modes_map[df.iloc[i]["dir"]])
            # op_hyper_parameters = [] #list(eval(df.iloc[i]["params"]))
            params = df.iloc[i]["params"]
            # params is like a "[1, 2, 3]" string. Turn it into python list. Do not use eval due to security issue.
            op_hyper_parameters = list(map(int, params[1:-1].split(",")))
            # 某些算子的参数超过30个，这里只取前30个
            op_hyper_parameters = op_hyper_parameters[:30]
            if len(op_hyper_parameters) < 30:
                # pad to 30
                op_hyper_parameters.extend([0] * (30 - len(op_hyper_parameters)))
            
            # todo 添加开关
            op_hyper_parameters = [math.log(i+1) for i in op_hyper_parameters]
            flops = int(df.iloc[i]["flops"])
            bytes = int(df.iloc[i]["bytes"])
            kduration = int(df.iloc[i]["kduration"]) / 1000.  # us
            space = int(df.iloc[i]["space"]) / 1000.  # us
            if space < 0:
                space = 0
            batch_size = int(df.iloc[i]["batch"])
            h = int(df.iloc[i]["h"])
            op = Operator(operator_type_id=operator_type_id,
                          operator_mode=operator_mode,
                          FLOPS=flops,
                          bytes=bytes,
                          batch_size=batch_size,
                          h = h,
                          hyper_parameters=op_hyper_parameters)
            current_node = GraphNode(i,
                                     op,
                                     duration=kduration,
                                     gap=space)
            nodes.append(current_node)
        root_node = nodes[0]
        return Graph(filename, env, batch_size, nodes, root_node)

    def subgraphs(self, subgraph_count: Optional[int] = None, subgraph_node_size: Optional[int] = None,
                  step: Optional[int] = None) -> \
            Tuple[List[List[GraphNode]], Dict[int, int]]:
        dummy_subgraph = False
        if subgraph_node_size is None:
            assert subgraph_count is not None
            subgraph_node_size = math.ceil(len(self.nodes) / subgraph_count)
            dummy_subgraph = True
        # subgraphs, node graph mapping
        if step is None:
            step = subgraph_node_size
        subgraphs = list()
        node_id_to_group_idx = dict()
        idx = 0
        while True:
            if idx >= len(self.nodes):
                break
            subgraph_nodes = self.nodes[idx:
                                        min(idx + subgraph_node_size, len(self.nodes))]
            subgraphs.append(subgraph_nodes)
            for node in subgraph_nodes:
                node_id_to_group_idx[node.node_id] = idx // step
            dummy_node_require = False
            while len(subgraph_nodes) < subgraph_node_size:
                subgraph_nodes.append(GraphNode.dummy_node())
                dummy_node_require = True
            if dummy_node_require:
                break
            idx += step
        # 添加dummy graph
        if dummy_subgraph and len(subgraphs) < subgraph_count:
            while(len(subgraphs) < subgraph_count):
                subgraphs.append([GraphNode.dummy_node()
                              for i in range(subgraph_node_size)])
        return subgraphs, node_id_to_group_idx


class MDataset(Dataset):
    def __init__(self, features: List[Dict], labels: List[Dict]):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]

        return x, y


def _load_graphs_of_filenames(inputs: List):
    _graphs = list()
    envs, filenames, csvs = inputs
    for env, filename, csv in zip(envs, filenames, csvs):
        filecsv = csv.loc[csv["filename"] == filename]
        graph = Graph.from_data(env, filename=filename, df=filecsv)
        logging.info(f"Loaded {filename}, {len(filecsv)} rows")
        _graphs.append(graph)
    return _graphs

def load_graphs(environment: Environment, train_or_eval: str = "train", use_dummy: bool = False, max_row: int = 100_000) -> List[Graph]:
    def _load_graphs():
        if use_dummy:
            return list(Graph.from_data(environment, dummy=True, seed=seed) for seed in range(500))
        data_dir = pathlib.Path(datasets_path) / f"{environment}" / train_or_eval
        # Load data from directory
        _graphs = list()
        if os.path.exists(str(data_dir / "merged.csv")):
            # load merged.csv
            logging.info(f"Loading merged.csv")
            csv = pds.read_csv(str(data_dir / "merged.csv"), nrows=max_row)
            csv = csv[csv["seq"] != "-1"]
            logging.info(f"Loaded merged.csv, {len(csv)} rows")
            # list all unique filenames
            filenames = csv["filename"].unique() 
            for filename in filenames:
                filecsv = csv.loc[csv["filename"] == filename]
                graph = Graph.from_data(environment, filename=filename, df=filecsv)
                logging.info(f"Loaded {filename}, {len(filecsv)} rows")
                _graphs.append(graph)
            return _graphs

        # if not merged, load data from each file
        curr_row = 0
        for filename in os.listdir(str(data_dir)):
            if not filename.endswith(".csv"):
                continue
            csv = pds.read_csv(str(data_dir / filename))
            # 删除optimizer的数据
            csv = csv[csv["seq"] != "-1"]
            logging.info(f"Loaded {filename}, {len(csv)} rows")
            graph = Graph.from_data(environment, filename=filename, df=csv)
            _graphs.append(graph)
            curr_row += len(csv)
            if curr_row >= max_row:
                break
        return _graphs

    logging.info(f"Loading graphs {train_or_eval}")
    graphs = _load_graphs()
    return graphs


def save_dataset_pkl(ds: MDataset, environment: Environment, executor: str, train_or_eval: str = "train",
                     normalization: str = 'Standard'):
    data_dir = pathlib.Path(pkl_path) / f"{environment}" / executor / train_or_eval / normalization
    if not os.path.exists(str(data_dir)):
        os.makedirs(str(data_dir))
    with open(str(data_dir / "features.pkl"), "wb") as f:
        pickle.dump(ds.features, f)
    with open(str(data_dir / "labels.pkl"), "wb") as f:
        pickle.dump(ds.labels, f)


def load_dataset_pkl(environment: Environment, executor: str, train_or_eval: str = "train",
                     normalization: str = 'Standard'):
    print(f"Loading dataset {environment} {executor} {train_or_eval} {normalization}")
    data_dir = pathlib.Path(pkl_path) / f"{environment}" / executor / train_or_eval / normalization
    with open(str(data_dir / "features.pkl"), "rb") as f:
        features = pickle.load(f)
    with open(str(data_dir / "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    return MDataset(features, labels)


def dateset_exists(environment: Environment, executor: str, train_or_eval: str = "train",
                   normalization: str = 'Standard'):
    data_dir = pathlib.Path(pkl_path) / f"{environment}" / executor / train_or_eval / normalization
    return os.path.exists(str(data_dir / "features.pkl"))


def save_scalers_pkl(scalers, environment: Environment, executor: str, train_or_eval: str = "train",
                     normalization: str = 'Standard'):
    data_dir = pathlib.Path(pkl_path) / f"{environment}" / executor / train_or_eval / normalization
    if not os.path.exists(str(data_dir)):
        os.makedirs(str(data_dir))
    with open(str(data_dir / "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)


def load_scalers_pkl(environment: Environment, executor: str, train_or_eval: str = "train",
                     normalization: str = 'Standard'):
    print(f"Loading scalers {environment} {executor} {train_or_eval}, {normalization}")
    data_dir = pathlib.Path(pkl_path) / f"{environment}" / executor / train_or_eval / normalization
    with open(str(data_dir / "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    return scalers


def save_graphs_pkl(graphs: List[Graph], environment: Environment, executor: str, train_or_eval: str = "train"):
    data_dir = pathlib.Path(pkl_path) / f"{environment}" / executor / train_or_eval
    if not os.path.exists(str(data_dir)):
        os.makedirs(str(data_dir))
    with open(str(data_dir / "graphs.pkl"), "wb") as f:
        pickle.dump(graphs, f)


def load_graphs_pkl(environment: Environment, executor: str, train_or_eval: str = "train"):
    print(f"Loading graphs {environment} {executor} {train_or_eval}")
    data_dir = pathlib.Path(pkl_path) / f"{environment}" / executor / train_or_eval
    with open(str(data_dir / "graphs.pkl"), "rb") as f:
        graphs = pickle.load(f)
    return graphs

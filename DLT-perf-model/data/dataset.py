import pathlib
import os
import pandas as pds
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict

from torch.utils.data import Dataset

from objects import Environment
from .graph import Graph

import pickle

datasets_path = str(pathlib.Path(os.getcwd()) / "datasets")
pkl_path = str(pathlib.Path(os.getcwd()) / "pkl")


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


@lru_cache(maxsize=None)
def load_graphs(environment: Environment, train_or_eval: str = "train", use_dummy: bool = False) -> List[Graph]:
    def _load_graphs():
        if use_dummy:
            return list(Graph.from_data(environment, dummy=True, seed=seed) for seed in range(500))
        data_dir = pathlib.Path(datasets_path) / f"{environment}" / train_or_eval
        # Load data from directory
        _graphs = list()
        for filename in os.listdir(str(data_dir)):
            if not filename.endswith(".csv"):
                continue
            csv = pds.read_csv(str(data_dir / filename))
            # 删除optimizer的数据
            csv = csv[csv["seq"] != "-1"]
            print(f"Loading {filename}, {len(csv)} rows")
            graph = Graph.from_data(environment, filename=filename, df=csv)
            _graphs.append(graph)
        return _graphs

    print(f"Loading graphs {train_or_eval}")
    graphs = _load_graphs()
    return graphs


def analyze_op_freq(graphs: List[Graph]):
    op_freq = defaultdict(int)
    for g in graphs:
        for node in g.nodes:
            op_freq[node.op.operator_type] += 1
    return op_freq


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

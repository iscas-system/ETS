from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import Dict
from typing import Tuple, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.nn import MSELoss

from config import Config
from data.dataset import MDataset, Graph
from executor.base_module import MModule
from executor.executor import Executor
from executor.metric import MetricUtil
from executor.util import nested_detach, pad_np_vectors
from objects import ModelType, Environment
from .gcn import GCNLayer


class GroupingBasedExecutor(Executor):
    def __init__(self, conf: Config | None = None):
        super().__init__(conf)
        self.executor_name = "GroupingBasedExecutor"

    @staticmethod
    def full_graph_feature(graph: Graph, subgraph_count: int = 10, dataset_params: Dict = {}) -> Tuple[
        Dict[str, np.ndarray], Dict]:
        subgraphs, node_id_to_group_idx = graph.subgraphs(subgraph_count=subgraph_count)

        feature_matrix = list()
        for subgraph in subgraphs:
            subgraph_features = list()
            for node in subgraph:
                node_feature = np.array(node.op.to_feature_array(
                    mode=dataset_params.get("mode", "complex")))
                subgraph_features.append(node_feature)
            if len(subgraph_features) == 0:
                feature_matrix.append(np.zeros(1))
                continue
            subgraph_features = pad_np_vectors(subgraph_features)
            feature = np.sum(subgraph_features, axis=0)
            feature = np.append(feature, len(subgraph))
            feature_matrix.append(feature)

        adjacency_matrix = list()
        for i, subgraph in enumerate(subgraphs):
            vector = np.zeros(len(subgraphs) + 1)
            for node in subgraph:
                neighbor_group_indices = list()
                for neighbor in node.neighbors:
                    neighbor_group_idx = node_id_to_group_idx[neighbor.node_id]
                    if neighbor_group_idx != i:
                        neighbor_group_indices.append(neighbor_group_idx)
                for idx in neighbor_group_indices:
                    vector[idx] = 1
            adjacency_matrix.append(vector)

        feature_matrix = pad_np_vectors(feature_matrix)

        def pad_matrix(matrix):
            if len(matrix) < subgraph_count + 1:  # optimizer_feature
                matrix.extend([np.zeros(matrix[0].shape) for _ in range(subgraph_count + 1 - len(matrix))])

        pad_matrix(feature_matrix)
        pad_matrix(adjacency_matrix)
        feature_matrix = np.array(feature_matrix)
        adjacency_matrix = np.array(adjacency_matrix)

        x = {
            "x_graph_id": graph.ID,
            "x_feature_matrix": feature_matrix,
            "x_adjacency_matrix": adjacency_matrix,
        }
        y = {
            "y_graph_id": graph.ID,
            "y_graph_duration": (graph.graph_duration,)
        }
        return x, y

    def _init_dataset(self, graphs: List[Graph]) -> MDataset:
        conf = self.conf
        X = list()
        Y = list()

        feature_matrix_maxsize = 0
        adjacency_matrix_maxsize = 0

        for graph in graphs:
            x, y = self.full_graph_feature(graph,
                                           subgraph_count=conf.dataset_subgraph_grouping_count,
                                           dataset_params=conf.dataset_params)
            feature_matrix_size = len(x["x_feature_matrix"][0])
            adjacency_matrix_size = len(x["x_adjacency_matrix"][0])
            feature_matrix_maxsize = max(feature_matrix_maxsize, feature_matrix_size)
            adjacency_matrix_maxsize = max(adjacency_matrix_maxsize, adjacency_matrix_size)

            X.append(x)
            Y.append(y)
        for x in X:
            x["x_feature_matrix"] = pad_np_vectors(x["x_feature_matrix"], maxsize=feature_matrix_maxsize)
            x["x_adjacency_matrix"] = pad_np_vectors(x["x_adjacency_matrix"], maxsize=adjacency_matrix_maxsize)

        dataset = MDataset(X, Y)
        return dataset

    @abstractmethod
    def _init_model(self) -> MModule | Any:
        pass

    @lru_cache(maxsize=None)
    def _get_scalers(self):
        train_ds = self.train_ds
        scaler_cls = self.conf.dataset_normalizer_cls
        graph_feature_array = list()
        y_array = list()

        for data in train_ds:
            feature, label = data
            x_feature_matrix = feature["x_feature_matrix"]
            assert isinstance(x_feature_matrix, list)
            graph_feature_array.extend(x_feature_matrix)
            y_array.append(label["y_graph_duration"])

        graph_feature_array = np.array(graph_feature_array)
        y_array = np.array(y_array)

        graph_feature_scaler = scaler_cls()
        graph_feature_scaler.fit(graph_feature_array)

        y_scaler = scaler_cls()
        y_scaler.fit(y_array)
        return [graph_feature_scaler, y_scaler]

    def _preprocess_dataset(self, ds: MDataset) -> MDataset:
        y_array = list()

        graph_feature_scaler, y_scaler = self.scalers
        graph_feature_arrays = list()
        for data in ds:
            feature, label = data
            # x. transform for each x feature matrix. do not merge them.
            x_feature_matrix = feature["x_feature_matrix"]
            x_feature_matrix = np.array(x_feature_matrix).astype(np.float32)
            graph_feature_array = graph_feature_scaler.transform(x_feature_matrix)
            graph_feature_arrays.append(graph_feature_array)
            # y. transform altogether
            y_array.append(label["y_graph_duration"])

        y_array = np.array(y_array).astype(np.float32)
        y_array = y_scaler.transform(y_array)

        processed_features = list()
        processed_labels = list()
        for i, data in enumerate(ds):
            feature, label = data
            x_adjacency_matrix = np.array(feature["x_adjacency_matrix"]).astype(np.float32)
            processed_features.append({
                "x_graph_id": feature["x_graph_id"],
                # "x_feature_matrix": torch.Tensor(graph_feature_arrays[i]).to(self.conf.device),
                # "x_adjacency_matrix": torch.Tensor(x_adjacency_matrix).to(self.conf.device)
                "x_feature_matrix": torch.Tensor(graph_feature_arrays[i]),
                "x_adjacency_matrix": torch.Tensor(x_adjacency_matrix)
            })
            processed_labels.append({
                "y_graph_id": label["y_graph_id"],
                # "y_graph_duration": torch.Tensor(y_array[i]).to(self.conf.device),
                "y_graph_duration": torch.Tensor(y_array[i]),
            })

        ds = MDataset(processed_features, processed_labels)
        return ds

    def to_device(self, features, labels):
        features["x_feature_matrix"] = features["x_feature_matrix"].to(self.conf.device)
        features["x_adjacency_matrix"] = features["x_adjacency_matrix"].to(self.conf.device)
        labels["y_graph_duration"] = labels["y_graph_duration"].to(self.conf.device)
        return features, labels

    def _evaluate(self, model, env: Environment, ds: MDataset) -> Dict[str, float]:
        input_batches, output_batches, eval_loss = self._dl_evaluate_pred(model, env, ds)

        batches_len = len(input_batches)

        def compute_graph_duration(_logits):
            _, y_scaler = self.scalers
            transformed: np.ndarray = y_scaler.inverse_transform(_logits)
            duration_dim = (0, 1)
            durations = transformed[:, duration_dim[0]:duration_dim[1]].sum(axis=1)
            return durations

        graph_id_to_duration_pred = defaultdict(int)
        for idx in range(batches_len):
            inputs = input_batches[idx]
            logits = output_batches[idx]
            logits = nested_detach(logits)
            logits = logits.cpu().numpy()
            graph_ids = inputs["x_graph_id"]
            graph_durations = compute_graph_duration(logits)
            for i, graph_id in enumerate(graph_ids):
                graph_duration = graph_durations[i].item()
                graph_id_to_duration_pred[graph_id] = graph_duration
        duration_metrics = MetricUtil.compute_duration_metrics(self.eval_graphs, graph_id_to_duration_pred)
        return {"eval_loss": eval_loss, **duration_metrics}


class MLPTest_GroupingBasedExecutor(GroupingBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.MLPTestGrouping

    @staticmethod
    def default_model_params() -> Dict[str, Any]:
        return {}

    @staticmethod
    def grid_search_model_params() -> Dict[str, List]:
        return {}

    def _init_model(self) -> MModule | Any:
        if self.train_mode == "single":
            sample_preprocessed_ds = self.preprocessed_train_ds
        elif self.train_mode == "meta":
            sample_preprocessed_ds = self.meta_preprocessed_train_dss[self.conf.meta_dataset_train_environments[0]]
        sample_x_dict = sample_preprocessed_ds.features[0]
        sample_y_dict = sample_preprocessed_ds.labels[0]
        shape = len(sample_x_dict["x_feature_matrix"]), len(sample_x_dict["x_feature_matrix"][0])
        return MLPTest_GroupingModel(input_shape=shape,
                                     output_dimension=len(sample_y_dict["y_graph_duration"]))


class MLPTest_GroupingModel(MModule):

    def __init__(self, input_shape, output_dimension, **kwargs):
        super().__init__(**kwargs)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=input_shape[0] * input_shape[1], out_features=128)
        self.output = torch.nn.Linear(128, output_dimension)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = X["x_feature_matrix"]
        X = self.flatten(X)
        X = self.linear1(X)
        Y = self.output(X)
        return Y

    def compute_loss(self, outputs, Y):
        graph_duration = Y["y_graph_duration"]
        loss = self.loss_fn(outputs, graph_duration)
        return loss


class GCNGroupingModel(MModule):
    def __init__(self, dim_feats, dim_h, y_graph_duration_len, n_layers, dropout):
        super(GCNGroupingModel, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(dim_feats, dim_h, F.relu, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(dim_h, dim_h, F.relu, dropout))
        # output layer
        self.layers.append(GCNLayer(dim_h, y_graph_duration_len, None, dropout))
        self.loss_fn = MSELoss()

    def forward(self, X):
        adj, features = X["x_adjacency_matrix"], X["x_feature_matrix"]
        h = features
        for layer in self.layers:
            h = layer(adj, h)
        graph_duration = torch.sum(h, dim=[1])
        return graph_duration

    def compute_loss(self, outputs, Y) -> torch.Tensor:
        y_graph_duration = Y["y_graph_duration"]
        loss = self.loss_fn(outputs, y_graph_duration)
        return loss


class GCNGroupingBasedExecutor(GroupingBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.GCNGrouping

    @staticmethod
    def default_model_params() -> Dict[str, Any]:
        return {
            "dim_h": None,
            "n_layers": 2,
            "dropout": 0.1
        }

    @staticmethod
    def grid_search_model_params() -> Dict[str, List]:
        return {
            "dim_h": [32, 64],
            "n_layers": [2, 3],
            "dropout": [0.1]
        }

    def _init_model(self) -> MModule | Any:
        if self.train_mode == "single":
            sample_preprocessed_ds = self.preprocessed_train_ds
        elif self.train_mode == "meta":
            sample_preprocessed_ds = self.meta_preprocessed_train_dss[self.conf.meta_dataset_train_environments[0]]
        sample_x_dict = sample_preprocessed_ds.features[0]
        sample_y_dict = sample_preprocessed_ds.labels[0]
        x_node_feature_size = len(sample_x_dict["x_feature_matrix"][0])
        y_graph_duration_len = len(sample_y_dict["y_graph_duration"])
        model_params = self.conf.model_params

        final_model_params = self.default_model_params()
        default_dim_h = x_node_feature_size if final_model_params.get("dim_h") is None else final_model_params.get(
            "dim_h")
        final_model_params["dim_h"] = model_params.get("dim_h", default_dim_h)
        final_model_params["n_layers"] = model_params.get("n_layers", final_model_params["n_layers"])
        final_model_params["dropout"] = model_params.get("dropout", final_model_params["dropout"])
        return GCNGroupingModel(
            dim_feats=x_node_feature_size,
            y_graph_duration_len=y_graph_duration_len,
            **final_model_params
        )

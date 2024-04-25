import logging
from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import Dict
from typing import List
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.nn import MSELoss, LSTM, GRU

from config import Config
from data import GraphNode, Graph
from data.dataset import MDataset
from executor.base_module import MModule
from executor.executor import Executor
from executor.metric import MetricUtil
from executor.util import nested_detach, pad_np_vectors
from objects import ModelType, Environment
from .gcn import GCNLayer
from .transformer import TransformerModel


class SubgraphBasedExecutor(Executor):
    def __init__(self, conf: Config | None = None):
        super().__init__(conf)
        self.executor_name = "SubgraphBasedExecutor"

    @staticmethod
    def subgraph_features(graph: Graph, subgraph_node_size: int = 10, step: int = 5, dataset_params: Dict = {}) -> \
            Tuple[List[Dict], List[Dict]]:
        subgraphs, _ = graph.subgraphs(subgraph_node_size=subgraph_node_size, step=step)
        X, Y = list(), list()

        def subgraph_feature(nodes: List[GraphNode]):
            feature_matrix = list()
            for node in nodes:
                feature = node.op.to_feature_array(
                    mode=dataset_params.get("mode", "complex"))
                feature = np.array(feature)
                feature_matrix.append(feature)

            feature_matrix = pad_np_vectors(feature_matrix)
            feature_matrix = np.array(feature_matrix)

            adj_matrix = [
                [0.] * len(nodes) for _ in range(len(nodes))
            ]
            node_id_to_node = {node.node_id: node for node in nodes}
            for curr_idx, node in enumerate(nodes):
                for neighbor in node.neighbors:
                    if neighbor.node_id not in node_id_to_node:
                        continue
                    adj_idx = nodes.index(neighbor)
                    adj_matrix[curr_idx][adj_idx] = 1.

            adj_matrix = np.array(adj_matrix)
            # x
            feature = {
                "x_graph_id": graph.ID,
                "x_node_ids": "|".join([str(node.node_id) for node in nodes]),
                "x_subgraph_feature": feature_matrix,
                "x_adj_matrix": adj_matrix
            }

            # y
            subgraph_duration = sum(node.duration + node.gap for node in subgraph)
            nodes_durations = list()
            for node in subgraph:
                node_duration_label = (
                    node.duration, node.gap
                )
                nodes_durations.append(node_duration_label)

            label = {
                "y_graph_id": graph.ID,
                "y_nodes_durations": nodes_durations,
                "y_subgraph_durations": (subgraph_duration,)
            }

            return feature, label

        for i, subgraph in enumerate(subgraphs):
            x, y = subgraph_feature(subgraph)
            X.append(x)
            Y.append(y)

        return X, Y

    def _init_dataset(self, graphs: List[Graph]) -> MDataset:
        conf = self.conf

        X = list()
        Y = list()

        subgraph_feature_maxsize = 0

        for graph in graphs:
            X_, Y_ = self.subgraph_features(graph=graph,
                                            subgraph_node_size=conf.dataset_subgraph_node_size,
                                            step=conf.dataset_subgraph_step,
                                            dataset_params=conf.dataset_params)
            for x in X_:
                subgraph_feature_size = len(x["x_subgraph_feature"][0])
                subgraph_feature_maxsize = max(subgraph_feature_maxsize, subgraph_feature_size)

            X.extend(X_)
            Y.extend(Y_)

        for x in X:
            x["x_subgraph_feature"] = pad_np_vectors(x["x_subgraph_feature"], maxsize=subgraph_feature_maxsize)

        dataset = MDataset(X, Y)
        return dataset

    @abstractmethod
    def _init_model(self) -> MModule | Any:
        pass

    @lru_cache(maxsize=None)
    def _get_scalers(self, raw_train_ds: MDataset):
        scaler_cls = self.conf.dataset_normalizer_cls

        x_subgraph_feature_array, y_nodes_durations_array, y_subgraph_durations_array = self._preprocess_required_data(
            ds=raw_train_ds)

        x_subgraph_feature_scaler = scaler_cls()
        x_subgraph_feature_scaler.fit(x_subgraph_feature_array)

        y_nodes_durations_scaler = scaler_cls()
        y_nodes_durations_scaler.fit(y_nodes_durations_array)

        y_subgraph_durations_scaler = scaler_cls()
        y_subgraph_durations_scaler.fit(y_subgraph_durations_array)

        return x_subgraph_feature_scaler, y_nodes_durations_scaler, y_subgraph_durations_scaler

    @staticmethod
    def _preprocess_required_data(ds: MDataset):
        x_subgraph_feature_array = list()
        y_nodes_durations_array = list()
        y_subgraph_durations_array = list()

        for data in ds:
            feature, label = data
            x_subgraph_feature = feature["x_subgraph_feature"]
            assert isinstance(x_subgraph_feature, list)
            x_subgraph_feature_array.extend(x_subgraph_feature)

            y_nodes_durations = label["y_nodes_durations"]
            assert isinstance(y_nodes_durations, list)
            y_nodes_durations_array.extend(y_nodes_durations)

            y_subgraph_durations = label["y_subgraph_durations"]
            y_subgraph_durations_array.append(y_subgraph_durations)

        x_subgraph_feature_array = np.array(x_subgraph_feature_array)
        y_nodes_durations_array = np.array(y_nodes_durations_array)
        y_subgraph_durations_array = np.array(y_subgraph_durations_array)
        return [x_subgraph_feature_array, y_nodes_durations_array, y_subgraph_durations_array]

    def _preprocess_dataset(self, ds: MDataset) -> MDataset:
        x_subgraph_feature_scaler, y_nodes_durations_scaler, y_subgraph_durations_scaler = self.scalers

        processed_features = list()
        processed_labels = list()

        for data in ds:
            feature, label = data
            x_subgraph_feature = feature["x_subgraph_feature"]
            assert isinstance(x_subgraph_feature, list)
            x_subgraph_feature = np.array(x_subgraph_feature).astype(np.float32)
            transformed_x_subgraph_feature = x_subgraph_feature_scaler.transform(x_subgraph_feature)

            x_adj_matrix = feature["x_adj_matrix"]
            x_adj_matrix = np.array(x_adj_matrix).astype(np.float32)

            y_nodes_durations = label["y_nodes_durations"]
            assert isinstance(y_nodes_durations, list)
            y_nodes_durations = np.array(y_nodes_durations).astype(np.float32)
            transformed_y_nodes_durations = y_nodes_durations_scaler.transform(y_nodes_durations)

            y_subgraph_durations = label["y_subgraph_durations"]
            y_subgraph_durations_array = (y_subgraph_durations,)
            y_subgraph_durations_array = y_subgraph_durations_scaler.transform(y_subgraph_durations_array)
            transformed_y_subgraph_durations = y_subgraph_durations_array[0]

            processed_features.append({
                "x_graph_id": feature["x_graph_id"],
                "x_node_ids": feature["x_node_ids"],
                # "x_subgraph_feature": torch.Tensor(transformed_x_subgraph_feature).to(device=self.conf.device),
                # "x_adj_matrix": torch.Tensor(x_adj_matrix).to(device=self.conf.device)
                "x_subgraph_feature": torch.Tensor(transformed_x_subgraph_feature),
                "x_adj_matrix": torch.Tensor(x_adj_matrix)
            })

            processed_labels.append({
                "y_graph_id": label["y_graph_id"],
                # "y_nodes_durations": torch.Tensor(transformed_y_nodes_durations).to(device=self.conf.device),
                # "y_subgraph_durations": torch.Tensor(transformed_y_subgraph_durations).to(device=self.conf.device)
                "y_nodes_durations": torch.Tensor(transformed_y_nodes_durations),
                "y_subgraph_durations": torch.Tensor(transformed_y_subgraph_durations)
            })

        ds = MDataset(processed_features, processed_labels)
        return ds

    def to_device(self, features, labels):
        features['x_subgraph_feature'] = features['x_subgraph_feature'].to(self.device)
        features['x_adj_matrix'] = features['x_adj_matrix'].to(self.device)
        features['y_nodes_durations'] = features['y_nodes_durations'].to(self.device)
        features['y_subgraph_durations'] = features['y_subgraph_durations'].to(self.device)
        return features, labels

    def _evaluate(self, model, env: Environment, ds: MDataset) -> Dict[str, float]:
        input_batches, output_batches, eval_loss = self._dl_evaluate_pred(model, env, ds)

        def compute_graph_nodes_durations(outputs_, node_ids_str_):
            # if self.train_mode == "single":
            #     raw_ds = self.train_ds
            # elif self.train_mode == "meta":
            #     raw_ds = self.meta_train_dss[env]
            x_subgraph_feature_scaler, y_nodes_durations_scaler, y_subgraph_durations_scaler = self.scalers
            node_to_durations = defaultdict(list)
            for i, output_ in enumerate(outputs_):
                node_ids = node_ids_str_[i]
                node_ids_ = node_ids.split("|")
                assert len(output_) == len(node_ids_)
                transformed: np.ndarray = y_nodes_durations_scaler.inverse_transform(output_)
                for i, node_id in enumerate(node_ids_):
                    node_to_durations[node_id].append(np.sum(transformed[i]))
            node_to_duration = {k: np.average(v) for k, v in node_to_durations.items()}
            return node_to_duration

        graph_id_to_node_to_duration = defaultdict(lambda: defaultdict(list))
        for inputs, outputs in zip(input_batches, output_batches):
            outputs = nested_detach(outputs)
            outputs = outputs.cpu().numpy()
            graph_ids = inputs["x_graph_id"]
            graph_groups = defaultdict(list)
            for i, graph_id in enumerate(graph_ids):
                graph_groups[graph_id].append(i)

            for graph_id, indices in graph_groups.items():
                group_x_node_ids = [v for i, v in enumerate(inputs["x_node_ids"]) if i in indices]
                group_outputs = [v for i, v in enumerate(outputs) if i in indices]
                node_to_durations = compute_graph_nodes_durations(group_outputs, group_x_node_ids)
                for node, duration in node_to_durations.items():
                    graph_id_to_node_to_duration[graph_id][node].append(duration)
        graph_id_to_duration_pred = dict()
        # TODO check this!!!
        for graph_id, node_to_duration in graph_id_to_node_to_duration.items():
            duration_pred = 0
            for _, duration_preds in node_to_duration.items():
                duration_pred += np.average(duration_preds)
            graph_id_to_duration_pred[graph_id] = duration_pred
        if self.train_mode == "single":
            eval_graphs = self.eval_graphs
        elif self.train_mode == "meta":
            eval_graphs = self.meta_eval_graphs[env]
        duration_metrics = MetricUtil.compute_duration_metrics(eval_graphs, graph_id_to_duration_pred)
        return {"eval_loss": eval_loss, **duration_metrics}


class MLPTest_SubgraphBasedExecutor(SubgraphBasedExecutor):
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
        x_node_feature_count = len(sample_x_dict["x_subgraph_feature"])
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        y_nodes_duration_count = len(sample_y_dict["y_nodes_durations"])
        y_nodes_duration_size = len(sample_y_dict["y_nodes_durations"][0])
        return MLPTest_SubgraphModel(x_node_feature_count,
                                     x_node_feature_size,
                                     y_nodes_duration_count,
                                     y_nodes_duration_size)


class MLPTest_SubgraphModel(MModule):

    def __init__(self, x_node_feature_count, x_node_feature_size, y_nodes_duration_count, y_nodes_duration_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.x_node_feature_count, self.x_node_feature_size, self.y_nodes_duration_count, self.y_nodes_duration_size \
            = x_node_feature_count, x_node_feature_size, y_nodes_duration_count, y_nodes_duration_size
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=self.x_node_feature_count * self.x_node_feature_size,
                                       out_features=64)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=64,
                                       out_features=32)
        self.relu2 = torch.nn.ReLU()
        self.output = torch.nn.Linear(32, self.y_nodes_duration_count * self.y_nodes_duration_size)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = X["x_subgraph_feature"]
        X = self.flatten(X)
        X = self.linear1(X)
        X = self.relu1(X)
        X = self.linear2(X)
        X = self.relu2(X)
        Y = self.output(X)
        Y = torch.reshape(Y, (-1, self.y_nodes_duration_count, self.y_nodes_duration_size))
        return Y

    def compute_loss(self, outputs, Y):
        nodes_durations = Y["y_nodes_durations"]
        loss = self.loss_fn(outputs, nodes_durations)
        return loss


class TransformerSubgraphBasedExecutor(SubgraphBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.Transformer

    @staticmethod
    def default_model_params() -> Dict[str, Any]:
        nhead: int = 8
        d_hid: int = 512
        nlayers: int = 6
        dropout: float = 0.5
        return {
            "nhead": nhead,
            "d_hid": d_hid,
            "nlayers": nlayers,
            "dropout": dropout
        }

    @staticmethod
    def grid_search_model_params() -> Dict[str, List]:
        return {
            "nhead": [4, 8],
            "d_hid": [1024, 2048],
            "nlayers": [4, 6, 8],
            "dropout": [0.2, 0.5],
        }

    @staticmethod
    def grid_search_transfer_params() -> Dict[str, List]:
        return {
            "freeze_layers": [2, 3],
            "reinit_proj": [False, True]
        }

    def _init_model(self) -> MModule | Any:
        if self.train_mode == "single":
            sample_preprocessed_ds = self.preprocessed_train_ds
        elif self.train_mode == "meta":
            sample_preprocessed_ds = self.meta_preprocessed_train_dss[self.conf.meta_dataset_train_environments[0]]
        sample_x_dict = sample_preprocessed_ds.features[0]
        sample_y_dict = sample_preprocessed_ds.labels[0]
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        nodes_durations_len = len(sample_y_dict["y_nodes_durations"][0])
        model_params = self.conf.model_params
        final_params = self.default_model_params()
        for k, v in final_params.items():
            final_params[k] = model_params.get(k, v)

        nhead = final_params["nhead"]
        while x_node_feature_size % nhead != 0:
            nhead -= 1
        if nhead != final_params["nhead"]:
            final_params["nhead"] = nhead
            logging.info(f"Transformer nhead set to {nhead}.")
            self.conf.model_params["nhead"] = nhead

        return TransformerModel(
            d_model=x_node_feature_size,
            output_d=nodes_durations_len,
            **final_params
        )


class LSTMModel(MModule):
    def __init__(self, feature_size, nodes_durations_len, num_layers, bidirectional, **kwargs):
        super().__init__(**kwargs)
        self.lstm = LSTM(input_size=feature_size, hidden_size=feature_size, num_layers=num_layers, batch_first=True,
                         bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.project = torch.nn.Linear(in_features=feature_size * num_directions, out_features=nodes_durations_len)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = X["x_subgraph_feature"]
        out, _ = self.lstm(X)
        Y = self.project(out)
        return Y

    def compute_loss(self, outputs, Y):
        node_durations = Y["y_nodes_durations"]
        loss = self.loss_fn(outputs, node_durations)
        return loss


class LSTMSubgraphBasedExecutor(SubgraphBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.LSTM

    @staticmethod
    def default_model_params() -> Dict[str, Any]:
        return {
            "num_layers": 4,
            "bidirectional": True,
        }

    @staticmethod
    def grid_search_model_params() -> Dict[str, List]:
        return {
            "num_layers": [2, 4],
            "bidirectional": [True, False],
        }

    def _init_model(self) -> MModule | Any:
        if self.train_mode == "single":
            sample_preprocessed_ds = self.preprocessed_train_ds
        elif self.train_mode == "meta":
            sample_preprocessed_ds = self.meta_preprocessed_train_dss[self.conf.meta_dataset_train_environments[0]]
        sample_x_dict = sample_preprocessed_ds.features[0]
        sample_y_dict = sample_preprocessed_ds.labels[0]
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        y_nodes_durations_len = len(sample_y_dict["y_nodes_durations"][0])
        model_params = self.conf.model_params
        final_params = self.default_model_params()
        for k, v in final_params.items():
            final_params[k] = model_params.get(k, v)
        return LSTMModel(
            feature_size=x_node_feature_size,
            nodes_durations_len=y_nodes_durations_len,
            **final_params
        )


class GRUModel(MModule):
    def __init__(self, feature_size, nodes_durations_len, num_layers, bidirectional, **kwargs):
        super().__init__(**kwargs)
        self.gru = GRU(input_size=feature_size, hidden_size=feature_size, num_layers=num_layers, batch_first=True,
                       bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.project = torch.nn.Linear(in_features=feature_size * num_directions, out_features=nodes_durations_len)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = X["x_subgraph_feature"]
        out, _ = self.gru(X)
        Y = self.project(out)
        return Y

    def compute_loss(self, outputs, Y):
        node_durations = Y["y_nodes_durations"]
        loss = self.loss_fn(outputs, node_durations)
        return loss


class GRUSubgraphBasedExecutor(SubgraphBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.GRU

    @staticmethod
    def default_model_params() -> Dict[str, Any]:
        return {
            "num_layers": 4,
            "bidirectional": True,
        }

    @staticmethod
    def grid_search_model_params() -> Dict[str, List]:
        return {
            "num_layers": [1, 2, 4],
            "bidirectional": [True, False],
        }

    def _init_model(self) -> MModule | Any:
        if self.train_mode == "single":
            sample_preprocessed_ds = self.preprocessed_train_ds
        elif self.train_mode == "meta":
            sample_preprocessed_ds = self.meta_preprocessed_train_dss[self.conf.meta_dataset_train_environments[0]]
        sample_x_dict = sample_preprocessed_ds.features[0]
        sample_y_dict = sample_preprocessed_ds.labels[0]
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        y_nodes_durations_len = len(sample_y_dict["y_nodes_durations"][0])
        model_params = self.conf.model_params
        final_params = self.default_model_params()
        for k, v in final_params.items():
            final_params[k] = model_params.get(k, v)
        return GRUModel(
            feature_size=x_node_feature_size,
            nodes_durations_len=y_nodes_durations_len,
            **final_params
        )


class GCNSubgraphModel(MModule):
    def __init__(self, dim_feats, dim_h, dim_out, n_layers, dropout):
        super(GCNSubgraphModel, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(dim_feats, dim_h, F.relu, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(dim_h, dim_h, F.relu, dropout))
        # output layer
        self.layers.append(GCNLayer(dim_h, dim_out, None, dropout))
        self.loss_fn = MSELoss()

    def forward(self, X):
        adj, features = X["x_adj_matrix"], X["x_subgraph_feature"]
        h = features
        for layer in self.layers:
            h = layer(adj, h)
        return h

    def compute_loss(self, outputs, Y) -> torch.Tensor:
        y_nodes_durations = Y["y_nodes_durations"]
        loss = self.loss_fn(outputs, y_nodes_durations)
        return loss


class GCNSubgraphBasedExecutor(SubgraphBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.GCNSubgraph

    @staticmethod
    def default_model_params() -> Dict[str, Any]:
        return {
            "dim_h": None,
            "n_layers": 2,
            "dropout": 0.1,
        }

    @staticmethod
    def grid_search_model_params() -> Dict[str, List]:
        return {
            "dim_h": [32, 64],
            "n_layers": [2, 4],
            "dropout": [0.2, 0.5],
        }

    def _init_model(self) -> MModule | Any:
        if self.train_mode == "single":
            sample_preprocessed_ds = self.preprocessed_train_ds
        elif self.train_mode == "meta":
            sample_preprocessed_ds = self.meta_preprocessed_train_dss[self.conf.meta_dataset_train_environments[0]]
        sample_x_dict = sample_preprocessed_ds.features[0]
        sample_y_dict = sample_preprocessed_ds.labels[0]
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        y_nodes_durations_len = len(sample_y_dict["y_nodes_durations"][0])
        model_params = self.conf.model_params
        final_params = self.default_model_params()
        for k, v in final_params.items():
            final_params[k] = model_params.get(k, v)
        if final_params["dim_h"] is None:
            final_params["dim_h"] = x_node_feature_size
        # final_params["device"] = self.conf.device
        return GCNSubgraphModel(
            dim_feats=x_node_feature_size,
            dim_out=y_nodes_durations_len,
            **final_params
        )

from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import torch

from predictor.base_module import pad_np_vectors, nested_detach
from predictor.data import Graph, GraphNode, MDataset


def subgraph_features(graph: Graph, subgraph_node_size: int = 10, step: int = 5, dataset_params: Dict = {}) -> \
        Tuple[List[Dict], List[Dict]]:
    subgraphs, _ = graph.subgraphs(
        subgraph_node_size=subgraph_node_size, step=step)
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
        for curr_idx, node in enumerate(nodes):
            if curr_idx + 1 < len(nodes):
                adj_matrix[curr_idx][curr_idx + 1] = 1.

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


def init_dataset(graphs: List[Graph]) -> MDataset:
    X = list()
    Y = list()

    subgraph_feature_maxsize = 0

    for graph in graphs:
        X_, Y_ = subgraph_features(graph=graph,
                                   subgraph_node_size=12,
                                   step=3,
                                   dataset_params={
                                       "duration_summed": False
                                   })
        for x in X_:
            subgraph_feature_size = len(x["x_subgraph_feature"][0])
            subgraph_feature_maxsize = max(
                subgraph_feature_maxsize, subgraph_feature_size)

        X.extend(X_)
        Y.extend(Y_)

    for x in X:
        x["x_subgraph_feature"] = pad_np_vectors(
            x["x_subgraph_feature"], maxsize=subgraph_feature_maxsize)

    dataset = MDataset(X, Y)
    return dataset


def preprocess_dataset(ds: MDataset, scalers) -> MDataset:
    x_subgraph_feature_scaler, y_nodes_durations_scaler, y_subgraph_durations_scaler = scalers

    processed_features = list()
    processed_labels = list()

    for data in ds:
        feature, label = data
        x_subgraph_feature = feature["x_subgraph_feature"]
        assert isinstance(x_subgraph_feature, list)
        x_subgraph_feature = np.array(x_subgraph_feature).astype(np.float32)
        transformed_x_subgraph_feature = x_subgraph_feature_scaler.transform(
            x_subgraph_feature)

        x_adj_matrix = feature["x_adj_matrix"]
        x_adj_matrix = np.array(x_adj_matrix).astype(np.float32)

        y_nodes_durations = label["y_nodes_durations"]
        assert isinstance(y_nodes_durations, list)
        y_nodes_durations = np.array(y_nodes_durations).astype(np.float32)
        transformed_y_nodes_durations = y_nodes_durations_scaler.transform(
            y_nodes_durations)

        y_subgraph_durations = label["y_subgraph_durations"]
        y_subgraph_durations_array = (y_subgraph_durations,)
        y_subgraph_durations_array = y_subgraph_durations_scaler.transform(
            y_subgraph_durations_array)
        transformed_y_subgraph_durations = y_subgraph_durations_array[0]

        processed_features.append({
            "x_graph_id": feature["x_graph_id"],
            "x_node_ids": feature["x_node_ids"],
            "x_subgraph_feature": torch.Tensor(transformed_x_subgraph_feature),
            "x_adj_matrix": torch.Tensor(x_adj_matrix)
        })

        processed_labels.append({
            "y_graph_id": label["y_graph_id"],
            "y_nodes_durations": torch.Tensor(transformed_y_nodes_durations),
            "y_subgraph_durations": torch.Tensor(transformed_y_subgraph_durations)
        })

    ds = MDataset(processed_features, processed_labels)
    return ds


def compute_graph_nodes_durations(outputs_, node_ids_str_, scalers):
    x_subgraph_feature_scaler, y_nodes_durations_scaler, y_subgraph_durations_scaler = scalers
    node_to_durations = defaultdict(list)
    for i, output_ in enumerate(outputs_):
        node_ids = node_ids_str_[i]
        node_ids_ = node_ids.split("|")
        assert len(output_) == len(node_ids_)
        transformed: np.ndarray = y_nodes_durations_scaler.inverse_transform(
            output_)
        for i, node_id in enumerate(node_ids_):
            # runtime 和 gap相加了
            # node_to_durations[node_id].append(np.sum(transformed[i]))
            node_to_durations[node_id].append(transformed[i])
    node_to_duration = {}
    for k, v in node_to_durations.items():
        duration = gap = 0
        for i in v:
            duration += i[0]
            gap += i[1]
        node_to_duration[k] = (duration / len(v), gap / len(v))
    # node_to_duration = {k: np.average(v)
    #                     for k, v in node_to_durations.items()}
    return node_to_duration


def compute_durations(input_batches, output_batches, scalers, eval_graphs, detail=True):
    graph_id_to_node_to_duration = defaultdict(lambda: defaultdict(list))
    for inputs, outputs in zip(input_batches, output_batches):
        outputs = nested_detach(outputs)
        outputs = outputs.cpu().numpy()
        graph_ids = inputs["x_graph_id"]
        graph_groups = defaultdict(list)
        for i, graph_id in enumerate(graph_ids):
            graph_groups[graph_id].append(i)

        for graph_id, indices in graph_groups.items():
            group_x_node_ids = [v for i, v in enumerate(
                inputs["x_node_ids"]) if i in indices]
            group_outputs = [v for i, v in enumerate(outputs) if i in indices]
            node_to_durations = compute_graph_nodes_durations(
                group_outputs, group_x_node_ids, scalers)
            for node, duration in node_to_durations.items():
                graph_id_to_node_to_duration[graph_id][node].append(duration)

    graph_id_to_duration_pred = dict()
    # average
    for graph_id, node_to_duration in graph_id_to_node_to_duration.items():
        duration_pred = 0
        for _, duration_preds in node_to_duration.items():
            duration_pred += np.average(np.sum(duration_preds))
        graph_id_to_duration_pred[graph_id] = duration_pred

    for graph in eval_graphs:
        pred = graph_id_to_duration_pred[graph.ID]
        graph.graph_duration_pred = pred

    for graph_id, node_to_duration in graph_id_to_node_to_duration.items():
        for tmp in eval_graphs:
            if tmp.ID == graph_id:
                graph = tmp
                for node in graph.nodes:
                    # pred_duration = np.average(node_to_duration[str(node.node_id)])
                    pred_duration = node_to_duration[str(node.node_id)][0]
                    node.duration_pred = pred_duration[0]
                    node.gap_pred = pred_duration[1]
                break
    return eval_graphs


def to_device(device, features, labels):
    features['x_subgraph_feature'] = features['x_subgraph_feature'].to(
        device)
    features['x_adj_matrix'] = features['x_adj_matrix'].to(device)
    labels['y_nodes_durations'] = labels['y_nodes_durations'].to(device)
    labels['y_subgraph_durations'] = labels['y_subgraph_durations'].to(
        device)
    return features, labels

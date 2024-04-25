from typing import List, Dict

import numpy as np

from data import Graph


class MetricUtil:
    @staticmethod
    def compute_duration_metrics(graphs: List[Graph], graph_id_to_duration_pred: Dict[str, float]) -> Dict:
        y_hat, y = list(), list()
        for graph in graphs:
            pred = graph_id_to_duration_pred[graph.ID]
            ground_truth = graph.graph_duration
            y_hat.append(pred)
            y.append(ground_truth)
        y_hat = np.array(y_hat)
        y = np.array(y)
        MRE = np.sum(np.abs(y - y_hat) / y) / len(y)
        MAE = np.sum(np.abs(y - y_hat)) / np.sum(y)
        
        y = y/1000
        y_hat = y_hat/1000
        RMSE = np.sqrt(np.sum(np.power(y - y_hat, 2)) / len(y))
        return {
            "MRE": MRE,
            "MAE": MAE,
            "RMSE": RMSE
        }
        
    @staticmethod
    def mre(y_hat: np.ndarray, y: np.ndarray) -> float:
        y = np.array(y)
        y_hat = np.array(y_hat)
        return np.sum(np.abs(y - y_hat) / y) / len(y)

    @staticmethod
    def rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
        y = np.array(y)
        y_hat = np.array(y_hat)
        y = y/1000
        y_hat = y_hat/1000
        return np.sqrt(np.sum(np.power(y - y_hat, 2)) / len(y))

import os

from predictor.base_module import MModule
import torch
from torch.nn import LSTM, L1Loss
from typing import Dict, List, Any, Union

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
predictors_path = os.path.join(BASE_DIR, 'data/predictors')


class LSTMModel(MModule):
    def __init__(self, feature_size, nodes_durations_len, num_layers, bidirectional, **kwargs):
        super().__init__(**kwargs)
        self.lstm = LSTM(input_size=feature_size, hidden_size=feature_size, num_layers=num_layers, batch_first=True,
                         bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.project = torch.nn.Linear(
            in_features=feature_size * num_directions, out_features=nodes_durations_len)
        self.loss_fn = L1Loss()

    @staticmethod
    def grid_search_model_params() -> Dict[str, List[Any]]:
        return {
            "num_layers": [4, 6, 8],
            "bidirectional": [True, False],
            "learning_rate": [1e-4, 1e-5],
            'batch_size': [32, 64],
            'epochs': [20],
            'optimizer': ['Adam', 'SGD'],
        }

    def forward(self, X):
        X = X["x_subgraph_feature"]
        out, _ = self.lstm(X)
        Y = self.project(out)
        return Y

    def compute_loss(self, outputs, Y):
        node_durations = Y["y_nodes_durations"]
        loss = self.loss_fn(outputs, node_durations)
        return loss


def init_LSTM_model() -> Union[MModule, Any]:
    def default_model_params() -> Dict[str, Any]:
        return {
            "num_layers": 4,
            "bidirectional": True,
        }

    x_node_feature_size = 66
    y_nodes_durations_len = 2
    print(x_node_feature_size, y_nodes_durations_len)
    model_params = {}
    final_params = default_model_params()
    for k, v in final_params.items():
        final_params[k] = model_params.get(k, v)
    print(final_params)
    return LSTMModel(
        feature_size=x_node_feature_size,
        nodes_durations_len=y_nodes_durations_len,
        **final_params
    )


def load_pretrained_predictor(env: str) -> MModule:
    model_path = os.path.join(predictors_path, env)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {env} not found.")

    files = os.listdir(model_path)
    files = [file for file in files if file.endswith('.pth')]
    if len(files) == 0:
        raise FileNotFoundError(f"Model {env} not found.")
    import __main__
    setattr(__main__, "LSTMModel", LSTMModel)
    predictor = torch.load(os.path.join(model_path, files[0]))
    return predictor

import json
from typing import Dict, List

import torch.cuda
from sklearn import preprocessing

from objects import Environment, OptimizerType, ModelType


class TransferConfigMixin:
    def __init__(self, transfer_config_js, **kwargs):
        super().__init__(**kwargs)
        self.transfer_params: Dict | None = transfer_config_js.get(
            "transfer_params", None)


class DatasetConfigMixin:
    def __init__(self, dataset_config_js, **kwargs):
        super().__init__(**kwargs)
        meta_configs = dataset_config_js.get("meta_configs", dict())
        self.dataset_environment_str: str = dataset_config_js.get("dataset_environment_str",
                                                                  "RTX2080Ti_CPU100")
        self.meta_dataset_train_environment_strs: [str] = meta_configs.get("meta_dataset_train_environment_strs",
                                                                           ["RTX2080Ti_CPU100"])
        self.meta_dataset_eval_environment_strs: [str] = meta_configs.get("meta_dataset_eval_environment_strs",
                                                                          ["RTX2080Ti_CPU100"])
        self.dataset_environment: Environment = Environment.from_str(
            self.dataset_environment_str)
        self.meta_dataset_train_environments: List[Environment] = [Environment.from_str(s) for s in
                                                                   self.meta_dataset_train_environment_strs]
        self.meta_dataset_eval_environments: List[Environment] = [Environment.from_str(s) for s in
                                                                  self.meta_dataset_eval_environment_strs]
        self.dataset_normalization = dataset_config_js.get(
            "dataset_normalization", "Standard")
        if self.dataset_normalization == "Standard":
            self.dataset_normalizer_cls = preprocessing.StandardScaler
        elif self.dataset_normalization == "MinMax":
            self.dataset_normalizer_cls = preprocessing.MinMaxScaler
        else:
            raise ValueError(
                f"Invalid dataset_normalization: {self.dataset_normalization}")
        self.dataset_subgraph_node_size = dataset_config_js.get(
            "dataset_subgraph_node_size", 10)
        self.dataset_subgraph_step = dataset_config_js.get(
            "dataset_subgraph_step", 5)
        self.dataset_subgraph_grouping_count = dataset_config_js.get(
            "dataset_subgraph_grouping_count", 10)
        self.dataset_params = dataset_config_js.get("dataset_params", dict())
        self.dataset_train_proportion = dataset_config_js.get(
            "train_ds_proportion", 0.7)
        self.dataset_dummy = dataset_config_js.get("dataset_dummy", False)

    def identifier(self) -> str:
        dataset_param_list = list()
        for k, v in self.dataset_params.items():
            dataset_param_list.append(f"{k}_{v}")
        s = f"{self.dataset_normalization}"
        if len(dataset_param_list) > 0:
            dataset_param_str = "|".join(dataset_param_list)
            s += f"|{dataset_param_str}"
        return s


class ModelConfigMixin:
    def __init__(self, model_config_js, **kwargs):
        super().__init__(**kwargs)
        self.model_type_str = model_config_js.get("model", "MLP")
        self.model_type: ModelType = ModelType[self.model_type_str]
        self.model_params = model_config_js.get("model_params", dict())
        self.resume_from_ckpt = model_config_js.get("resume_from_ckpt", None)


class DeviceConfigMixin:
    def __init__(self, device_config_js, **kwargs):
        super().__init__(**kwargs)
        if torch.cuda.is_available():
            self.device = device_config_js.get("device", "cuda:0")
        else:
            self.device = device_config_js.get("device_type", "cpu")


class Config(DatasetConfigMixin, ModelConfigMixin, DeviceConfigMixin, TransferConfigMixin):

    @staticmethod
    def from_dict(d):
        return Config(d)

    def to_dict(self):
        return self.raw_config

    @staticmethod
    def from_file(config_filepath):
        with open(config_filepath) as f:
            config_js = json.load(f)
        return Config.from_dict(config_js)

    def __init__(self, train_config_js):
        super().__init__(dataset_config_js=train_config_js,
                         model_config_js=train_config_js,
                         device_config_js=train_config_js,
                         transfer_config_js=train_config_js)
        self.raw_config = train_config_js
        # training
        self.all_seed = train_config_js.get("all_seed", 42)
        self.num_train_epochs = train_config_js.get("epochs", 50)
        self.batch_size = train_config_js.get("batch_size", 16)
        self.logging_steps = train_config_js.get("logging_steps", 100)
        self.eval_steps = train_config_js.get("eval_steps", 100)
        self.load_best_model_at_end = train_config_js.get(
            "load_best_model_at_end", True)
        # self.save_strategy = train_config_js.get("save_strategy", "epoch")
        self.optimizer_cls_str = train_config_js.get("optimizer", "Adam")
        self.optimizer_cls = OptimizerType[train_config_js.get(
            "optimizer", "Adam")].value
        self.learning_rate = train_config_js.get("learning_rate", 1e-3)
        meta_configs = train_config_js.get("meta_configs", 1e-3)
        self.meta_base_learning_rate = meta_configs.get("learning_rate", 5e-4)
        self.meta_train_steps = meta_configs.get("meta_train_steps", 1000)
        self.meta_task_per_step = meta_configs.get("meta_task_per_step", 8)
        self.meta_fast_adaption_step = meta_configs.get(
            "meta_fast_adaption_step", 5)
        self.meta_adaption_batch_size = meta_configs.get(
            "meta_adaption_batch_size", 128)
        self.meta_shots = meta_configs.get("meta_shots", 16)
        self.meta_learner_learning_rate = meta_configs.get(
            "meta_learning_rate", 1e-3)


dataset_subgraph_node_sizes = [10, 20, 50]
dataset_subgraph_grouping_counts = [10, 20, 30]

train_configs = {
    ModelType.MLP: {
        "model": "MLP",
        "all_seed": 42,
        # "dataset_environment_str": "TEST_CPU100",
        # "dataset_environment_str": "RTX2080Ti_CPU100",
        "dataset_environment_str": "T4_CPU100",
        "dataset_normalization": "Standard",
        "dataset_params": {
            "duration_summed": False,
        },
        "dataset_dummy": False,
        "batch_size": 512,
        "eval_steps": 500,
        "learning_rate": 1e-3,
        "epochs": 5,
        "optimizer": "Adam",
        "meta_configs": {
            "learning_rate": 0.005,
            "meta_learning_rate": 0.001,
            "meta_train_steps": 1000,
            "meta_task_per_step": 8,
            "meta_fast_adaption_step": 5,
            # "meta_dataset_train_environment_strs": ["TEST_CPU100"],
            # "meta_dataset_eval_environment_strs": ["TEST_CPU100"],
            "meta_dataset_train_environment_strs": ["T4_CPU100"],
            "meta_dataset_eval_environment_strs": ["T4_CPU100"],
            # "meta_dataset_train_environment_strs": ["RTX2080Ti_CPU100"],
            # "meta_dataset_eval_environment_strs": ["RTX2080Ti_CPU100"],

        },
    },
    ModelType.MLPTestSubgraph: {
        "model": "MLPTestSubgraph",
        "all_seed": 42,
        "dataset_environment_str": "T4_CPU100",
        "dataset_normalization": "MinMax",
        "dataset_params": {
            "duration_summed": False,
        },
        "dataset_dummy": False,
        "batch_size": 16,
        "eval_steps": 100,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "learning_rate": 0.005,
            "meta_learning_rate": 0.001,
            "meta_train_steps": 1000,
            "meta_task_per_step": 8,
            "meta_fast_adaption_step": 5,
            "meta_dataset_train_environment_strs": ["T4_CPU100"],
            "meta_dataset_eval_environment_strs": ["T4_CPU100"],
        },
    },
    ModelType.PerfNet: {
        "model": "PerfNet",
        "dataset_environment_str": "RTX2080Ti_CPU100",
        "meta_dataset_environment_strs": ["RTX2080Ti_CPU100"],
        "dataset_normalization": "Standard",
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False,
        },
        "dataset_dummy": True,
        "batch_size": 16,
        "eval_steps": 100,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "learning_rate": 0.005,
            "meta_learning_rate": 0.001,
            "meta_train_steps": 1000,
            "meta_task_per_step": 8,
            "meta_fast_adaption_step": 5,
            "meta_dataset_train_environment_strs": ["RTX2080Ti_CPU100"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti_CPU100"],
        },
    },
    ModelType.GBDT: {
        "model": "PerfNet",
        "dataset_environment_str": "RTX2080Ti_CPU100",
        "meta_dataset_environment_strs": ["RTX2080Ti_CPU100"],
        "dataset_normalization": "Standard",
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": True,
        },
        "dataset_dummy": True,
        "batch_size": 16,
        "eval_steps": 100,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "learning_rate": 0.005,
            "meta_learning_rate": 0.001,
            "meta_train_steps": 1000,
            "meta_task_per_step": 8,
            "meta_fast_adaption_step": 5,
            "meta_dataset_train_environment_strs": ["RTX2080Ti_CPU100"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti_CPU100"],
        },
    },
    ModelType.LSTM: {
        "model": "LSTM",
        "dataset_environment_str": "RTX2080Ti_CPU100",
        "meta_dataset_environment_strs": ["RTX2080Ti_CPU100"],
        "dataset_normalization": "Standard",
        "dataset_subgraph_node_size": dataset_subgraph_node_sizes,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False,
        },
        "model_params": {
            "num_layers": 5,
            "bidirectional": True
        },
        "dataset_dummy": True,
        "batch_size": 16,
        "eval_steps": 100,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "learning_rate": 0.005,
            "meta_learning_rate": 0.001,
            "meta_train_steps": 1000,
            "meta_task_per_step": 8,
            "meta_fast_adaption_step": 5,
            "meta_dataset_train_environment_strs": ["RTX2080Ti_CPU100"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti_CPU100"],
        },
    },
    ModelType.GRU: {
        "model": "GRU",
        "dataset_environment_str": "RTX2080Ti_CPU100",
        "meta_dataset_environment_strs": ["RTX2080Ti_CPU100"],
        "dataset_normalization": "Standard",
        "dataset_subgraph_node_size": dataset_subgraph_node_sizes,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False,
        },
        "model_params": {
            "num_layers": 5,
            "bidirectional": True
        },
        "dataset_dummy": True,
        "batch_size": 16,
        "eval_steps": 100,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "learning_rate": 0.005,
            "meta_learning_rate": 0.001,
            "meta_train_steps": 1000,
            "meta_task_per_step": 8,
            "meta_fast_adaption_step": 5,
            "meta_dataset_train_environment_strs": ["RTX2080Ti_CPU100"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti_CPU100"],
        },
    },
    ModelType.MLPTestGrouping: {
        "model": "MLPTestGrouping",
        "all_seed": 42,
        "dataset_environment_str": "RTX2080Ti_CPU100",
        "dataset_normalization": "Standard",
        "dataset_subgraph_grouping_count": dataset_subgraph_grouping_counts,
        "dataset_params": {
            "duration_summed": False,
        },
        "dataset_dummy": True,
        "batch_size": 16,
        "eval_steps": 100,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "learning_rate": 0.005,
            "meta_learning_rate": 0.001,
            "meta_train_steps": 1000,
            "meta_fast_adaption_step": 5,
            "meta_dataset_train_environment_strs": ["RTX2080Ti_CPU100"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti_CPU100"],
        },
    },
    ModelType.GCNGrouping: {
        "model": "GCNGrouping",
        "dataset_environment_str": "RTX2080Ti_CPU100",
        "dataset_normalization": "Standard",
        "dataset_subgraph_grouping_count": dataset_subgraph_grouping_counts,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False,
        },
        "dataset_dummy": True,
        "batch_size": 16,
        "eval_steps": 100,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "learning_rate": 0.005,
            "meta_learning_rate": 0.001,
            "meta_train_steps": 1000,
            "meta_fast_adaption_step": 5,
            "meta_dataset_train_environment_strs": ["RTX2080Ti_CPU100"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti_CPU100"],
        },
    },
    ModelType.GCNSubgraph: {
        "model": "GCNGrouping",
        "dataset_environment_str": "RTX2080Ti_CPU100",
        "dataset_normalization": "Standard",
        "dataset_subgraph_node_size": dataset_subgraph_node_sizes,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False,
        },
        "dataset_dummy": True,
        "batch_size": 16,
        "eval_steps": 100,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "learning_rate": 0.005,
            "meta_learning_rate": 0.001,
            "meta_train_steps": 1000,
            "meta_task_per_step": 8,
            "meta_fast_adaption_step": 5,
            "meta_dataset_train_environment_strs": ["RTX2080Ti_CPU100"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti_CPU100"],
        },
    },
    ModelType.Transformer: {
        "model": "Transformer",
        "dataset_environment_str": "RTX2080Ti_CPU100",
        "dataset_normalization": "Standard",
        "dataset_subgraph_node_size": dataset_subgraph_node_sizes,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False,
        },
        "model_params": {
            "nlayers": 6,
            "d_hid": 64,
            "dropout": 0.0
        },
        "dataset_dummy": True,
        "batch_size": 16,
        "eval_steps": 100,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "learning_rate": 0.005,
            "meta_learning_rate": 0.001,
            "meta_train_steps": 1000,
            "meta_task_per_step": 8,
            "meta_fast_adaption_step": 5,
            "meta_dataset_train_environment_strs": ["RTX2080Ti_CPU100"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti_CPU100"],
        },
    },
}

transfer_configs = {
    ModelType.Transformer: {
        "model": "Transformer",
        "dataset_environment_str": "RTX2080Ti_CPU100",
        "dataset_normalization": "Standard",
        "dataset_subgraph_node_size": dataset_subgraph_node_sizes,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False,
        },
        "transfer_params": {
            "freeze_layers": 3,
            "reinit_proj": True
        },
        "dataset_dummy": True,
        "batch_size": 16,
        "eval_steps": 100,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "resume_from_ckpt": "2023-06-09_14-57-45/ckpt_300.pth"
    },
}

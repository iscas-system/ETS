import json
import logging
import time
from itertools import product
from typing import List, Dict

import numpy as np

from config import Config, train_configs, transfer_configs
from executor import get_executor_cls
from logger import init_logging
from objects import ModelType

init_logging()


def launch_grid_search(train_model: ModelType, _configs: Dict[ModelType, Dict], on_common_params: bool = True,
                       model_specific_gs_params_name="model_params"):
    conf_dict = _configs[train_model]
    executor_cls = get_executor_cls(model_type=train_model)

    grid_search_params = dict()
    for k, v in conf_dict.items():
        if not isinstance(v, list):
            grid_search_params[k] = [v]
        else:
            grid_search_params[k] = v
        if not on_common_params:
            grid_search_params[k] = [grid_search_params[k][0]]
    grid_search_item_lens = list((len(v) for v in grid_search_params.values()))

    model_specific_gs_params = getattr(executor_cls, f"grid_search_{model_specific_gs_params_name}")()
    for k, v in model_specific_gs_params.items():
        grid_search_item_lens.append(len(v))
    total_search_items = np.prod(grid_search_item_lens)

    logging.info(
        f"{train_model} grid search on common params: {on_common_params}, model specific grid search params name: {model_specific_gs_params}.")
    logging.info(f"total search items: {total_search_items}.")

    search_param_keys = sorted(grid_search_params.keys())
    search_items = [grid_search_params[k] for k in search_param_keys]

    all_search_confs = []
    all_start_time = time.time()
    for search_item_combination in list(product(*search_items)):

        # main search params. all models share
        curr_conf = dict()
        for i in range(len(search_item_combination)):
            search_param_key = search_param_keys[i]
            search_item = search_item_combination[i]
            curr_conf[search_param_key] = search_item

        # model param grid search params. specialized to each model
        keys = sorted(model_specific_gs_params.keys())
        items = [model_specific_gs_params[k] for k in model_specific_gs_params]
        for model_param_search_item_combination in list(product(*items)):
            curr_conf[model_specific_gs_params_name] = dict()
            for i in range(len(model_param_search_item_combination)):
                search_key = keys[i]
                search_item = model_param_search_item_combination[i]
                curr_conf[model_specific_gs_params_name][search_key] = search_item

            # conf established, launch training
            conf = Config(curr_conf)
            all_search_confs.append(conf)
            curr_conf_idx = len(all_search_confs)
            now = time.time()
            logging.info(
                f"{train_model} grid search {curr_conf_idx}/{total_search_items} starts. curr time usage: {now - all_start_time:.2f}s")
            logging.info(
                f'{train_model} grid search conf {curr_conf_idx}/{total_search_items} = {json.dumps(conf.to_dict(), indent="    ")}'
            )
            executor = executor_cls(conf=Config(curr_conf))
            executor.train()
            train_over_time = time.time()
            logging.info(
                f"{train_model} grid search {curr_conf_idx}/{total_search_items} done. training duration: {train_over_time - now:.2f}s")


def launch_train_once(train_model: ModelType, _configs: Dict[ModelType, Dict], mode: str="single"):
    assert mode in ["single", "meta"]
    conf_dict = _configs[train_model]
    confirmed_params = dict()
    for k, v in conf_dict.items():
        if not isinstance(v, list):
            confirmed_params[k] = v
        else:
            confirmed_params[k] = v[0]
    now = time.time()
    logging.info(f"{train_model} {mode} train starts. conf = {json.dumps(confirmed_params, indent='    ')}")
    conf = Config.from_dict(confirmed_params)
    executor_cls = get_executor_cls(model_type=train_model)
    executor = executor_cls(conf)
    if mode == "single":
        executor.single_train()
    elif mode == "meta":
        executor.meta_train()
    train_over_time = time.time()
    logging.info(f"{train_model} {mode} train ends. training duration = {train_over_time - now:.2f}s.")


def launch(models: List[ModelType], launch_lambda):
    for i, train_model in enumerate(models):
        now = time.time()
        logging.info(f"launching {train_model} starts. rest models: {[m.name for m in train_models[i + 1:]]}")
        launch_lambda(train_model)
        launch_over = time.time()
        logging.info(f"launching {train_model} ends. launching duration = {launch_over - now:.2f}s.")


train_models = [
    ModelType.MLP,
    # ModelType.GBDT,
    # ModelType.GCNSubgraph,
    # ModelType.PerfNet,
    # ModelType.Transformer,
    # ModelType.MLPTestSubgraph,
    # ModelType.MLPTestGrouping,
    # ModelType.LSTM,
    # ModelType.GRU,
    # ModelType.GCNGrouping
]

transfer_models = [
    ModelType.Transformer,
]

tasks = {
    "single_train",
    # "meta_train",√√√
    # "grid_search",
    # "single_transfer",
    # "grid_search_transfer"
}

if __name__ == '__main__':
    if "single_train" in tasks:
        # launch single train
        launch(models=train_models,
                     launch_lambda=lambda train_model: launch_train_once(train_model, train_configs, "single"))

    if "meta_train" in tasks:
        # launch single train
        launch(models=train_models,
                     launch_lambda=lambda train_model: launch_train_once(train_model, train_configs, "meta"))

    if "grid_search" in tasks:
        # launch grid search on model params
        launch(models=train_models,
                     launch_lambda=lambda train_model: launch_grid_search(train_model, train_configs,
                                                                          on_common_params=True,
                                                                          model_specific_gs_params_name="model_params"))

    if "single_transfer" in tasks:
        # launch single transfer train
        launch(models=transfer_models,
                     launch_lambda=lambda train_model: launch_train_once(train_model, transfer_configs, "single"))

    if "grid_search_transfer" in tasks:
        # launch grid search on transfer params
        launch(models=transfer_models,
                     launch_lambda=lambda train_model: launch_grid_search(train_model, transfer_configs,
                                                                          on_common_params=False,
                                                                          model_specific_gs_params_name="transfer_params"))

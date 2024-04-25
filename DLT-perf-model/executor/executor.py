import json
import logging
import os
import pathlib
import random
import time
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import learn2learn as l2l
import torch.optim
from learn2learn.data import InfiniteIterator
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from config import Config
from data.dataset import MDataset, load_graphs, Graph, load_dataset_pkl, save_dataset_pkl, dateset_exists, \
    save_scalers_pkl, \
    load_scalers_pkl, load_graphs_pkl, save_graphs_pkl
from objects import ModelType, ckpts_dir, Environment
from .base_module import MModule
from .util import nested_detach


class Executor(ABC):
    def __init__(self, conf: Config):
        self.model_type: ModelType = self._init_model_type()
        self.model_ckpts_dir = ckpts_dir / self.model_type.name
        self.conf: Config = conf
        self._check_params()
        self.train_mode: str | None = None  # "single" or "meta"
        self.executor_name = 'BaseExecutor'

    def _prepare_single_dataset(self):
        # todo 使用pickle保存对象

        self.set_seed()
        self.train_records: Dict = dict()

        # load dataset
        if dateset_exists(self.conf.dataset_environment, self.executor_name, 'train'):
            self.eval_graphs = load_graphs(self.conf.dataset_environment,
                                           train_or_eval="eval",
                                           use_dummy=self.conf.dataset_dummy)
            self.preprocessed_train_ds = load_dataset_pkl(self.conf.dataset_environment, self.executor_name, 'train',
                                                          self.conf.dataset_normalization)
            self.preprocessed_eval_ds = load_dataset_pkl(self.conf.dataset_environment, self.executor_name, 'eval',
                                                         self.conf.dataset_normalization)
            self.scalers = load_scalers_pkl(self.conf.dataset_environment, self.executor_name, 'train',
                                            self.conf.dataset_normalization)
            # self.eval_graphs = load_graphs_pkl(self.conf.dataset_environment, self.executor_name, 'eval')
            print('load dataset from file done')
            return
        # 后续useless,可以设置为None
        self.eval_graphs = load_graphs(self.conf.dataset_environment,
                                       train_or_eval="eval",
                                       use_dummy=self.conf.dataset_dummy)
        self.train_graphs = load_graphs(self.conf.dataset_environment,
                                        train_or_eval="train",
                                        use_dummy=self.conf.dataset_dummy)
        self.preprocessed_train_ds: MDataset | None = None
        self.preprocessed_eval_ds: MDataset | None = None

        train_ds = self._init_dataset(self.train_graphs)
        eval_ds = self._init_dataset(self.eval_graphs)
        # todo 训练姐和测试机应该分别归一化
        self.scalers = self._get_scalers(train_ds)
        self.preprocessed_train_ds = self._init_preprocessed_dataset(train_ds)
        self.preprocessed_eval_ds = self._init_preprocessed_dataset(eval_ds)

        # save
        save_dataset_pkl(self.preprocessed_train_ds, self.conf.dataset_environment, self.executor_name, 'train',
                         self.conf.dataset_normalization)
        save_dataset_pkl(self.preprocessed_eval_ds, self.conf.dataset_environment, self.executor_name, 'eval',
                         self.conf.dataset_normalization)
        save_scalers_pkl(self.scalers, self.conf.dataset_environment, self.executor_name, 'train',
                         self.conf.dataset_normalization)
        # save_graphs_pkl(self.eval_graphs, self.conf.dataset_environment, self.executor_name, 'eval')

    def _prepare_meta_dataset(self):
        self.meta_train_graphs: Dict[Environment, List[Graph]] = dict()
        self.meta_eval_graphs: Dict[Environment, List[Graph]] = dict()
        self.meta_train_dss: Dict[Environment, MDataset] = dict()
        self.meta_preprocessed_train_dss: Dict[Environment, MDataset] = dict()
        self.meta_preprocessed_train_data_loaders: Dict[Environment, InfiniteIterator] = dict()
        self.meta_eval_dss: Dict[Environment, MDataset] = dict()
        self.meta_preprocessed_eval_dss: Dict[Environment, MDataset] = dict()

        for env in self.conf.meta_dataset_train_environments:
            graphs = load_graphs(env, train_or_eval="train", use_dummy=self.conf.dataset_dummy)
            self.meta_train_graphs[env] = graphs
            self.meta_train_dss[env] = self._init_dataset(graphs)
            self.meta_preprocessed_train_dss[env] = self._init_preprocessed_dataset(self.meta_train_dss[env])

        for env in self.conf.meta_dataset_eval_environments:
            graphs = load_graphs(env, train_or_eval="train", use_dummy=self.conf.dataset_dummy)
            self.meta_eval_graphs[env] = graphs
            self.meta_eval_dss[env] = self._init_dataset(graphs)
            self.meta_preprocessed_eval_dss[env] = self._init_preprocessed_dataset(self.meta_eval_dss[env])

        self.train_records: Dict = dict()
        self.set_seed()

    @staticmethod
    @abstractmethod
    def default_model_params() -> Dict[str, Any]:
        pass

    @abstractmethod
    def _get_scalers(self, ds: MDataset) -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def grid_search_model_params() -> Dict[str, List]:
        pass

    @staticmethod
    def grid_search_transfer_params() -> Dict[str, List] | None:
        return None

    def set_seed(self):
        seed = self.conf.all_seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            # When running on the CuDNN backend, two further options must be set
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        logging.info(f"Random seed set as {seed}")

    def _check_params(self):
        pass

    @abstractmethod
    def _init_model_type(self) -> ModelType:
        pass

    def _generate_save_path(self, prefix: str = "") -> str:
        time_format = "%Y-%m-%d_%H-%M-%S"
        time_str = time.strftime(time_format)
        save_path_name = f"{prefix}{time_str}"
        save_path = f"{str(self.model_ckpts_dir / save_path_name)}"
        return save_path

    def _ensure_save_dir(self, save_path):
        p = pathlib.Path(save_path)
        if p.exists():
            assert p.is_dir()
            return
        try:
            os.makedirs(save_path)
        except IOError:
            logging.fatal("Cannot create save path: %s" % save_path)
            exit(-1)

    @abstractmethod
    def _init_dataset(self, graphs: List[Graph]) -> MDataset:
        pass

    def init_model(self) -> MModule | Any:
        if self.conf.resume_from_ckpt is not None:
            ckpt_filepath = pathlib.Path(self.model_ckpts_dir, self.conf.resume_from_ckpt)
            model = self._load_ckpt(ckpt_filepath)
            return model
        return self._init_model()

    @abstractmethod
    def _init_model(self) -> MModule | Any:
        pass

    @staticmethod
    def _load_ckpt(ckpt_filepath) -> MModule | Any:
        model = torch.load(ckpt_filepath)
        return model

    @abstractmethod
    def to_device(self, features, labels):
        pass

    def single_train(self):
        self.train_mode = "single"
        self._prepare_single_dataset()
        save_path = self._generate_save_path(prefix="single_train")
        processed_train_ds = self.preprocessed_train_ds
        train_dl = DataLoader(processed_train_ds, batch_size=self.conf.batch_size, shuffle=True)
        model = self.init_model()
        model.to(self.conf.device)
        if self.conf.transfer_params is not None:
            model.prepare_transfer(**self.conf.transfer_params)
        model.train()
        curr_train_step = -1
        optimizer_cls = self.conf.optimizer_cls
        lr = self.conf.learning_rate
        optimizer = optimizer_cls(model.parameters(), lr=lr)
        start = time.time_ns()
        logging.info(f"{self.model_type} start single training.")
        for epoch in range(self.conf.num_train_epochs):
            logging.info(f"{self.model_type} training epoch %d" % epoch)
            for i, data in enumerate(tqdm(train_dl)):
                optimizer.zero_grad()
                features, labels = data
                features, labels = self.to_device(features, labels)
                outputs = model(features)
                loss = model.compute_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                curr_train_step += 1
                loss_value = float(nested_detach(loss))
                self.train_records.setdefault("loss", list())
                self.train_records["loss"].append(loss_value)
                if curr_train_step % self.conf.eval_steps == 0:
                    now = time.time_ns()
                    train_dur = (now - start) / 1e9
                    logging.info(f"{self.model_type} trained for {train_dur} seconds.")
                    logging.info(f"{self.model_type} eval at step {curr_train_step}.")
                    model.eval()
                    metrics = self._evaluate(model, self.conf.dataset_environment, self.preprocessed_eval_ds)
                    logging.info(f"{self.model_type} train loss: {loss_value}, eval metrics: {metrics}")
                    metrics["train_loss"] = loss_value

                    self.train_records.setdefault("eval_metrics", list())
                    self.train_records["eval_metrics"].append({
                        "metrics": metrics,
                        "step": curr_train_step,
                        "duration": train_dur
                    })
                    self.save_model(save_path=save_path, model=model, curr_steps=curr_train_step,
                                    curr_loss_value=loss_value)
                    model.train()
        self.save_train_plot(save_path)

    def meta_fast_adapt(self, compute_loss, adaptation_task, learner, meta_fast_adaption_step):
        adaptation_data, adaptation_labels = adaptation_task

        # Fast Adaptation
        for step in range(meta_fast_adaption_step):
            with torch.backends.cudnn.flags(enabled=False):
                train_error = compute_loss(learner(adaptation_data), adaptation_labels)
                learner.adapt(train_error)

    def meta_sample_task(self, env: Environment, batch_size: int) -> Tuple[
        Environment, Tuple[torch.Tensor, torch.Tensor]]:
        ds = self.meta_preprocessed_train_dss[env]
        if env not in self.meta_preprocessed_train_data_loaders:
            sampler = RandomSampler(ds, replacement=False)
            self.meta_preprocessed_train_data_loaders[env] = InfiniteIterator(
                DataLoader(self.meta_preprocessed_train_dss[env], sampler=sampler, batch_size=batch_size))
        dl = self.meta_preprocessed_train_data_loaders[env]
        task = next(dl)
        return task

    def meta_train(self):
        self.train_mode = "meta"
        self._prepare_meta_dataset()
        save_path = self._generate_save_path(prefix="meta_train")
        model = self.init_model()
        model.to(self.conf.device)
        lr = self.conf.meta_base_learning_rate
        start = time.time_ns()
        logging.info(f"{self.model_type} start meta training.")
        meta_lr = self.conf.meta_learner_learning_rate
        meta_train_steps = self.conf.meta_train_steps
        meta_tps = self.conf.meta_task_per_step
        meta_shots = self.conf.meta_shots
        meta_fast_adaption_step = self.conf.meta_fast_adaption_step

        meta_model = l2l.algorithms.MAML(model, lr=meta_lr)
        opt = self.conf.optimizer_cls(model.parameters(), lr=lr)

        for curr_train_step in range(meta_train_steps):
            iteration_error = 0.0
            for _ in range(meta_tps):
                learner = meta_model.clone()
                envs = self.meta_preprocessed_train_dss.keys()
                env = random.choice(list(envs))
                meta_adaption_batch_size = self.conf.meta_adaption_batch_size
                eval_data_size = meta_adaption_batch_size - meta_shots
                adaptation_task = self.meta_sample_task(env=env, batch_size=meta_shots)
                self.meta_fast_adapt(model.compute_loss, adaptation_task, learner, meta_fast_adaption_step)
                # Fast Adaptation

                evaluation_task = self.meta_sample_task(env=env, batch_size=eval_data_size)
                evaluation_data, evaluation_labels = evaluation_task
                # Compute validation loss
                with torch.backends.cudnn.flags(enabled=False):
                    predictions = learner(evaluation_data)
                    valid_error = model.compute_loss(predictions, evaluation_labels)
                    # valid_error.backward()
                iteration_error += valid_error

            iteration_error /= meta_tps
            loss_value = iteration_error.item()
            print('Loss : {:.3f}'.format(loss_value))
            self.train_records.setdefault("loss", list())
            self.train_records["loss"].append(loss_value)

            opt.zero_grad()
            # Take the meta-learning step
            iteration_error.backward()
            opt.step()

            if curr_train_step % self.conf.eval_steps == 0:
                now = time.time_ns()
                train_dur = (now - start) / 1e9
                logging.info(f"{self.model_type} trained for {train_dur} seconds.")
                logging.info(f"{self.model_type} eval at step {curr_train_step}.")
                metrics = dict()
                for env, ds in self.meta_preprocessed_eval_dss.items():
                    learner = model.clone()
                    adaptation_task = self.meta_sample_task(env=env, batch_size=meta_shots)
                    self.meta_fast_adapt(model.compute_loss, adaptation_task, learner, meta_fast_adaption_step)
                    metrics[str(env)] = self._evaluate(learner, env, ds)
                logging.info(f"{self.model_type} train loss: {loss_value}, eval metrics: {metrics}")
                metrics["train_loss"] = loss_value

                self.train_records.setdefault("eval_metrics", list())
                self.train_records["eval_metrics"].append({
                    "metrics": metrics,
                    "step": curr_train_step,
                    "duration": train_dur
                })
                self.save_model(save_path=save_path, model=meta_model, curr_steps=curr_train_step,
                                curr_loss_value=loss_value)
                model.train()

    def save_train_plot(self, save_path):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        eval_metrics = self.train_records["eval_metrics"]

        def get_list(metric_key):
            l = list()
            for eval_metric in eval_metrics:
                l.append(eval_metric["metrics"][metric_key])
            return l

        x_step = self.conf.eval_steps
        X = [x_step * i for i in range(len(eval_metrics))]
        # train loss, eval loss
        line_plots = (
            ["train_loss", "eval_loss"],
            ["MRE"],
            ["RMSE"]
        )
        for i, line_plot_keys in enumerate(line_plots):
            ax = axes[i]
            for key in line_plot_keys:
                ax.plot(X, get_list(key), label=key)
            ax.set_xlabel("steps")
            ax.legend()
        fig_save_path = str(pathlib.Path(save_path, "train_plot.png"))
        fig.savefig(fig_save_path)

    def _init_preprocessed_dataset(self, raw: MDataset) -> MDataset:
        preprocessed = self._preprocess_dataset(raw)
        return preprocessed

    @abstractmethod
    def _preprocess_dataset(self, ds: MDataset) -> MDataset:
        pass

    def evaluate(self):
        model = self.init_model()
        self.train_mode = "single"
        self._prepare_single_dataset()
        metrics = self._evaluate(model, self.conf.dataset_environment, self.preprocessed_eval_ds)
        logging.info(f"{self.model_type} evaluated metrics: {metrics}")
        return metrics

    def meta_evaluate(self):
        model = self.init_model()
        self.train_mode = "meta"
        self._prepare_meta_dataset()
        meta_shots = self.conf.meta_shots
        meta_fast_adaption_step = self.conf.meta_fast_adaption_step
        metrics = dict()
        for env, ds in self.meta_preprocessed_eval_dss.items():
            learner = model.clone()
            adaptation_task = self.meta_sample_task(env=env, batch_size=meta_shots)
            self.meta_fast_adapt(model.compute_loss, adaptation_task, learner, meta_fast_adaption_step)
            metrics[str(env)] = self._evaluate(model, env, ds)
        logging.info(f"{self.model_type} evaluated metrics: {metrics}")
        return metrics

    @abstractmethod
    def _evaluate(self, model, env: Environment, ds: MDataset) -> Dict[str, float]:
        pass

    def _dl_evaluate_pred(self, model: MModule, env: Environment, ds: MDataset):
        processed_eval_ds = ds
        dl = DataLoader(processed_eval_ds, batch_size=self.conf.batch_size, shuffle=False)
        input_batches = list()
        output_batches = list()
        eval_losses = list()
        for data in dl:
            features, labels = data
            features['x_op_feature'] = features["x_op_feature"].to(device=self.conf.device)
            labels['y_node_durations'] = labels['y_node_durations'].to(device=self.conf.device)
            with torch.no_grad():
                outputs = model(features)
            loss = model.compute_loss(outputs, labels)
            eval_loss = float(nested_detach(loss))
            eval_losses.append(eval_loss)
            input_batches.append(features)
            output_batches.append(outputs)
        eval_loss = np.mean(eval_losses)

        return input_batches, output_batches, eval_loss

    def save_model(self, save_path, model, curr_steps: int, curr_loss_value: float):
        d = {
            "train_config": self.conf.raw_config,
            "train_records": self.train_records
        }
        self._ensure_save_dir(save_path=save_path)
        with open(pathlib.Path(save_path, "train_records.json"), "w") as f:
            json.dump(d, f, indent="\t")
        self._save_ckpt_to(model, pathlib.Path(save_path, f"ckpt_{curr_steps}.pth"))

    @staticmethod
    def _save_ckpt_to(model, filepath):
        torch.save(model, filepath)

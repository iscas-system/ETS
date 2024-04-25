import json
import os
import pathlib
from typing import Dict, List

from logger import init_logging
from objects import ModelType,GPUType, ckpts_dir

init_logging()


class TrainRecord:
    class EvalMetric:
        def __init__(self, d):
            self.train_loss: d.get("train_loss", 0)
            self.metrics: Dict = d.get("metrics", dict())
            self.step: int = d.get("step", 0)
            self.duration: float = d.get("duration", 0)

    def __init__(self, ckpt_dir: pathlib.Path, d: Dict):
        self.ckpt_dir: pathlib.Path = ckpt_dir
        self.raw = d
        self.train_config: Dict = d["train_config"]
        train_records_dict = d["train_records"]
        eval_metrics_raw = train_records_dict["eval_metrics"]
        self.eval_metrics = [
            TrainRecord.EvalMetric(eval_metric) for eval_metric in eval_metrics_raw
        ]

    def optimal_eval_metric(self, metric_name="MSE", standard="min"):
        if standard == "min":
            reduce_func = min
        elif standard == "max":
            reduce_func = max
        else:
            raise ValueError(f"Unknown standard: {standard}")
        return reduce_func(self.eval_metrics, key=lambda x: x.metrics[metric_name])


def load_train_records(gpu_type: GPUType, train_models: List[ModelType], prefix=''):
    train_records = dict()
    for train_model in train_models:
        train_records[train_model] = list()
        if prefix!='':
            train_model_dir = ckpts_dir / prefix / gpu_type.name /train_model.name
        else:
            train_model_dir = ckpts_dir /gpu_type.name/ train_model.name
        if not train_model_dir.exists():
            continue
        for dirname in os.listdir(str(train_model_dir)):
            ckpt_dir = train_model_dir / dirname
            if not ckpt_dir.is_dir():
                continue
            train_record_path = ckpt_dir / "train_records.json"
            with open(str(train_record_path), "r") as d:
                d = json.load(d)
                train_records[train_model].append(TrainRecord(ckpt_dir=ckpt_dir, d=d))
    return train_records


if __name__ == '__main__':
    load_train_records([ModelType.MLP, ModelType.LSTM])

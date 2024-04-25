import os
import pathlib

import pandas as pd

from benchmark.models.load import ModelDescriptions
import random

from predictor.predict import model_predict

# 用于调度的任务集合

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
datasets_dir = BASE_DIR / 'data' / 'datasets'
gpus = [ 'P4CPUALL','T4CPUALL', '2080TiCPUALL', '3080TiCPUALL']
models = [
    'resnet18',
    'resnet50',
    'resnet101',
    'wide_resnet50_2',
    'alexnet',
    'densenet121',
    'densenet169',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_v2_s'
    'googlenet'
    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small',
    'mnasnet0_5',
    'mnasnet0_75',
    'mnasnet1_0',
    'mnasnet1_3',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0',
    'squeezenet1_0',
    'squeezenet1_1',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19_bn',
    'vgg19',
]

def get_file(gpu: str, model: str, batch: int, h: int, dtype: int) -> str:
    files = os.listdir(os.path.join(datasets_dir, gpu))
    for file in files:
        if file.startswith(model):
            if os.stat(os.path.join(datasets_dir, gpu, file)).st_size == 0:
                continue
            df = pd.read_csv(os.path.join(datasets_dir, gpu, file))
            if df.iloc[0]['batch'] == batch and df.iloc[0]['h'] == h and df.iloc[0]['input_type'] == dtype:
                return os.path.join(datasets_dir, gpu, file)
    return None


def check_exist(config: dict) -> bool:
    for gpu in gpus:
        measure_file = get_file(gpu, config['model'], config['batch'], config['h'], config['dtype'])
        if measure_file is None:
            return False
        config['files'][gpu] = {
            'file': measure_file,
        }

    return True


def generate_tasks(n=5):
    batches = [8, 16, 32, 64 , 128, 256]
    hws = [[32, 32], [64, 64], [128, 128], [160, 160], [256, 256], [384, 384], [512, 512],
           [768, 768]]
    dtype = [1]

    tasks = []
    while len(tasks) < n:
        tmp = {
            'model': random.choice(models),
            'batch': random.choice(batches),
            'h': random.choice(hws)[0],
            'dtype': random.choice(dtype),
            'files': {},
        }
        if check_exist(tmp):
            print(tmp)
            tasks.append(tmp)
    return tasks


if __name__ == '__main__':
    tasks = generate_tasks(n=300)
    for task in tasks:
        for gpu, info in task['files'].items():
            eval_graphs = model_predict(gpu, info['file'])
            task[gpu] = {
                'predict': eval_graphs[0].graph_duration,
                'measure': eval_graphs[0].graph_duration_pred,
                'file': info['file']
            }
        print(task)
    with open('../scheduler/tasks.json', 'w') as f:
        import json
        json.dump(tasks, f)

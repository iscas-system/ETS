from datetime import datetime
import os.path
import pathlib
import random

import pandas as pd

from predictor.predict import model_predict
from scripts.generate_tasks import get_file
import uuid

# 从数据集中选5个比较准的模型，作为log


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
logs_dir = BASE_DIR / 'data' / 'predictor_logs'
gpu = 'T4CPUALL'

models = [
    'resnet18',
    'resnet50',
    'resnet101',
    'alexnet',
    'densenet121',
    'efficientnet_b1',
    'googlenet'
    'mobilenet_v2',
    'mnasnet0_75',
    'shufflenet_v2_x1_0',
    'squeezenet1_0',
    'vgg11',
]


def prepare_logs(n=5):
    batches = [8, 16, 32, 64, 128, 256]
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
        }
        measure_file = get_file(gpu, tmp['model'], tmp['batch'], tmp['h'], tmp['dtype'])
        if measure_file is None:
            continue
        eval_graphs = model_predict(gpu, measure_file)
        if abs(eval_graphs[0].graph_duration - eval_graphs[0].graph_duration_pred) / eval_graphs[
            0].graph_duration < 0.05:
            tasks.append(measure_file)
    return tasks


if __name__ == '__main__':
    tasks = prepare_logs()
    for i in range(len(tasks)):
        df = pd.read_csv(tasks[i])
        base_name = os.path.basename(tasks[i])
        model = base_name.split('.')[0]
        batch, h, input_type = df.iloc[0]['batch'], df.iloc[0]['h'], df.iloc[0]['input_type']
        now = datetime.now()
        log_id = str(uuid.uuid4())
        meta_info = {
            'model': model,
            'batch_size': batch,
            'input_size': h,
            'dtype': input_type,
            'gpu': gpu,
            'log_id': log_id,
            'time': now.strftime("%Y-%m-%d %H:%M:%S"),
            'error': ''
        }
        log_dir = os.path.join(logs_dir, log_id)
        os.system('mkdir ' + log_dir)
        with open(os.path.join(log_dir, 'meta_info.txt'), 'w') as f:
            f.write(str(meta_info))
        name = model + '-' + str(batch) + '-' + str(h) + '-' + str(input_type) + '-' + gpu + '-' + now.strftime(
            "%Y%m%d%H%M%S")
        os.system('cp ' + tasks[i] + ' ' + os.path.join(log_dir, name + '.measure.csv'))

import pyprof
from utils.log import init_log
from benchmark.run import run_model_config
import torch
import argparse


def _run_models_benchmark(model_name):
    batches = [2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 512]
    hws = [[16, 16], [32, 32], [64, 64], [128, 128], [160, 160], [256, 256], [384, 384], [512, 512],
           [768, 768]]
    dtypes = [torch.FloatTensor]
    for batch in batches:
        for hw in hws:
            for dtype in dtypes:
                run_model_config(batch, hw[0], dtype, model_name)

if __name__ == '__main__':
    pyprof.init()
    init_log()
    # nohup ./../../../benchmark/run_models.sh 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=bool, default=True)  # 固定模型
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    if args.models:
        _run_models_benchmark(args.model_name)

import argparse
import os
import pathlib
from time import sleep

import pyprof

import torch
from benchmark.run import run_rand_config
from util import InputType
from utils.log import init_log



if __name__ == '__main__':
    pyprof.init()
    init_log()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=500)
    parser.add_argument('--configs', type=str, default='/root/pytorch_benchmark/benchmark/rand_configs.dict')
    args = parser.parse_args()

    with open(args.configs, 'r') as f:
        configs = f.readlines()
        for config_str in configs[args.start:args.end]:
            config = eval(config_str)
            if config['dtype'] != 1:
                continue
            config['dtype'] = InputType.id2types[config['dtype']]
            run_rand_config(**config)
            sleep(10)
            torch.cuda.empty_cache()
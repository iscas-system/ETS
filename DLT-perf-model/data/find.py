import argparse
import os.path
import pathlib
import random

import pandas as pd

# dataset_path = str(pathlib.Path(os.getcwd()) / ".." / "datasets")
dataset_path = '/root/guohao/repos/DLT-perf-model/datasets'
config_path = str(pathlib.Path(os.getcwd())  / "configs" / "models")


def find_data(data_set: str,model='resnet50', batch=32, h=64,input_type=1):
    """
    找出对应数据集中所需要的数据，用于测试
    """
    train_data = os.path.join(dataset_path, data_set, 'models','output')
    # train_data = os.path.join(dataset_path, data_set, 'train')
    for file in os.listdir(train_data):
        if not file.endswith('.csv'):
            continue
        if not file.startswith(model):
            continue
        df = pd.read_csv(os.path.join(train_data, file))
        if (df['batch']==batch).any() and (df['h']==h).any() and (df['input_type']==input_type).any():
            print(file)
            return 
    return


if __name__ == '__main__':
    find_data('RTX2080Ti_CPUALL')  
    # find_data('TESLAT4_CPUALL')
    # find_data('P4_CPUALL')

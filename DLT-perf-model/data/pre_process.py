import argparse
import os.path
import pathlib
import random

import pandas as pd

# dataset_path = str(pathlib.Path(os.getcwd()) / ".." / "datasets")
dataset_path = '/root/guohao/repos/DLT-perf-model/datasets'
config_path = str(pathlib.Path(os.getcwd())  / "configs" / "models")


def split_data(data_set: str):
    """
    将models划分为训练集和测试集
    """
    data_set_path = os.path.join(dataset_path, data_set)
    model_path = os.path.join(data_set_path, 'models')
    train_path = os.path.join(data_set_path, 'train')
    eval_path = os.path.join(data_set_path, 'eval')
    duplicate_path = os.path.join(data_set_path, 'duplicate')
    
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    if not os.path.exists(duplicate_path):
        os.mkdir(duplicate_path)

    # 删除多余数据
    for i in range(7000, 20000, 500):
        os.system('mv ' + os.path.join(model_path, f'rand_{i}') + '.*' + ' ' + duplicate_path)
    
    model_names = []
    with open(os.path.join(config_path, 'train.txt'), 'r') as f:
        train_models = f.readlines()
        for model in train_models:
            model = model.rstrip('\n')
            model_names.append(model)
            
    files = []
    for file in os.listdir(model_path):
        if file.endswith('.csv'):
            files.append(file)
    random.shuffle(files)
    
    train_num = (len(files)//5)*4
    
    for file in files[:train_num]:
        model_name = file.split('.')[0]
        if model_name in model_names or 'rand' in model_name:
            os.system('mv ' + os.path.join(model_path, file) + ' ' + train_path)
    for file in files[train_num:]:
        model_name = file.split('.')[0]
        if model_name in model_names or 'rand' in model_name:
            os.system('mv ' + os.path.join(model_path, file) + ' ' + eval_path)
    return


def delete_unvalid_data(data_set: str):
    """
    删除无效数据:space小于0和input_type不为1的数据
    """
    train_data = os.path.join(dataset_path, data_set, 'models')
    for file in os.listdir(train_data):
        if not file.endswith('.csv'):
            continue
        file_size = os.path.getsize(os.path.join(train_data, file))
        if file_size == 0:
            print('remove invalid file: ' +  os.path.join(train_data, file))
            os.remove(os.path.join(train_data, file))
            continue
        df = pd.read_csv(os.path.join(train_data, file))
        if (df['space'] < -1000).any() :
            print('remove invalid file: ' +  os.path.join(train_data, file))
            os.remove(os.path.join(train_data, file))
            continue
        # 删除过大的数据
        if (df['batch'] > 256).any() or (df['h'] > 512).any() or  (df['input_type'] != 1).any():
            print('remove invalid file: ' +  os.path.join(train_data, file))
            os.remove(os.path.join(train_data, file))
            continue
        if (df['batch'] < 8).any() or (df['h'] < 32).any():
            print('remove invalid file: ' +  os.path.join(train_data, file))
            os.remove(os.path.join(train_data, file))
            continue
        if len(df) > 3000:
            print('remove invalid file: ' +  os.path.join(train_data, file))
            os.remove(os.path.join(train_data, file))
            continue
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_set', type=str, default='RTX2080Ti_CPU100')
    parser.add_argument('--data_set', type=str, default='T4_CPU100')

    args = parser.parse_args()

    # delete_unvalid_data('RTX2080Ti_CPUALL')  
    # split_data('RTX2080Ti_CPUALL')
    # delete_unvalid_data('T4_CPUALL')
    # split_data('T4_CPUALL')
    delete_unvalid_data('P4_CPU100')
    split_data('P4_CPU100')

# Build paths inside the project like this: BASE_DIR / 'subdir'.
import logging
import uuid
import os
import subprocess
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from analyzer.do_analyze import DoAnalyze
from analyzer.do_parse import DoParse
from analyzer.do_prof import DoProf
from predictor.data import OperatorType
from predictor.predict import model_predict
from service import util

BASE_DIR = Path(__file__).resolve().parent.parent
logs_dir = os.path.join(BASE_DIR, 'data', 'predictor_logs')
script_path = os.path.join(BASE_DIR, 'benchmark', 'run.py')

nsys_path = '/usr/local/cuda/bin/nsys'
python_path = '/root/miniconda3/envs/gh-torch/bin/python'
predictor_path = os.path.join(BASE_DIR, 'data/predictors')

gpu = os.environ.get('GPU_NAME', 'T4CPUALL')


class LogStatus(Enum):
    Pending = 1
    Perfed = 2
    Predicted = 3
    Error = 4


def do_process(input_file: str):
    file_name = os.path.basename(input_file)
    log_path = os.path.dirname(input_file)
    try:
        if not input_file.endswith('.sqlite'):
            raise Exception('no sqlite file')
        config, _ = file_name.split('.')
        output_file = os.path.join(log_path, f'{config}.measure.csv')
        if os.path.exists(output_file):
            logging.info('file %s already processed' % input_file)
            return 'success'

        logging.info('process %s' % input_file)
        kernels_dict = DoParse(input_file)
        logging.info('parse %s done' % input_file)

        if kernels_dict is None or len(kernels_dict) == 0:
            raise Exception('kernels_dict is None or len(kernels_dict) == 0')

        for i in range(len(kernels_dict)):
            if kernels_dict[i]['dir'] != 'fprop':
                print(f'num ops {i}')
                break
        kernels_df = DoProf(kernels_dict)
        logging.info('prof %s done' % input_file)

        results = DoAnalyze(kernels_df)
        logging.info('process %s done' % input_file)
        result_df = results[-1].format()
        result_df = result_df[result_df["seq"] != "-1"]
        result_df = result_df[result_df["seq"] != -1]
        result_df.to_csv(output_file, index=False)
    except Exception as e:
        logging.error(f'process error: {e}')
        # meta_info file
        with open(os.path.join(log_path, 'meta_info.txt'), 'r') as f:
            meta_info = eval(f.read())
        meta_info['error'] = f'process error: {e}'
        with open(os.path.join(log_path, 'meta_info.txt'), 'w') as f:
            f.write(str(meta_info))
        return f'process error: {e}'
    return 'success'


def do_perf(model, batch_size, input_size, dtype):
    log_id = str(uuid.uuid4())
    # create log dir
    log_dir = os.path.join(logs_dir, log_id)
    os.mkdir(log_dir)
    os.chdir(log_dir)

    now = datetime.now()
    name = model + '-' + str(batch_size) + '-' + str(input_size) + '-' + dtype + '-' + gpu + '-' + now.strftime(
        "%Y%m%d%H%M%S")

    # meta_info file
    meta_info = {
        'model': model,
        'batch_size': batch_size,
        'input_size': input_size,
        'dtype': dtype,
        'gpu': gpu,
        'log_id': log_id,
        'time': now.strftime("%Y-%m-%d %H:%M:%S"),
        'error': ''
    }
    with open(os.path.join(log_dir, 'meta_info.txt'), 'w') as f:
        f.write(str(meta_info))
    try:
        ret = subprocess.check_call([
            nsys_path, 'profile', '-f', 'true', '--output', name, '-c', 'cudaProfilerApi', '--capture-range-end',
            'repeat:20', '--export', 'sqlite', '--cuda-memory-usage=true', python_path,
            script_path, '--model', model, '--batch_size', str(batch_size), '--input_size', str(input_size), '--dtype',
            dtype
        ])

        sqlite_file = os.path.join(log_dir, name + '.sqlite')
        if os.path.exists(sqlite_file):
            do_process(sqlite_file)
            return log_id
        else:
            raise Exception('perf failed, no sqlite file')
    except Exception as e:
        logging.error('perf failed, ' + str(e))
        meta_info['error'] = str(e)
        with open(os.path.join(log_dir, 'meta_info.txt'), 'w') as f:
            f.write(str(meta_info))
        return log_id


def list_logs():
    dirs = os.listdir(logs_dir)
    res = []
    for _, log_id in enumerate(dirs):
        log_dir = os.path.join(logs_dir, log_id)
        status = LogStatus.Pending
        if util.find_file_with_suffix(log_dir, '.measure.csv'):
            status = LogStatus.Perfed
        if util.find_file_with_suffix(log_dir, '.predict.csv'):
            status = LogStatus.Predicted
        meta_file = os.path.join(log_dir, 'meta_info.txt')
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta_info = eval(f.read())
            meta_info['status'] = status.name
            res.append(meta_info)
        else:
            continue
    return res


def perf_detail(log_id: str):
    log_path = os.path.join(logs_dir, log_id)
    if not os.path.exists(log_path):
        return 'no such uuid'
    predict_file = os.path.join(log_path, 'predict.csv')
    print(predict_file)
    if os.path.exists(predict_file):
        predict_df = pd.read_csv(os.path.join(log_path, predict_file))
        return predict_df.to_dict(orient='records')
    else:
        return 'no predict file, please do predict first'


def do_predict(log_id: str, gpu_name: str):
    try:
        log_path = os.path.join(logs_dir, log_id)
        if not os.path.exists(log_path):
            raise Exception('log path not exists')
        measure_file = util.find_file_with_suffix(log_path, '.measure.csv')
        if measure_file is None:
            raise Exception('no measure file, please do process first')

        eval_graphs = model_predict(gpu_name, os.path.join(log_path, measure_file))
        eval_graph = eval_graphs[0]
        res = []
        for node in eval_graph.nodes:
            node_id = node.op.operator_type_id
            tmp = {
                'op': OperatorType.get_op_name(node_id),
                'duration': node.duration,
                'duration_pred': node.duration_pred,
                'gap': node.gap,
                'gap_pred': node.gap_pred,
                'total': node.duration + node.gap,
                'total_pred': node.duration_pred + node.gap_pred,
            }
            res.append(tmp)
        res_df = pd.DataFrame(res)
        res_df.to_csv(os.path.join(log_path, 'predict.csv'), index=False)
    except Exception as e:
        logging.error('predict failed, ' + str(e))
        return 'failed, ' + str(e)
    return 'success'


if __name__ == '__main__':
    # densenet报错
    do_perf('resnet101',32, 128, 'float')
    # do_process(
    #     '/root/guohao/pytorch_benchmark/data/predict_logs/predictor_log9/resnet101-32-128-float-T4CPUALL-20240115065504.sqlite')
    # do_predict('2f0d7452-ac6a-41df-9a6e-e899a2421f5d', 'T4CPUALL')
    # res = list_logs()
    # print(res)

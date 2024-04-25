import argparse
import logging
import os
from multiprocessing import Process, Pool

from analyzer.do_analyze import DoAnalyze
from analyzer.do_parse import DoParse
from analyzer.do_prof import DoProf

process_num = 8
data_path = '/root/guohao/pytorch_benchmark/data/cnn/'
# data_path = 'E:\python_project\pytorch_benchmark\data\cnn'
output_path = os.path.join(data_path, 'output')
input_path = os.path.join(data_path, 'data')
log_path = os.path.join(data_path, 'log')

if __name__ == '__main__':
    input_file = '/root/guohao/pytorch_benchmark/convnext_base.18.sqlite'
    try:
        input_file = os.path.join(input_path, input_file)
        print('process %s' % input_file)
        file_name = os.path.basename(input_file)
        model_name, config_idx, _ = file_name.split('.')
        output_file = os.path.join(output_path, f'{model_name}.{config_idx}_7.csv')
        kernels_dict = DoParse(input_file)
        # check dtype
        for kernel in kernels_dict:
            layer = kernel['layer'][0]
            if 'Float' not in layer:
                print(layer)
            else:
                break
        kernels_df = DoProf(kernels_dict)
        results = DoAnalyze(kernels_df)
        results[-1].format().to_csv(output_file, index=False)
    except Exception as e:
        print(e)
        with open(os.path.join(log_path, 'error.log'), 'a') as f:
            f.write(input_file + '\n')
    print('process %s done' % input_file)

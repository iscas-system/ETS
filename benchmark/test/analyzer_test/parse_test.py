import json
import logging
import os

import pandas as pd

from analyzer.do_analyze import DoAnalyze, format2csv
from utils.log import init_log

if __name__ == '__main__':
    init_log()
    # path = '/root/guohao/pytorch_benchmark/tmp/data.1.sqlite.csv'
    # path = '/root/guohao/pytorch_benchmark/tmp/'
    # outpath = os.path.join(path, 'out')
    # for _, _, files in os.walk(path):
    #     for file in files:
    #         if file.endswith('sqlite.csv') and 'data' in file:
    #             data_i = file.split('.')[1]
    #             logging.info(f'parse {file}, data_i is {data_i}')
    #             # if data_i != '93':
    #             #     continue
    #             df = pd.read_csv(os.path.join(path, file))
    #             iters = DoParse(df)
    #             for i in range(len(iters)):
    #                 df = iters[i].format()
    #                 df.to_csv(os.path.join(outpath, 'data.{}.{}.csv'.format(data_i,i )), index=False)
    file = "/root/guohao/pytorch_benchmark/efficientnet_b7.9.sqlite.csv"
    df = pd.read_csv(file)
    iters = DoAnalyze(df)
    df = iters[0].format()
    print(df)
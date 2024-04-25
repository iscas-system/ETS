import argparse
import logging
import multiprocessing
import os.path
from multiprocessing import Process, Pool

from analyzer.do_analyze import DoAnalyze
from analyzer.do_parse import DoParse
from analyzer.do_prof import DoProf

process_num = 8
start_cpu_idx = 5
output_path = '/root/guohao/pytorch_benchmark/data/cnn/output/'
input_path = '/root/guohao/pytorch_benchmark/data/cnn/data/'
log_path = '/root/guohao/pytorch_benchmark/analyzer/'


def do_process(input_file: str):
    if not input_file.endswith('.sqlite'):
        return
    try:
        file_name = os.path.basename(input_file)
        model_name, config_idx, _ = file_name.split('.')
        output_file = os.path.join(output_path, f'{model_name}.{config_idx}_7.csv')
        if os.path.exists(output_file):
            # print('file %s already processed' % input_file)
            return None, None
        print('process %s' % input_file)
        kernels_dict = DoParse(input_file)
        print('parse %s done' % input_file)
        if kernels_dict is None or len(kernels_dict) == 0:
            return None, None
        # check dtype
        for kernel in kernels_dict:
            layer = kernel['layer'][0]
            if 'Float' not in layer:
                return None, None
            else:
                break
        print('prof %s start' % input_file)
        kernels_df = DoProf(kernels_dict)
        print('prof %s done' % input_file)

        results = DoAnalyze(kernels_df)
        print('process %s done' % input_file)
        return results[-1], output_file
    except Exception as e:
        print(input_file + ":" + str(e))
        with open(os.path.join(log_path, 'error.log'), 'a') as f:
            f.write(input_file + '\n')
    return None, None


def consumer(consumer_idx, file_q, result_q):
    """
    file_q的消费者，tar_file_q的生产者
    :param file_q:
    :return:
    """
    # set affinity
    import psutil
    p = psutil.Process()
    p.cpu_affinity([start_cpu_idx + consumer_idx])
    print(f"Child #{consumer_idx}: Set my affinity to {start_cpu_idx + consumer_idx}, affinity now {p.cpu_affinity()}",
          flush=True)
    while True:
        file = file_q.get()
        if file is None:
            result_q.put((None, None))
            break
        df, output_path = do_process(file)
        if df is None:
            continue
        print('put result of %s to save ' % file)
        result_q.put((df, output_path))
        print('put result of %s to save done' % file)
    print('consumer done')
    return


def unzip_process(tar_file_q, file_q):
    """
    tar_file_q的消费者，file_q的生产者
    :param tar_file_q:
    :return:
    """
    while True:
        tar_file = tar_file_q.get()
        if tar_file is None:
            for i in range(process_num + 1):
                file_q.put(None)
            break
        print('extract %s' % tar_file)
        dir_name = os.path.basename(tar_file)[:-7]
        dir_path = os.path.join(input_path, dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            os.system('tar -zxvf %s -C %s' % (os.path.join(input_path, tar_file), dir_path))
        print('extract %s done' % tar_file)
        for _, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.sqlite'):
                    file_q.put(os.path.join(dir_path, file))
        # os.system('rm -rf %s' % os.path.join(input_path, dir_name))
    print('unzip process done')
    return


def io_process(result_q):
    end_num = 0
    print('io process start')
    while True:
        df, output_path = result_q.get()
        if df is None:
            end_num += 1
            if end_num == process_num:
                break
            else:
                continue
        print('save %s' % output_path)
        df.format().to_csv(output_path, index=False)
    print('io process done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/root/guohao/pytorch_benchmark/data/cnn/')
    args = parser.parse_args()
    input_path = os.path.join(args.path, 'data')
    output_path = os.path.join(args.path, 'output')
    log_path = os.path.join(args.path, 'log')
    print('input path: %s' % input_path)

    tarfile_q = multiprocessing.Queue()
    file_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    # extract process
    p1 = multiprocessing.Process(target=unzip_process, args=(tarfile_q, file_q))

    # io process
    p2 = multiprocessing.Process(target=io_process, args=(result_q,))

    # worker process
    processors = []
    for i in range(process_num):
        processors.append(Process(target=consumer, args=(i, file_q, result_q)))
    p1.start()
    p2.start()
    for p in processors:
        p.start()
    # tar_file_q的生产者
    for _, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.tar.gz'):
                tarfile_q.put(file)
    tarfile_q.put(None)
    p1.join()
    for p in processors:
        p.join()
    p2.join()

    for _, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.tar.gz'):
                dir_name = os.path.basename(file)[:-7]
                dir_path = os.path.join(input_path, dir_name)
                os.system('rm -rf %s' % dir_path)
    print('all process done')

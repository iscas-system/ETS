import argparse
import logging
import random

import pyprof
import torch.cuda.profiler as profiler

from benchmark.models.input import get_input, get_in_channel
from benchmark.models.load import load_trad_model, ModelDescriptions, load_nlp_model
from benchmark.RandWireNN.model import Model
from benchmark.RandWireNN.preprocess import load_data, change_model_dtype

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.nvtx as nvtx

from benchmark.util import InputType
from utils.log import init_log


def run_nlp_config(batch_size, input_size, dtype, bidirectional, hidden_size, num_layers, model):
    num_classes = 3
    dtype = InputType.str2type[dtype]
    model_info = f'model={model}, batch={batch_size}, h={input_size},w={input_size}, dtype={dtype},'
    logging.info('start run ' + model_info)

    model = load_nlp_model(model=model, input_size=input_size, num_classes=num_classes, bidirectional=bidirectional,
                           hidden_size=hidden_size, num_layers=num_layers)
    input = torch.randn(batch_size, input_size)
    target = torch.randn(batch_size, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    _run(model, 7, input, target, criterion, optimizer, model_info)


def run_model_config(batch_size, input_size, dtype, model_name):
    dtype = InputType.str2type[dtype]
    model_info = f'model={model_name}, batch={batch_size}, h={input_size},w={input_size}, dtype={dtype},'
    logging.info('start run ' + model_info)
    model_description = None
    for md in ModelDescriptions:
        if md.value.name == model_name:
            model_description = md
            break
    if 'vit' in model_description.value.name:
        model_description.value.configs['image_size'] = input_size

    out_features = random.randint(3, 1000)
    model_description.value.configs['num_classes'] = out_features
    model = load_trad_model(model=model_description.value.name, dtype=dtype,
                            configs=model_description.value.configs)
    in_channel = get_in_channel(model)
    input = get_input(type=model_description.value.type, batch=batch_size,
                      in_channel=in_channel, h=input_size, w=input_size, dtype=dtype)
    target = torch.randn(batch_size, out_features)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    _run(model, 7, input, target, criterion, optimizer, model_info)


def run_rand_config(p, k, m, graph_mode, node_num, c, hw, learning_rate, batch_size, dtype, model_mode, dataset_mode,
                    **kwargs):
    n = 7
    model_name = 'rand_{}_{}_{}_{}_{}'.format(p, k, m, graph_mode, node_num)
    model_info = f'model={model_name}, batch={batch_size}, h={hw[0]},w={hw[1]}, dtype={torch.FloatTensor}'
    print('p:{}, k:{}, m:{}, graph_mode:{}, node_num:{}, c:{}, hw:{}, learning_rate:{}, batch_size:{}, dtype:{}, '
          'model_mode:{}, dataset_mode:{}'.format(p, k, m, graph_mode, node_num, c, hw, learning_rate, batch_size,
                                                  dtype, model_mode, dataset_mode))
    try:
        model = Model(node_num, p, c, c, graph_mode, model_mode,
                      dataset_mode,
                      True)
        model = change_model_dtype(model, dtype)

        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        input_data, target = load_data(dataset_mode, batch_size, 3, hw[0],
                                       hw[1],
                                       dtype)
    except Exception as e:
        print(e)
        return
    _run(model, n, input_data, target, criterion, optimizer, model_info)


def _run(model: torch.nn.Module, n: int, input: torch.Tensor, target: torch.Tensor, criterion, optimizer, model_info):
    """
    run model n times
    :param model:
    :param batch_size:
    :param n:
    :return:
    """
    # run_cuda(model, n, input, target, criterion, optimizer, model_info)
    try:
        run_cuda(model, n, input, target, criterion, optimizer, model_info)
    except BaseException as e:
        if 'out of memory' in str(e):
            logging.warning(e)
            logging.warning('out of memory')
        else:
            logging.warning('unknown exception')
            logging.warning(e)
        return

    with torch.autograd.profiler.emit_nvtx():
        logging.info('start profile')
        profiler.start()
        run_cuda(model, n, input, target, criterion, optimizer, model_info)
        profiler.stop()
        logging.info('end profile')


def run_cuda(model: torch.nn.Module, n: int, input: torch.Tensor, target: torch.Tensor, criterion, optimizer,
             model_info):
    model = model.cuda()
    model.train()
    criterion = criterion.cuda()
    input = input.cuda()
    target = target.cuda()
    for i in range(n):
        nvtx.range_push('layer:' + model_info + f'iter={i}')
        output = model(input)
        nvtx.range_pop()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    pyprof.init()
    init_log()
    parser = argparse.ArgumentParser(description='Run model')
    parser.add_argument('--model', type=str, default='resnet101',
                        help='model name')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--dtype', type=str, default='float')
    args = parser.parse_args()
    run_model_config(args.batch_size, args.input_size, args.dtype, args.model)
    # run_nlp_config(32, 64, 'float', True, 512, 2, 'lstm')
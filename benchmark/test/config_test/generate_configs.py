import torch
import random
from benchmark.util import InputType

def generate_er_configs():
    # 图参数
    ps = [0.75]  # graph probatility
    # ER之和p有关
    ks = [4]  # each node is connected to k nearest neighbors in ring topology,
    ms = [5]  # number of edges to attach from a new node to existing nodes in existing nodes
    graph_modes = ['ER']  # kinds of random graph
    node_nums = [8, 16, 32]
    # 模型参数
    cs = [78, 109, 154]
    hws = [[16, 16], [32, 32], [64, 64], [128, 128], [160, 160], [256, 256], [384, 384], [512, 512],
           [768, 768]]
    learning_rates = [1e-1]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128]
    dtypes = [torch.FloatTensor, torch.HalfTensor, torch.DoubleTensor]
    model_modes = ['CIFAR10', 'CIFAR100', 'SMALL_REGIME', 'REGULAR_REGIME']  # 不同的模型结构
    dataset_modes = ['CIFAR10']  # 只要一个就可以了
    configs = []
    for p in ps:
        for c in cs:
            for k in ks:
                for m in ms:
                    for hw in hws:
                        for graph_mode in graph_modes:
                            for node_num in node_nums:
                                for learning_rate in learning_rates:
                                    for batch_size in batch_sizes:
                                        for dtype in dtypes:
                                            for model_mode in model_modes:
                                                for dataset_mode in dataset_modes:
                                                    d = {'p': p, 'c': c, 'k': k, 'm': m, 'hw': hw,
                                                         'graph_mode': graph_mode,
                                                         'node_num': node_num, 'learning_rate': learning_rate,
                                                         'batch_size': batch_size, 'dtype': InputType.types2id[dtype],
                                                         'model_mode': model_mode, 'dataset_mode': dataset_mode}
                                                    configs.append(d)
    return configs

def generate_ws_configs():
    # 图参数
    ps = [0.75]  # graph probatility
    # WS和k, p有关
    ks = [3,4,5]  # each node is connected to k nearest neighbors in ring topology,
    ms = [5]  # number of edges to attach from a new node to existing nodes in existing nodes
    graph_modes = ['WS']  # kinds of random graph
    node_nums = [8, 16, 32]
    # 模型参数
    cs = [78, 109, 154]
    hws = [[16, 16], [32, 32], [64, 64], [128, 128], [160, 160], [256, 256], [384, 384], [512, 512],
           [768, 768]]
    learning_rates = [1e-1]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128]
    dtypes = [torch.FloatTensor, torch.HalfTensor, torch.DoubleTensor]
    model_modes = ['CIFAR10', 'CIFAR100', 'SMALL_REGIME', 'REGULAR_REGIME']  # 不同的模型结构
    dataset_modes = ['CIFAR100']  # 只要一个就可以了
    configs = []
    for p in ps:
        for c in cs:
            for k in ks:
                for m in ms:
                    for hw in hws:
                        for graph_mode in graph_modes:
                            for node_num in node_nums:
                                for learning_rate in learning_rates:
                                    for batch_size in batch_sizes:
                                        for dtype in dtypes:
                                            for model_mode in model_modes:
                                                for dataset_mode in dataset_modes:
                                                    d = {'p': p, 'c': c, 'k': k, 'm': m, 'hw': hw,
                                                         'graph_mode': graph_mode,
                                                         'node_num': node_num, 'learning_rate': learning_rate,
                                                         'batch_size': batch_size, 'dtype': InputType.types2id[dtype],
                                                         'model_mode': model_mode, 'dataset_mode': dataset_mode}
                                                    configs.append(d)
    return configs

def generate_ba_configs():
    # 图参数
    ps = [0.75]  # graph probatility
    # BA之和m有关
    ks = [4]  # each node is connected to k nearest neighbors in ring topology,
    ms = [3,5,7]  # number of edges to attach from a new node to existing nodes in existing nodes
    graph_modes = ['BA']  # kinds of random graph
    node_nums = [8, 16, 32]
    # 模型参数
    cs = [78, 109, 154]
    hws = [[16, 16], [32, 32], [64, 64], [128, 128], [160, 160], [256, 256], [384, 384], [512, 512],
           [768, 768]]
    learning_rates = [1e-1]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128]
    dtypes = [torch.FloatTensor, torch.HalfTensor, torch.DoubleTensor]
    model_modes = ['CIFAR10', 'CIFAR100', 'SMALL_REGIME', 'REGULAR_REGIME']  # 不同的模型结构
    dataset_modes = ['MNIST']  # 只要一个就可以了
    configs = []
    for p in ps:
        for c in cs:
            for k in ks:
                for m in ms:
                    for hw in hws:
                        for graph_mode in graph_modes:
                            for node_num in node_nums:
                                for learning_rate in learning_rates:
                                    for batch_size in batch_sizes:
                                        for dtype in dtypes:
                                            for model_mode in model_modes:
                                                for dataset_mode in dataset_modes:
                                                    d = {'p': p, 'c': c, 'k': k, 'm': m, 'hw': hw,
                                                         'graph_mode': graph_mode,
                                                         'node_num': node_num, 'learning_rate': learning_rate,
                                                         'batch_size': batch_size, 'dtype':InputType.types2id[dtype] ,
                                                         'model_mode': model_mode, 'dataset_mode': dataset_mode}
                                                    configs.append(d)
    return configs

if __name__ == '__main__':
    configs = generate_ws_configs()
    configs += generate_ba_configs()
    configs += generate_er_configs()
    print(len(configs))
    configs = random.sample(configs, 45000)
    with open('/root/guohao/pytorch_benchmark/benchmark/rand_configs.dict', 'w') as f:
        for config in configs:
            f.write(str(config)+'\n')

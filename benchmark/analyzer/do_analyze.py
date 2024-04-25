import argparse
import logging
import os

import pandas as pd
from enum import Enum
from utils.log import init_log

from analyzer.op import getOpCodes
from analyzer.util import parse_grid, parse_block, parse_params, parseModelInfo


class Phase(Enum):
    forward = 1
    backward = 2
    optimizer = 3


class KernelInfo:
    def __init__(self, op: str, params: str, kernel: str, grid: str, block: str, layer: str,
                 rStartTime: int, rEndTime: int, kStartTime: int, bytes: int, flops: int, kEndTime: int,
                 staticSharedMemory: int, sharedMemoryExecuted: int, **kwargs):
        self.op = op
        self.kernel = kernel
        self.layer = layer
        self.params = parse_params(params)
        self.grid = parse_grid(grid)
        self.block = parse_block(block)
        self.flops = flops
        self.bytes = bytes
        self.rStartTime = rStartTime
        self.rEndTime = rEndTime
        self.kStartTime = kStartTime
        self.kEndTime = kEndTime
        self.staticSharedMemory = staticSharedMemory
        self.dir = 0
        # self.dynamicSharedMemory = dynamicSharedMemory
        # self.localMemoryTotal = localMemoryTotal
        # self.localMemoryPerThread = localMemoryPerThread
        self.sharedMemoryExecuted = sharedMemoryExecuted

    def format(self):
        return {
            'op': self.op,
            'dir': self.dir.value,
            # 'kernel': self.kernel,
            'params': self.params,
            'grid': self.grid,
            'block': self.block,
            'flops': self.flops,
            'bytes': self.bytes,
            'rStartTime': self.rStartTime,
            'rEndTime': self.rEndTime,
            'kStartTime': self.kStartTime,
            'kEndTime': self.kEndTime,
            'staticSharedMemory': self.staticSharedMemory,
            # 'sharedMemoryExecuted': self.sharedMemoryExecuted,
            'space': 0,  # 核函数对齐后的算子间隙
            'space1': 0,  # 核函数没有对其，seq之间的间隙
            # 'dynamicSharedMemory': self.dynamicSharedMemory,
            # 'localMemoryTotal': self.localMemoryTotal,
            # 'localMemoryPerThread': self.localMemoryPerThread,
        }


class OpItem:
    def __init__(self, seq: int, altseq, layer: str, mod: str, op: str, **kwargs):
        self.seq = seq
        self.altseq = altseq
        self.layer = layer  # forward时初始化
        self.mod = mod
        self.op = op
        self.fpop = []  # 前向传播
        self.bpop = []  # 反向传播
        self.update = []  # 梯度更新
        self.optimizer = []  # optimizer更新参数，一般包含一个mul和两个add

    def add_fpop(self, kernel: KernelInfo):
        if 'add' in self.op and 'add' not in kernel.op:
            self.op = kernel.op
        kernel.dir = Phase.forward
        self.fpop.append(kernel)

    def add_bpop(self, kernel: KernelInfo):
        if 'add' in self.op and 'add' not in kernel.op:
            self.op = kernel.op
        kernel.dir = Phase.backward
        self.bpop.append(kernel)

    def add_optimizer(self, kernel: KernelInfo):
        kernel.dir = Phase.optimizer
        self.optimizer.append(kernel)

    def merge_op(self, op_item):
        """
        合并altseq
        :param op_item:
        :return:
        """
        self.fpop += op_item.fpop
        self.bpop += op_item.bpop
        self.optimizer += op_item.optimizer

    def merge_kernels(self, kernels):
        """
        合并前向传播的多个kernel
        :return:
        """
        if len(kernels) == 0:
            return {}
        kernels_info = kernels[0].format()
        kernels_info['seq'] = self.seq
        kernels_info['op'] = (getOpCodes())[self.op]
        kernels_info['kernels'] = 1
        kernels_info['rduration'] = kernels_info['rEndTime'] - kernels_info['rStartTime']
        kernels_info['kduration'] = kernels_info['kEndTime'] - kernels_info['kStartTime']
        """
        合并前向传播
        :return:
        """
        for i in range(1, len(kernels)):
            kernel = kernels[i]
            kernels_info['kernels'] += 1
            if len(kernel.params) > len(kernels_info['params']):
                kernels_info['params'] = kernel.params
            kernels_info['flops'] = max(kernels_info['flops'], kernel.flops)
            kernels_info['bytes'] = max(kernels_info['bytes'], kernel.bytes)
            kernels_info['rStartTime'] = min(kernels_info['rStartTime'], kernel.rStartTime)
            kernels_info['rEndTime'] = max(kernels_info['rEndTime'], kernel.rEndTime)
            kernels_info['kStartTime'] = min(kernels_info['kStartTime'], kernel.kStartTime)
            kernels_info['kEndTime'] = max(kernels_info['kEndTime'], kernel.kEndTime)
            kernels_info['staticSharedMemory'] = max(kernels_info['staticSharedMemory'], kernel.staticSharedMemory)
            # kernels_info['sharedMemoryExecuted'] = max(kernels_info['sharedMemoryExecuted'],
            #                                            kernel.sharedMemoryExecuted)
            kernels_info['rduration'] += kernel.rEndTime - kernel.rStartTime
            kernels_info['kduration'] += kernel.kEndTime - kernel.kStartTime
            kernels_info['grid'] = max(kernels_info['grid'], kernel.grid)
            kernels_info['block'] = max(kernels_info['block'], kernel.block)
        return kernels_info

    # todo 转化为excel，包含forward, backward, 和update的时间
    def format(self):
        forward_info = self.merge_kernels(self.fpop)
        backward_info = self.merge_kernels(self.bpop)
        update_info = []
        # for kernel in self.optimizer:
        #     kernel_info = kernel.format()
        #     kernel_info.update(
        #         {
        #             "seq": self.seq,
        #             "op": (getOpCodes())[self.op],
        #             "kernels": 1,
        #             "rduration": kernel_info['rEndTime'] - kernel_info['rStartTime'],
        #             "kduration": kernel_info['kEndTime'] - kernel_info['kStartTime']
        #         }
        #     )
        #     update_info.append(kernel_info)
        return forward_info, backward_info, update_info


class IterInfo:
    def __init__(self, op_items, optimizers, model_info):
        self.opItems = op_items
        self.model_info = model_info
        self.optimizers = optimizers

    def merge_dummy(self):
        op_lists = sorted(self.opItems.values(), key=lambda x: x.seq, reverse=False)
        remove_idx = []
        merge_ops = ["multi_head_attention_forward", "cross_entropy"]
        for i in range(len(op_lists)):
            if i not in remove_idx and op_lists[i].op in merge_ops:
                for j in range(i + 1, len(op_lists)):
                    if op_lists[i].op == op_lists[j].op:
                        op_lists[i].merge_op(op_lists[j])
                        remove_idx.append(j)
                    else:
                        break
        for idx in remove_idx:
            del self.opItems[op_lists[idx].seq]

    def format(self):
        self.merge_dummy()
        forward_info = []
        backward_info = []
        update_info = []
        model_info = parseModelInfo(self.model_info)
        for seq, op_item in self.opItems.items():
            forward, backward, _ = op_item.format()
            if len(forward) > 0:
                forward.update(model_info)
                forward_info.append(forward)
            if len(backward) > 0:
                backward.update(model_info)
                backward_info.append(backward)
        for kernel in self.optimizers:
            kernel.dir = Phase.optimizer
            kernel_info = kernel.format()
            kernel_info.update(
                {
                    "seq": -1,
                    "op": (getOpCodes())[kernel.op],
                    "kernels": 1,
                    "rduration": kernel_info['rEndTime'] - kernel_info['rStartTime'],
                    "kduration": kernel_info['kEndTime'] - kernel_info['kStartTime']
                }
            )
            kernel_info.update(model_info)
            update_info.append(kernel_info)
        backward_info = sorted(backward_info, key=lambda x: x['seq'], reverse=True)
        tmp = [pd.DataFrame(forward_info), pd.DataFrame(backward_info), pd.DataFrame(update_info)]
        df = pd.concat(tmp, ignore_index=True)
        # df = pd.DataFrame(forward_info)
        # df = pd.concat([df, pd.DataFrame(backward_info)], ignore_index=True)
        # df = pd.concat([df, pd.DataFrame(update_info)], ignore_index=True)
        for i in range(len(df) - 1):
            df.loc[i, 'space'] = df.loc[i + 1, 'kStartTime'] - (df.loc[i, 'kStartTime'] + df.loc[i, 'kduration'])
            df.loc[i, 'space1'] = df.loc[i + 1, 'kStartTime'] - df.loc[i, 'kEndTime']
        return df


def DoAnalyze(df: pd.DataFrame) -> list:
    iters = []
    iter = i = 0
    print('start Parse')
    while i <= len(df):
        if i == len(df):
            iters.append(IterInfo(opItems, optimizers, model_info))
            break
        iter += 1  # 第几次训练
        # parse forward
        # logging.info(f"parse forward for iter {iter},index {i}")
        model_info = '-'
        opItems = {}
        optimizers = []
        phase = Phase.forward
        last_op = None
        start_seq = 0xffffffff  # iteration开始的seq
        while i < len(df):
            # logging.info(f"parse info for iter {iter},index {i}")
            row_dict = df.iloc[i].to_dict()
            i += 1
            # 状态流转
            if phase == Phase.forward and row_dict['dir'] == 'bprop':
                # logging.info(f"parse backward for iter {iter},index {i}")
                phase = Phase.backward
            elif phase == Phase.backward and row_dict['dir'] == 'fprop' and row_dict['seq'] == '-' and row_dict[
                'layer'] == '-' and row_dict['trace'] != '-':
                # logging.info(f"parse optimizer for iter {iter},index {i}")
                phase = Phase.optimizer
                start_idx = i  # optimizer开始的行数
            elif phase == Phase.optimizer and row_dict['seq'] != '-':
                iters.append(IterInfo(opItems, optimizers, model_info))
                i -= 1
                break
            if row_dict['layer'] != '-':
                model_info = row_dict['layer']
            # 处理
            if phase == Phase.forward:
                if row_dict['seq'] != '-':
                    row_dict['seq'] = int(row_dict['seq'])
                else:
                    continue
                seq = row_dict['seq']
                start_seq = min(seq, start_seq)
                if seq not in opItems.keys():
                    opItems[seq] = OpItem(**row_dict)
                opItems[seq].add_fpop(KernelInfo(**row_dict))
                last_op = opItems[seq]
            elif phase == Phase.backward:
                if row_dict['seq'] == '-' and row_dict['altseq'] == '-':
                    if row_dict['dir'] == 'fpop' and last_op != None:
                        last_op.add_update(KernelInfo(**row_dict))
                    else:
                        last_op = None
                        continue
                else:
                    if row_dict['seq'] == '-':
                        row_dict['seq'] = row_dict['altseq']
                    seq = row_dict['seq'] = int(row_dict['seq'])
                    if seq not in opItems.keys():
                        last_op = None
                        continue  # 没有对应的forward
                    opItems[seq].add_bpop(KernelInfo(**row_dict))
                    last_op = opItems[seq]
            elif phase == Phase.optimizer:
                optimizers.append(KernelInfo(**row_dict))
    return iters


def format2csv(iters: list):
    """
    格式化输出
    :param iters:
    :return:
    """
    for iter in iters:
        forward_info = []
        backward_info = []
        update_info = []
        for seq in iter.keys():
            forward, backward, update = iter[seq].format()
            forward_info.append(forward)
            backward_info.append(backward)
            update_info += update

        df = pd.DataFrame(forward_info)
    return forward_info, backward_info, update_info


if __name__ == '__main__':
    init_log()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input file')
    parser.add_argument('--output', type=str, help='output path')
    args = parser.parse_args()

    file_name = os.path.basename(args.input)
    model_name, config_idx, _, _ = file_name.split('.')
    df = pd.read_csv(args.input)
    iters = DoAnalyze(df)
    for i in range(1, len(iters)):
        df = iters[i].format()
        df.to_csv(os.path.join(args.output, f'{model_name}_{config_idx}_{i}.csv'), index=False)

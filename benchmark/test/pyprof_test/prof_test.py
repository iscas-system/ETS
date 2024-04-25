"""
this package is used to config_test prof in pyprof
"""
import argparse

import torch
from pyprof.prof.data import Data
from pyprof.prof.utility import Utility
from pyprof.prof.output import Output
from pyprof.prof.prof import findFpropKernel, foo


def doProf(args):

    output = Output(args)
    idx = -1
    # Read in all the kernel info
    with open(args.file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        idx += 1
        kernel = eval(line)
        assert (kernel)
        kernels.append(kernel)

        k = kernel
        d = Data(k)

        mod = k['mod']
        op = k['op']

        flops = 0
        params = {"na": "na"}
        tc = "na"
        bytes = 0
        # # handle seq
        # if len(d.altSeqId) > 0 and len(d.seqId) > 0:
        #     if(min(d.altSeqId) < min(d.seqId)):
        #         d.seqId = [min(d.altSeqId)]
        #     else:
        #         d.seqId = [min(d.seqId)]

        if (d.dir == "bprop"):
            d.seqMarker = k['seqMarker']
            seq = k['seqId']
            if len(seq) > 1:
                pass
            seq = k['seqId'][:1]
            assert (len(seq) == 1), seq
            # assert (seq[0] != 0)
            assert (len(d.seqMarker) > 0)
            # If there is no useful marker associated, use the
            # sequence number to find the kernel from fprop
            if len(d.argMarker) == 0:
                index = findFpropKernel(seq[0])
                if index >= 0:
                    d.argMarker = kernels[index]['marker']
                    d.modMarker = kernels[index]['reprMarkers']
                    mod = kernels[index]['mod']
                    op = kernels[index]['op']

                    d.layer = kernels[index]['layer']
                    d.trace = kernels[index]['trace']

        # Check if marker has our annotations
        if len(d.argMarker) and Utility.hasNVTX(d.argMarker[0]):
            if op[0] == 'linear':
                print(op)
            print(op)
            xx = foo(mod, op, d)

            bytes = xx.bytes()
            flops = xx.flops()
            op = xx.op()
            params = xx.params()
            tc = xx.tc()

        if type(op) is list:
            if len(op):
                op = op[0]
            else:
                op = ""

        if type(mod) is list:
            if len(mod):
                mod = mod[0]
            else:
                mod = ""

        d.index = idx + 1

        # The following 8 come from operator class functions.
        d.setParams(params)
        d.tc = tc
        d.flops = flops
        d.bytes = bytes
        d.mod = mod
        d.op = op

        # memory
        output.add(d)
    output.save()

kernels = []
if __name__ == "__main__":
    # config_test nvprof
    # doParse("/root/guohao/ml_predict/out/pyprof/resnet/nvprof/resnet.sql")
    # config_test nsight
    parser = argparse.ArgumentParser()
    # 75 76 77 78
    parser.add_argument('--file', type=str, default="/root/guohao/pytorch_benchmark/tmp/data.78.sqlite.dict")
    parser.add_argument('--output', type=str, default='/root/guohao/pytorch_benchmark/tmp/data.78.sqlite.csv')
    args = parser.parse_args()
    doProf(args)

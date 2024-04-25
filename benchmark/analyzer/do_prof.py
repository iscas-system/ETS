"""
This script reads the output (Python dictionary) created by parse.py.
For every kernel (line) in the input it determines
	module / class name e.g. torch.nn.functional
	operator name e.g. linear
	kernel parameters e.g. GEMM M, N, K, datatype
	bytes
	flops
	tensor core usage
	direction (fprop, bprop)
	and other things. Please see the tool usage.
"""

from pyprof.prof.usage import parseArgs
from pyprof.prof.output import Output
from pyprof.prof.utility import Utility
from pyprof.prof.pointwise import Pointwise
from pyprof.prof.convert import Convert
from pyprof.prof.blas import *
from pyprof.prof.embedding import Embedding
from pyprof.prof.reduction import *
from pyprof.prof.dropout import Dropout
from pyprof.prof.softmax import *
# from pooling import * # work in progress
from pyprof.prof.linear import Linear
from pyprof.prof.optim import Adam
from pyprof.prof.misc import *
from pyprof.prof.conv import Conv
from pyprof.prof.activation import Activation
from pyprof.prof.index_slice_join_mutate import Cat, Reshape, MaskedScatter, Gather, Nonzero, IndexSelect, MaskedSelect
from pyprof.prof.recurrentCell import RNNCell
from pyprof.prof.normalization import BatchNorm
from pyprof.prof.randomSample import RandPerm
from pyprof.prof.loss import MSELoss
from pyprof.prof.data import Data
from pyprof.prof.memory import OneZero, Fill, Full


def findFpropKernel(seq, kernel_seq_map, kernel_alt_seq_map):
    # Find the last fprop kernel with the same seqId
    # First look at seqId and then at altSeqId
    if seq in kernel_seq_map.keys():
        return kernel_seq_map[seq][-1]
    if seq in kernel_alt_seq_map.keys():
        return kernel_alt_seq_map[seq][-1]
    return -1
    # print("Error: seqId {} not found.".format(seq), file=sys.stderr)
    # assert False


def foo(mod, op, d):
    if (op[0] == "linear"):
        xx = Linear(d)

    # rnncell, lstmcell, grucell
    elif (mod[0] in ["LSTMCell", "GRUCell"]) and (op[0] == "forward"):
        xx = RNNCell(d)

    elif op[0] in [
        "conv1d",
        "conv2d",
    ]:
        xx = Conv(d)

    elif (op[0] in Pointwise.ops):
        xx = Pointwise(d)

    elif (op[0] in Convert.ops):
        xx = Convert(d)

    elif op[0] in ["__matmul__", "matmul"]:
        xx = Matmul(d)

    elif op[0] == "embedding":
        xx = Embedding(d)

    # reduction
    elif op[0] == "sum":
        xx = Sum(d)

    elif op[0] == "mean":
        xx = Mean(d)

    elif op[0] == "norm":
        xx = Norm(d)

    elif op[0] == "dropout":
        xx = Dropout(d)

    # Index, Slice, Join, Mutate
    elif (op[0] == "cat"):
        xx = Cat(d)

    elif (op[0] == "reshape"):
        xx = Reshape(d)

    elif (op[0] == "masked_scatter_"):
        xx = MaskedScatter(d)

    elif (op[0] == "gather"):
        xx = Gather(d)

    elif (op[0] == "nonzero"):
        xx = Nonzero(d)

    elif (op[0] == "index_select"):
        xx = IndexSelect(d)

    elif (op[0] == "masked_select"):
        xx = MaskedSelect(d)

    # blas
    elif op[0] in ["addmm", "addmm_"]:
        xx = Addmm(d)

    elif op[0] == "mm":
        xx = Mm(d)

    elif op[0] == "bmm":
        xx = Bmm(d)

    # softmax
    elif op[0] == "softmax":
        xx = Softmax(d)

    elif op[0] == "log_softmax":
        xx = LogSoftmax(d)

    # loss
    elif op[0] == "mse_loss":
        xx = MSELoss(d)

    # optimizers
    elif op[0] == "adam":
        xx = Adam(d)

    # normalization
    elif op[0] == "batch_norm":
        xx = BatchNorm(d)

    # random
    elif op[0] == "randperm":
        xx = RandPerm(d)

    # memory
    elif op[0] in OneZero.ops:
        xx = OneZero(d)

    elif op[0] == "fill_":
        xx = Fill(d)

    elif op[0] == "full":
        xx = Full(d)

    # misc
    elif op[0] == "copy_":
        xx = Copy(d)

    elif op[0] == "clone":
        xx = Clone(d)

    elif op[0] == "contiguous":
        xx = Contiguous(d)

    elif op[0] == "any":
        xx = Any(d)

    elif (op[0] in Activation.ops):
        xx = Activation(d)

    elif op[0] == "to":
        xx = Convert(d)

    else:
        xx = Foo(d)

    return xx


kernels = []


def DoProf(parse_infos: []):
    # Read cmd line arguments
    kernels = []
    output = Output()
    kernel_seq_map = {}
    kernel_alt_seq_map = {}
    idx = -1
    # Read in all the kernel info
    for kernel in parse_infos:
        idx += 1
        # print(idx)
        assert (kernel)
        kernels.append(kernel)
        if kernel['dir'] == 'fprop':
            for seqid in kernel['seqId']:
                if seqid not in kernel_seq_map.keys():
                    kernel_seq_map[seqid] = [idx]
                else:
                    kernel_seq_map[seqid].append(idx)
            for seqid in kernel['altSeqId']:
                if seqid not in kernel_alt_seq_map.keys():
                    kernel_alt_seq_map[seqid] = [idx]
                else:
                    kernel_alt_seq_map[seqid].append(idx)
        k = kernel
        d = Data(k)

        mod = k['mod']
        op = k['op']

        flops = 0
        params = {"na": "na"}
        tc = "na"
        bytes = 0
        # # handle seq
        # if len(d.altSeqId) > 0 and len(d.seqId) > 0 and int(d.altSeqId) < int(d.seqId):
        #     d.seqId = d.altSeqId

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
                index = findFpropKernel(seq[0], kernel_seq_map, kernel_alt_seq_map)
                if index >= 0:
                    d.argMarker = kernels[index]['marker']
                    d.modMarker = kernels[index]['reprMarkers']
                    mod = kernels[index]['mod']
                    op = kernels[index]['op']

                    d.layer = kernels[index]['layer']
                    d.trace = kernels[index]['trace']

        # Check if marker has our annotations
        if len(d.argMarker) and Utility.hasNVTX(d.argMarker[0]):
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
        if op == 'bias':
            op = 'linear'
        d.op = op

        # memory
        output.add(d)
    return output.df

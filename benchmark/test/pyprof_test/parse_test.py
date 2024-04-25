"""
this package is used to config_test parse in pyprof
"""
import sys

import torch
from pyprof.parse import db, nsight, nvvp, kernel
from tqdm import tqdm


def dbIsNvvp(dbInstance):
    cmd = "SELECT * FROM sqlite_master where type='table' AND name='StringTable'"
    result = dbInstance.select(cmd)
    return True if len(result) == 1 else False


def doParse(file):
    dbInstance = db.DB(file)
    nvvpInstance = None
    if dbIsNvvp(dbInstance):
        nvvpInstance = nvvp.NVVP(dbInstance)
    else:
        nvvpInstance = nsight.Nsight(dbInstance)

    kInfo = nvvpInstance.getKernelInfo()
    if len(kInfo) == 0:
        print("Found 0 kernels. Exiting.", file=sys.stderr)
        dbInstance.close()
        sys.exit(0)
    else:
        print("Found {} kernels. Getting info for each kernel.".format(len(kInfo)), file=sys.stderr)

    nvvpInstance.createMarkerTable()

    prevSeqId = -1
    prevSubSeqId = -1
    prevOp = "na"

    kernel.Kernel.profStart = nvvpInstance.getProfileStart()

    for i in tqdm(range(len(kInfo)), ascii=True):
        info = kInfo[i]
        k = kernel.Kernel()

        # Calculate/encode object ID
        if info["tid"] == None or info["pid"] == None:
            print("no tid/pid in kernel info {}".format(info))
            continue
        nvvpInstance.encode_object_id(info)

        # Set kernel info
        k.setKernelInfo(info)

        # Get and set marker and seqid info
        info = nvvpInstance.getMarkerInfo(k.objId, k.rStartTime, k.rEndTime)
        k.setMarkerInfo(info)

        # If the seqId contains both 0 and non zero integers, remove 0.
        if any(seq != 0 for seq in k.seqId) and (0 in k.seqId):
            k.seqId.remove(0)

        # Set direction (it uses seq id)
        k.setDirection()

        # Set op
        k.setOp()

        # The following code is based on heuristics.
        # TODO: Refactor.
        # Assign subSeqId, adjust seqId and altSeqId
        # seqId can be 0.
        # A kernel can have multiple seqIds both in fprop and bprop.
        # In bprop, seqIds might not decrease monotonically. I have observed a few blips.
        if len(k.seqId):
            assert (k.dir in ["fprop", "bprop"])
            if (k.dir == "fprop"):
                # Check if there is a sequence id larger than the previous
                inc = (k.seqId[-1] > prevSeqId)
                if inc:
                    currSeqId = [x for x in k.seqId if x > prevSeqId][0]
                else:
                    currSeqId = prevSeqId
            else:
                currSeqId = k.seqId[0]

            # if ((currSeqId == prevSeqId) and (k.op == prevOp)):
            if ((currSeqId == prevSeqId) and (k.op == prevOp)) or ((k.op[0] == "forward") and (k.op == prevOp) and
                                                                   (k.mod[0] in ["LSTMCell", "GRUCell",
                                                                                 "RNNCell"])):
                # The second condition is to trap cases when pytorch does not use cudnn for a LSTMCell.
                k.subSeqId = prevSubSeqId + 1

            prevSeqId = currSeqId
            prevSubSeqId = k.subSeqId
            prevOp = k.op

            new_altSeqId = k.altSeqId
            if currSeqId in k.altSeqId:
                k.altSeqId.remove(currSeqId)
            for s in k.seqId:
                if s != currSeqId:
                    new_altSeqId.append(s)
            k.altSeqId = new_altSeqId
            k.seqId = [currSeqId]

            k.altSeqId = list(set(k.altSeqId))
            if (len(k.altSeqId)):
                (k.altSeqId).sort()

        k.print()
    dbInstance.close()


if __name__ == "__main__":
    # config_test nvprof
    # doParse("/root/guohao/ml_predict/out/pyprof/resnet/nvprof/resnet.sql")
    # config_test nsight
    # 21 & 22
    #
    doParse("/root/guohao/pytorch_benchmark/tmp/data.78.sqlite")

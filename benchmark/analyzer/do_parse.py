"""
Parse the SQLite3 database from NVprof or Nsight and print a dictionary for every kernel.
"""

import sys
import os

from pyprof.parse.db import DB
from pyprof.parse.kernel import Kernel
from pyprof.parse.nvvp import NVVP
from pyprof.parse.nsight import Nsight


def dbIsNvvp(db):
    cmd = "SELECT * FROM sqlite_master where type='table' AND name='StringTable'"
    result = db.select(cmd)
    return True if len(result) == 1 else False


def DoParse(sqlite_file: str, memory=True):
    kernels = []
    db = DB(sqlite_file)
    nvvp = None
    if dbIsNvvp(db):
        nvvp = NVVP(db)
    else:
        nvvp = Nsight(db, memory)
    tables = db.select("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [x['name'] for x in tables]
    # print(tables)
    if nvvp.kernelT not in tables or nvvp.runtimeT not in tables or nvvp.markerT not in tables:
        print("No kernel table found. Exiting.")
        db.close()
        return kernels
    kInfo = nvvp.getKernelInfo()
    if len(kInfo) == 0:
        print("Found 0 kernels. Exiting.", file=sys.stderr)
        db.close()
        sys.exit(0)
    else:
        print("Found {} kernels. Getting info for each kernel.".format(len(kInfo)), file=sys.stderr)

    nvvp.createMarkerTable()

    prevSeqId = -1
    prevSubSeqId = -1
    prevOp = "na"

    Kernel.profStart = nvvp.getProfileStart()

    for i in range(len(kInfo)):
        info = kInfo[i]
        k = Kernel()

        # Calculate/encode object ID
        nvvp.encode_object_id(info)

        # Set kernel info
        k.setKernelInfo(info)

        # Get and set marker and seqid info
        info = nvvp.getMarkerInfo(k.objId, k.rStartTime, k.rEndTime)
        k.setMarkerInfo(info)

        # If the seqId contains both 0 and non zero integers, remove 0.
        if any(seq != 0 for seq in k.seqId) and (0 in k.seqId):
            k.seqId.remove(0)

        # Set direction (it uses seq id)
        k.setDirection()
        # if i >= len(kInfo)-4:
        #     print(k.pyprofMarkers)
        #     print(k.seqMarkers)
        #     print(k.otherMarkers)
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
                                                                   (k.mod[0] in ["LSTMCell", "GRUCell", "RNNCell"])):
                # The second condition is to trap cases when pytorch does not use cudnn for a LSTMCell.
                k.subSeqId = prevSubSeqId + 1

            prevSeqId = currSeqId
            prevSubSeqId = k.subSeqId
            prevOp = k.op

            # Keep currSeqId in k.seqId, move everything else to k.altSeqId
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
        kernels.append(k.print())
    db.close()
    return kernels

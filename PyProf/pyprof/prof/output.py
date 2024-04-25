#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import errno
import os
import sys

import pandas as pd


class Output():
    """
        This class handles printing of a columed output and a CSV.
        """

    # The table below is organized as
    # user_option: [output_header, attribute_in_Data_class, type, min_width_in_columed_output]
    table = {
        "idx": ["Idx", "index", int, 7],
        "seq": ["SeqId", "seqId", str, 7],
        "altseq": ["AltSeqId", "altSeqId", str, 7],
        "tid": ["TId", "tid", int, 12],
        "layer": ["Layer", "layer", str, 10],
        "trace": ["Trace", "trace", str, 25],
        "dir": ["Direction", "dir", str, 5],
        "sub": ["Sub", "sub", int, 3],
        "mod": ["Module", "mod", str, 15],
        "op": ["Op", "op", str, 15],
        "kernel": ["Kernel", "name", str, 0],
        "params": ["Params", "params", str, 0],
        "sil": ["Sil(ns)", "sil", int, 10],
        "tc": ["TC", "tc", str, 2],
        "device": ["Device", "device", int, 3],
        "stream": ["Stream", "stream", int, 3],
        "grid": ["Grid", "grid", str, 12],
        "block": ["Block", "block", str, 12],
        "flops": ["FLOPs", "flops", int, 12],
        "bytes": ["Bytes", "bytes", int, 12],
        "rStartTime": ["rStartTime", "rStartTime", int, 12],
        "rEndTime": ["rEndTime", "rEndTime", int, 12],
        "kStartTime": ["kStartTime", "kStartTime", int, 12],
        "kEndTime": ["kEndTime", "kEndTime", int, 12],
        "staticSharedMemory": ["staticSharedMemory", "staticSharedMemory", int, 12],
        "dynamicSharedMemory": ["dynamicSharedMemory", "dynamicSharedMemory", int, 12],
        "localMemoryTotal": ["localMemoryTotal", "localMemoryTotal", int, 12],
        "localMemoryPerThread": ["localMemoryPerThread", "localMemoryPerThread", int, 12],
        "sharedMemoryExecuted": ["sharedMemoryExecuted", "sharedMemoryExecuted", int, 12],
    }

    def __init__(self):
        self.cols = list(self.table.keys())
        self.df = pd.DataFrame(columns=self.cols)

    def add(self, a):
        if a.dir == "":
            direc = "na"
        else:
            direc = a.dir

        if a.op == "":
            op = "na"
        else:
            op = a.op

        if a.mod == "":
            mod = "na"
        else:
            mod = a.mod

        d = {}
        for col in self.cols:
            attr = Output.table[col][1]
            val = getattr(a, attr)

            if col == "layer":
                assert (isinstance(val, list))
                val = ":".join(val)
                val = "-" if val == "" else val

            if col == "trace":
                assert (isinstance(val, list))
                # if len(val):
                #     val = val[-1]
                #     val = val.split("/")[-1]
                # else:
                val = ",".join(val)
                val = "-" if val == "" else val

            if col in ["seq", "altseq"]:
                assert (isinstance(val, list))
                val = ",".join(map(str, val))
                val = "-" if val == "" else val

            d[col] = [val]

        self.df = pd.concat(
            [self.df, pd.DataFrame.from_dict(d)], ignore_index=True)

    def save(self, save_path):
        self.df.to_csv(save_path, index=False)

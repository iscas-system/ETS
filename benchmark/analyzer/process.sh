#!/usr/bin/env bash
# dell05
project_path=/root/guohao/pytorch_benchmark
data_path=/root/guohao/data
output_path=/root/guohao/cnn/output
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gh-torch
export PYTHONPATH=${project_path}:${PYTHONPATH}
# variables


python /root/guohao/pytorch_benchmark/analyzer/process.py --path /root/guohao/data/TeslaT4/rand

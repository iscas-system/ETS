#!/usr/bin/env bash
# python environment
project_path=/root/guohao/pytorch_benchmark
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gh-torch
export PYTHONPATH=${project_path}:${PYTHONPATH}

# variables
data_path=${project_path}/data/cnn/data
output_path=$project_path/data/cnn/output

for file in $(ls ${data_path}/*.sqlite); do
  echo ${file}
  python -m pyprof.parse --memory true --file ${file} >${file}.dict
  python -m pyprof.prof ${file}.dict --output ${file}.csv
  python /root/guohao/pytorch_benchmark/analyzer/runtime_parse.py --input ${file}.csv --output ${output_path}
  rm ${file}.dict
done

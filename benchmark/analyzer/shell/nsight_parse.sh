#!/usr/bin/env bash
# python environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gh-torch
export PYTHONPATH=/root/guohao/pytorch_benchmark:${PYTHONPATH}

# variables
data_path=/root/guohao/pytorch_benchmark/tmp

for file in $(ls ${data_path}/*.sqlite); do
  echo ${file}
  python -m pyprof.parse --memory true --file ${file} >${file}.dict
  python -m pyprof.prof ${file}.dict --output ${file}.csv
done

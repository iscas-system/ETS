#!/usr/bin/env bash
# python environment
export PYTHONPATH=/root/guohao/pytorch_benchmark:${PYTHONPATH}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gh-torch

# get all models
scriptPath=~/guohao/pytorch_benchmark/benchmark
projectPath=~/guohao/pytorch_benchmark/
GPU=TESLAT4
dataPath=~/guohao/data/${GPU}/models
#python ${scriptPath}/save_models.py

modelsNameFile=${scriptPath}/models.txt
for line in $(cat ${modelsNameFile}); do
  modelNames=($(echo $line | tr ',' ' '))
  for name in "${modelNames[@]}"; do
    nsys profile -f true --output $name -c cudaProfilerApi --capture-range-end repeat:1000 --export sqlite --cuda-memory-usage=true python ${scriptPath}/run.py --model_name $name > ${dataPath}/log/${name}.log
#    nsys profile -f true --output resnet18 -c cudaProfilerApi --capture-range-end repeat:5 --export sqlite --cuda-memory-usage=true python run.py --model_name resnet18
    rm *.nsys-rep
    rm *.qdstrm
    cp nohup.out ${dataPath}/log/${name}.nohup.out
    echo "" > nohup.out
    echo "tar ${name} result file "
    tar -czvf ${GPU}_${name}.tar.gz ./
    mv ${GPU}_${name}.tar.gz ${dataPath}/
    rm *.sqlite
  done
done

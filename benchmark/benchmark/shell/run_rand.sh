#!/usr/bin/env bash
# python environment
export PYTHONPATH=/root/guohao/pytorch_benchmark:${PYTHONPATH}


## dell05
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate gh-torch
#GPU=2080Ti

# P4
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gh-torch
GPU=TESLAP4

### dell03
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate gh-torch
#GPU=TESLAT4

# get all models
scriptPath=/root/guohao/pytorch_benchmark/benchmark
projectPath=/root/guohao/pytorch_benchmark/
dataPath=/root/guohao/data
#python ${scriptPath}/generate_configs.py

configsFile=${scriptPath}/rand_configs.dict

start_line=0
step=500
end_line=`expr $start_line + $step`
while [ $start_line -lt 30000 ]
do
    echo "process ${start_line}:${end_line}"
    outputName='rand_'${start_line}
    echo $outputName
    nsys profile -f true --output $outputName -c cudaProfilerApi --capture-range-end repeat:1000 --export sqlite --cuda-memory-usage=true python ${scriptPath}/run_config.py --start $start_line --end $end_line > ${dataPath}/rand/log/${start_line}.log
    rm *.nsys-rep
    rm *.qdstrm
    cp nohup.out ${dataPath}/rand/log/${start_line}.nohup.out
    echo "" > nohup.out
    tar -czvf ${GPU}_${outputName}.tar.gz ./
    mv ${GPU}_${outputName}.tar.gz ${dataPath}/rand/
    rm *.sqlite
    start_line=$end_line
    end_line=`expr $start_line + $step`
done

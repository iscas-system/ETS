#!/usr/bin/env bash
GPU=${GPU}

# python environment
export PYTHONPATH=/root/pytorch_benchmark:${PYTHONPATH}

# get all models
scriptPath=/root/pytorch_benchmark/benchmark
dataPath=/root/data/${GPU}/rand


# create datapath
if [ ! -d ${dataPath}/output ]; then
  mkdir -p ${dataPath}/output
fi
if [ ! -d ${dataPath}/log ]; then
  mkdir -p ${dataPath}/log
fi
if [ ! -d ${dataPath}/data ]; then
  mkdir -p ${dataPath}/data
fi

cd ${dataPath}/output
echo ${PYTHONPATH}
echo ${PATH}
echo ${LD_LIBRARY_PATH}


configsFile=${scriptPath}/rand_configs.dict

start_line=0
step=500
end_line=`expr $start_line + $step`
while [ $start_line -lt 15000 ]
do
    echo "process ${start_line}:${end_line}"
    outputName='rand_'${start_line}
    echo $outputName
    nsys profile -f true --output $outputName -c cudaProfilerApi --capture-range-end repeat:1000 --export sqlite --cuda-memory-usage=true python3 ${scriptPath}/run_config.py --start $start_line --end $end_line --configs ${scriptPath}/rand_configs.dict > ${dataPath}/log/${start_line}.log
    rm *.nsys-rep
    rm *.qdstrm
    cp nohup.out ${dataPath}/log/${start_line}.nohup.out
    echo "" > nohup.out
    tar -czvf ${GPU}_${outputName}.tar.gz ./
    mv ${GPU}_${outputName}.tar.gz ${dataPath}/data/
    rm *.sqlite
    start_line=$end_line
    end_line=`expr $start_line + $step`
done




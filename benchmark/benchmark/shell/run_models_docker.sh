#!/usr/bin/env bash
GPU=${GPU}
# python environment
export PYTHONPATH=/root/pytorch_benchmark:${PYTHONPATH}

scriptPath=/root/pytorch_benchmark/benchmark
dataPath=/root/data/${GPU}/models

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

modelsNameFile=${scriptPath}/models.txt
for line in $(cat ${modelsNameFile}); do
  modelNames=($(echo $line | tr ',' ' '))
  for name in "${modelNames[@]}"; do
    echo "run ${name}"
    nsys profile -f true --output $name -c cudaProfilerApi --capture-range-end repeat:1000 --export sqlite --cuda-memory-usage=true python3 ${scriptPath}/run.py --model_name $name > ${dataPath}/log/${name}.log
    rm *.nsys-rep
    rm *.qdstrm
    cp nohup.out ${dataPath}/log/${name}.nohup.out
    echo "" > nohup.out
    echo "tar ${name} result file "
    tar -czvf ${GPU}_${name}.tar.gz ./
    mv ${GPU}_${name}.tar.gz ${dataPath}/data/
    rm *.sqlite
  done
done

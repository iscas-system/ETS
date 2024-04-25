### 运行

```shell
    export PYTHONPATH=/root/guohao/pytorch_benchmark:${PYTHONPATH}
    nsys profile -f true --output resnet18 -c cudaProfilerApi --capture-range-end repeat:10 --export sqlite --cuda-memory-usage=true python3 run.py --model_name resnet18
```

### 处理

```shell
python -m pyprof.parse --memory true --file rand_0.50.sqlite > rand_0.50.dict
python -m pyprof.prof ${file}.dict --output ${file}.csv
```

/root/miniconda3/envs/gh-torch/lib

### 环境准备
```
docker run -d --runtime=nvidia --gpus all --network=host -e GPU=2080Ti_CPU75 --cpus=0.75 -m 20g -v  ~/data:/root/data --name=CPU75 benchmark:v1.1
chmod +x NsightSystems-linux-public-2022.2.1.31-5fe97ab.run
./NsightSystems-linux-public-2022.2.1.31-5fe97ab.run
pip install /root/guohao/PyProf/
```

### 参数设计

cv

1. 模型数量 (20)*5种
2. 数据类型 half, float, double 3种 3. batch_size,1,2,4,8,16,32,64,128,256, 512 8种
4. image_shape, h_w, 常见图形尺寸5种

# 随机生成



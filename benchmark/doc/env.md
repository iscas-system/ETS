
## 数据采集
#### install pytorch
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch
```

#### install pyprof:
```shell
cp PyProf .
pip install .
```


#### 采集固定模型
```shell
nohup bash ../../../benchmark/run_models.sh 2>&1 &
```


#### 采集随机模型
```shell
nohup bash ../../../benchmark/run_rand.sh 2>&1 &
```


## 数据处理 
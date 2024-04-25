## ETS: Deep Learning Trainging It eration Time Prediction based on Execution Trace Sliding Window
This paper introduces ETS, a novel iteration time prediction method utilizing execution trace sliding windows. Our observation reveals that DL models exhibit a highly sequential runtime execution nature. Building upon this insight, we leverage sliding windows to extract a novel type of sequential features from the runtime execution trace. These features comprehensively capture DL framework overhead and address the diversity challenge in DL model sizes. By combining a best-practice method to train a prediction model, we achieve high accuracy and rapid convergence simultaneously.


### Contents
```shell
├── benchmark: benchmark to collect performance data
├── DLT-perf-model: implementation of ets
└── PyProf: pytorch profiler
```

### Usage
#### install pyprof to profile pytorch models

#### using benchmark collect and process performancen data on different GPUs


#### training prediction models
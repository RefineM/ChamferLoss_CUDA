# ChamferLoss_CUDA

![Python](https://img.shields.io/badge/python-3.8+-blue)
![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C)
![CUDA Version](https://img.shields.io/badge/CUDA-11.8+-76B900)
![Test Status](https://img.shields.io/badge/test_status-succeed-green)

A pytorch cuda extension for chamfer loss. 

## usage
conda virtual environment setup:
```
conda env create -f env.yaml
```
build extension: 
```
pip install .
pip install -e . # for development
```
run test script:
```
CUDA_VISIBLE_DEVICES=0 python example.py
```

## theory
*  $Q = \{[q_i]\}_{i=1}^N$ 
*  $R = \{[r_j]\}_{j=1}^M$
* Only query points (optimized) have gradients, ref points are fixed.
* top- $k$ Chamfer Loss：
  
$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \frac{1}{k} \sum_{j \in \text{TopK}_Q(q_i)} \|q_i - r_j\|_2^2
$$

* The gradient of $q_i$ is：
  
$$
\frac{\partial \mathcal{L}}{\partial q_i} 
= \frac{2}{Nk} \sum_{j \in \text{TopK}_Q(q_i)} (q_i - r_j)
$$

## TODO
[√]  top-1 chamferloss  
[ ]  top-k chamferloss


## test 
data: ref_tensor: `[5_0000,3]`, query_tensor: `[5_0000,3]`, float32, [-1,1]

|method|time(s)|memory size|result|
| --- | --- | --- | --- |
|torch.cdist()|0.09895||0.0010496085742488503 |
|ours.knn_forward()|0.00012||0.0010516365291550756|

data: ref_tensor: `[1_000_000,3]`, query_tensor: `[1_000_000,3]`, float32, [-1,1]

|method|forward time|gpu memory(MB)|result|
| --- | --- | --- | --- |
|torch.cdist()| - |OOM  |- |
|ours.forward_kernel()|0.00012|496|0.0001400598557665944 |

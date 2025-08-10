# ChamferLoss_CUDA

![Python](https://img.shields.io/badge/python-3.8+-blue)
![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C)
![CUDA Version](https://img.shields.io/badge/CUDA-11.8+-76B900)
![Test Status](https://img.shields.io/badge/test_status-❌_failed-yellow)

A pytorch cuda extension for chamfer loss. 

## usage
### for windows:
* install visual studio
* conda virtual environment setup:
```
conda env create -f env.yaml

```
* build extension: 
```
pip install .
pip install -e . # for development
```
* run test script:
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



## test
* data: ref_tensor: `[N,3]`, query_tensor: `[M,3]`

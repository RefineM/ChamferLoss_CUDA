# ChamferLoss_CUDA
A pytorch cuda extension for chamfer loss 

## usage
* dependency: `pytorch-gpu`   
* install: `pip install .`
* test script: `example.py`

## theory
* $Q = \{q_i\}_{i=1}^N$ , $R = \{r_j\}_{j=1}^M$
* Only query points (optimized) have gradients, ref points are fixed.
* top-$k$ Chamfer Loss：
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

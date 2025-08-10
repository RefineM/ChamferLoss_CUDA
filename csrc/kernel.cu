#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <float.h>

#define TILE_SIZE 256   // shared memory tile size
#define MAX_K 1          // max number of neighbors to find

// forward kernel: find top-K nearest neighbors for each query point
__global__ void knn_forward_kernel(
    const float* __restrict__ query, // [N,3]
    const float* __restrict__ ref,   // [M,3]
    int64_t* __restrict__ idx_out, // [N]
    float* __restrict__ dist_out,    // [N]
    int N,
    int M,
    int K
) {
    // create shared memory for ref tile
    // shared memory is shared amongst all threads in this block
    extern __shared__ float s_ref[]; // size = TILE_SIZE * 3 floats
    
    // get current query thread index
    int qid = blockIdx.x * blockDim.x + threadIdx.x; // 2d index -> 1d index
    if (qid >= N) return;

    // read query point
    // point_idx * bytes_per_point + point_component
    float qx = query[qid * 3 + 0];
    float qy = query[qid * 3 + 1];
    float qz = query[qid * 3 + 2];
    
    // find top-K nearest neighbors
    int Keff = (K <= M) ? K : M; // effective K = min(K, M)

    // initialize best distances and indices
    float best_dists[MAX_K];
    int best_idxs[MAX_K];
    for (int i = 0; i < Keff; ++i) {
        best_dists[i] = FLT_MAX;
        best_idxs[i] = -1;
    }

    // loop over ref in tiles
    for (int t = 0; t < M; t += TILE_SIZE) {
        // load ref tile into shared memory 
        int tile_sz = min(TILE_SIZE, M - t); // tile size
        int tile_elems = tile_sz * 3; // elements_num in the tile

        // cooperative load into shared memory
        for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
            // threadIdx.x, threadIdx.x + blockDim.x, threadIdx.x + 2*blockDim.x...
            s_ref[i] = ref[(t * 3) + i]; 
        }
        __syncthreads(); // wait for all threads to finish loading

        // scan tile
        for (int r = 0; r < tile_sz; ++r) {
            float rx = s_ref[r * 3 + 0];
            float ry = s_ref[r * 3 + 1];
            float rz = s_ref[r * 3 + 2];
            float dx = qx - rx;
            float dy = qy - ry;
            float dz = qz - rz;
            float d2 = dx*dx + dy*dy + dz*dz;

            // insertion into top-K (small K, insertion-sort style)
            if (d2 < best_dists[Keff - 1]) {
                int pos = Keff - 1;
                while (pos > 0 && d2 < best_dists[pos - 1]) {
                    best_dists[pos] = best_dists[pos - 1];
                    best_idxs[pos] = best_idxs[pos - 1];
                    pos--;
                }
                best_dists[pos] = d2;
                best_idxs[pos] = t + r;
            }
        }
        __syncthreads();
    }

    // write out the minimum distance and its index 
    // TODO: return top-K
    idx_out[qid] = (long long)best_idxs[0];
    dist_out[qid] = best_dists[0];
}


// backward kernel: grad_query = 2 * (q - r_min) * scale
__global__ void knn_backward_kernel(
    const float* __restrict__ query,    // [N,3]
    const float* __restrict__ ref,      // [M,3]
    const int64_t* __restrict__ idx,  // [N]
    float* __restrict__ grad_query,     // [N,3]
    float scale,                        // scalar = grad_output / N
    int N
) {
    // each thread computes one query point's gradient
    int qid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (qid >= N) return; 
    float qx = query[qid * 3 + 0];
    float qy = query[qid * 3 + 1];
    float qz = query[qid * 3 + 2];

    // get nearest ref index and coordinates
    int rid = (int)idx[qid];
    float rx = ref[rid * 3 + 0];
    float ry = ref[rid * 3 + 1];
    float rz = ref[rid * 3 + 2];

    // loss = 1/N * sum_i (q_i - r_i)^2 -> ∂loss/∂q_i = 2/N * (q_i - r_i)
    // distance² = (qx-rx)² + (qy-ry)² + (qz-rz)² -> ∂(distance²)/∂qx = 2(qx-rx)
    grad_query[qid * 3 + 0] = 2.0f * (qx - rx) * scale;
    grad_query[qid * 3 + 1] = 2.0f * (qy - ry) * scale;
    grad_query[qid * 3 + 2] = 2.0f * (qz - rz) * scale;
}


// C++ callable wrappers 
std::vector<torch::Tensor> knn_forward_cuda(torch::Tensor query, torch::Tensor ref, int K);
torch::Tensor knn_backward_cuda(torch::Tensor query, torch::Tensor ref, torch::Tensor idx, torch::Tensor grad_output);

// Implementations
std::vector<torch::Tensor> knn_forward_cuda(torch::Tensor query, torch::Tensor ref, int K) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA tensor");
    TORCH_CHECK(ref.is_cuda(), "ref must be CUDA tensor");
    TORCH_CHECK(query.dim() == 2 && query.size(1) == 3, "query must be [N,3]");
    TORCH_CHECK(ref.dim() == 2 && ref.size(1) == 3, "ref must be [M,3]");

    auto query_c = query.contiguous();
    auto ref_c = ref.contiguous();
    int N = (int)query_c.size(0);
    int M = (int)ref_c.size(0);

    auto idx = torch::empty({N}, query.options().dtype(torch::kLong));
    auto dist = torch::empty({N}, query.options().dtype(torch::kFloat));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const size_t shmem_bytes = TILE_SIZE * 3 * sizeof(float);

    // launch
    knn_forward_kernel<<<blocks, threads, shmem_bytes>>>(
        query_c.data_ptr<float>(),
        ref_c.data_ptr<float>(),
        idx.data_ptr<int64_t>(),
        dist.data_ptr<float>(),
        N,
        M,
        K
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("knn_forward_kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("knn_forward_kernel launch failed");
    }
    return {idx, dist};
}


torch::Tensor knn_backward_cuda(torch::Tensor query, torch::Tensor ref, torch::Tensor idx, torch::Tensor grad_output) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA tensor");
    TORCH_CHECK(ref.is_cuda(), "ref must be CUDA tensor");
    TORCH_CHECK(idx.is_cuda(), "idx must be CUDA tensor");
    TORCH_CHECK(grad_output.numel() == 1, "grad_output must be scalar");

    auto query_c = query.contiguous();
    auto ref_c = ref.contiguous();
    auto idx_c = idx.contiguous();

    int N = (int)query_c.size(0);
    float grad_out = grad_output.item<float>();
    float scale = grad_out / float(N);

    auto grad_query = torch::zeros_like(query_c);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    knn_backward_kernel<<<blocks, threads>>>(
        query_c.data_ptr<float>(),
        ref_c.data_ptr<float>(),
        idx_c.data_ptr<int64_t>(),
        grad_query.data_ptr<float>(),
        scale,
        N
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("knn_backward_kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("knn_backward_kernel launch failed");
    }
    return grad_query;
}

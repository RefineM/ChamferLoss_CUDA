#include <torch/extension.h>
#include <vector>

// declare the cuda-backend functions implemented in chamfer_kernel.cu
std::vector<torch::Tensor> knn_forward_cuda(torch::Tensor query, torch::Tensor ref, int K);
torch::Tensor knn_backward_cuda(torch::Tensor query, torch::Tensor ref, torch::Tensor idx, torch::Tensor grad_output);

// python-callable wrappers 
std::vector<torch::Tensor> knn_forward(torch::Tensor query, torch::Tensor ref, int K) {
    return knn_forward_cuda(query, ref, K);
}
torch::Tensor knn_backward(torch::Tensor query, torch::Tensor ref, torch::Tensor idx, torch::Tensor grad_output) {
    return knn_backward_cuda(query, ref, idx, grad_output);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_forward", &knn_forward, "KNN forward (CUDA)");
    m.def("knn_backward", &knn_backward, "KNN backward (CUDA)");
}

import torch
from torch import nn
from torch.autograd import Function
import chamferloss_cuda  
import time

class ChamferFunction(Function):
    """
    Chamfer loss function
    * custom autograd function for chamfer loss
    """
    @staticmethod
    def forward(ctx, query, ref, K=1):
        # query: [N,3], ref: [M,3]
        assert query.device.type == 'cuda' and ref.device.type == 'cuda'
        idx, dist = chamferloss_cuda.knn_forward(query.contiguous(), ref.contiguous(), int(K))
        print(idx)
        # idx: LongTensor [N], dist: FloatTensor [N] (squared dist)
        ctx.save_for_backward(query, ref, idx)
        return dist.mean()  # scalar

    @staticmethod
    def backward(ctx, grad_output):
        query, ref, idx = ctx.saved_tensors
        # grad_output is a scalar tensor; pass into cuda wrapper, which expects scalar
        grad_query = chamferloss_cuda.knn_backward(query.contiguous(), ref.contiguous(), idx.contiguous(), grad_output)
        # we only had query requiring grad; no grad for ref in this design.
        return grad_query, None, None


class ChamferLoss(nn.Module):
    def __init__(self, ref_points: torch.Tensor, K=1):
        super().__init__()
        assert ref_points.device.type == 'cuda'
        self.register_buffer('ref', ref_points)
        self.K = K

    def forward(self, query):
        return ChamferFunction.apply(query, self.ref, self.K)


if __name__ == '__main__':
    
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    print(torch.__version__)       
    print(torch.version.cuda)       

    torch.manual_seed(42)
    N = 100_0000   # query count for test 
    M = 100_0000   # ref count
    ref = torch.rand(M,3, device='cuda', dtype=torch.float32)
    query = torch.rand(N,3, device='cuda', dtype=torch.float32, requires_grad=True)

    loss_fn = ChamferLoss(ref, K=1)

    torch.cuda.synchronize()
    t0 = time.time()
    loss = loss_fn(query)    # forward
    torch.cuda.synchronize()
    fwd = time.time() - t0

    t0 = time.time()
    loss.backward()          # backward
    torch.cuda.synchronize()
    bwd = time.time() - t0

    print("N, M:", N, M)
    print("loss:", loss.item())
    print("forward:", fwd, "s, backward:", bwd, "s")

import torch
from torch import nn
from torch.autograd import Function
import chamferloss_cuda  
import time

class ChamferFunction(Function):
    """
    custom autograd function for chamfer loss
    """
    @staticmethod
    def forward(ctx, 
                query: torch.Tensor, 
                ref: torch.Tensor, 
                K:int = 1
        ):
        # query: [N,3], ref: [M,3]
        assert query.device.type == 'cuda' and ref.device.type == 'cuda'
        time01 = time.time()
        idx, dist = chamferloss_cuda.knn_forward(query.contiguous(), ref.contiguous(), int(K))
        print(f"fwd time: {time.time() - time01}")
        # idx: IntTensor [N], dist: FloatTensor [N] (squared dist)
        ctx.save_for_backward(query, ref, idx)
        return dist.mean()

    @staticmethod
    def backward(ctx, grad_output):
        query, ref, idx = ctx.saved_tensors
        # grad_output is a scalar tensor. pass it into cuda wrapper
        time01 = time.time()
        grad_query = chamferloss_cuda.knn_backward(query.contiguous(), ref.contiguous(), idx.contiguous(), grad_output)
        print(f"bwd time: {time.time() - time01}")
        # only query requires grad
        return grad_query, None, None


class ChamferLoss(nn.Module):
    def __init__(self, 
                 ref_points: torch.Tensor, 
                 K:int = 1):
        super().__init__()
        assert ref_points.device.type == 'cuda'
        self.register_buffer('ref', ref_points)
        self.K = K

    def forward(self, query):
        return ChamferFunction.apply(query, self.ref, self.K)


if __name__ == '__main__':
    
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name(torch.cuda.current_device())}")
      
    torch.manual_seed(0)
    N = 100_0000   # query count for test 
    M = 100_0000   # ref count
    ref = torch.rand(M,3, device='cuda', dtype=torch.float32) * 2 - 1
    query = torch.rand(N,3, device='cuda', dtype=torch.float32, requires_grad=True) * 2 - 1

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

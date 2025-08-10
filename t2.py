import torch
from sklearn.neighbors import KDTree
import time
from example import ChamferLoss
import chamferloss_cuda

class OptimizedDiffChamfer(torch.nn.Module):
    def __init__(self, 
                 ref_points: torch.Tensor, 
                 nearest_num=1, 
                 leaf_size=1, 
                 block_size=1000000):
        """
        可微分的单向Chamfer距离
        Args:
            ref_points (torch.Tensor): [N, 3] 参考点
            k (int): 最近邻数量
            leaf_size (int): KDTree的叶子大小
            block_size (int): 分块大小
        """
        super().__init__()
        self.ref = ref_points.detach().cpu().numpy()
        self.tree = KDTree(self.ref, leaf_size=leaf_size)
        self.nearest_num = nearest_num
        self.block_size = block_size
        self.register_buffer('ref_gpu', ref_points.clone())
        print(f"OptimizedDiffChamfer: \n"
                f"ref_points_num={ref_points.shape[0]} " 
                f"nearest_num={nearest_num} "
                f"leaf_size={leaf_size} "
                f"block_size={block_size}"
            )
        
    def forward(self, 
                query_points: torch.Tensor):
        """
        Args:
            query_points (torch.Tensor): [B, 3] 查询点
        """
        N = query_points.shape[0]
        device = query_points.device
        
        # 分离计算图获取最近邻索引 (分块并行)
        with torch.no_grad():
            # 查询最近邻索引时不传播梯度
            query_np = query_points.detach().cpu().numpy()
            nn_indices_list = []
            # 分块 防止oom
            for i in range(0, N, self.block_size):
                block = query_np[i:i+self.block_size]
                _, nn_indices = self.tree.query(block, k=self.nearest_num)
                nn_indices_list.append(torch.from_numpy(nn_indices))
            nn_indices = torch.cat(nn_indices_list).to(device)
        
        # 分块计算可微分距离 (减少显存占用)
        total_loss = 0.0
        for i in range(0, N, self.block_size):
            # 当前块数据
            block_points = query_points[i:i+self.block_size]
            block_indices = nn_indices[i:i+self.block_size]
            # 获取候选点
            candidates = self.ref_gpu[block_indices]  # [B, k, 3]
            # 可微分距离计算
            diff = block_points.unsqueeze(1) - candidates  # [B, k, 3]
            dists = torch.norm(diff, dim=2, p=2)  # [B, k]
            min_dists, _ = torch.min(dists, dim=1)  # [B]
            total_loss += min_dists.sum()
        return total_loss * total_loss / N / N, nn_indices

def chamfer_loss_torchcdist(query, ref, k=1):
    # query: [N, 3], ref: [M, 3]
    dist_matrix = torch.cdist(query, ref, p=2)  # [N, M]
    # 取每个query点距离最近k个ref点的距离
    topk_dists, idx = torch.topk(dist_matrix, k, largest=False)
    # 取均值作为loss
    loss = topk_dists.mean()
    return loss*loss, idx

def idx_diff(idx_torch, idx_cuda):
    print("索引是否完全一致:", torch.equal(idx_torch, idx_cuda))
    # 如果不完全一致，可以统计不一致的数量和位置
    diff = (idx_torch != idx_cuda)
    print("不一致索引数量:", diff.sum().item())
    print("不一致索引的位置示例:", diff.nonzero())



# 创建测试数据
torch.manual_seed(0)
ref = torch.rand(20000, 3, device='cuda', requires_grad=False) # 参考点不计算梯度
query = torch.rand(20000, 3, device='cuda', requires_grad=True) # 查询点计算梯度

# 测试
start = time.time()
method = OptimizedDiffChamfer(ref, nearest_num=1).cuda()
print(f"tree_time={time.time() - start}")

# 时间测试
start = time.time()
loss1, idx1 = method(query)
fwd_time = (time.time() - start)
loss1.backward()
bwd_time = (time.time() - start) - fwd_time
print(f"forward_time={fwd_time}")
print(f"backward_time={bwd_time}")

# 检查梯度
print("loss_value=", loss1.item())
# print("query_points has grad?", query.grad is not None)
# print("norm(query.grad):", torch.norm(query.grad).item())


loss2, idx2 = chamfer_loss_torchcdist(query, ref, k=1)
loss2.backward()
print("Loss:", loss2.item())
print("Grad norm:", query.grad.norm().item())


loss_fn = ChamferLoss(ref, K=1)
torch.cuda.synchronize()
t0 = time.time()
loss3 = loss_fn(query)    # forward
torch.cuda.synchronize()
fwd = time.time() - t0

t0 = time.time()
loss3.backward()          # backward
torch.cuda.synchronize()
bwd = time.time() - t0

print("loss:", loss3.item())
print("forward:", fwd, "s, backward:", bwd, "s")

print(idx1.squeeze())
idx2, dist = chamferloss_cuda.knn_forward(query.contiguous(), ref.contiguous(), 1)
print(idx2)
idx_diff(idx1.squeeze(), idx2)
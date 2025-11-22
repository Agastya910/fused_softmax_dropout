# Bad naive implementation (example only) — will NOT be accepted by tests.
import torch
def fused_softmax_dropout(x, p, seed=0):
    # naive Python (no Triton), likely to be too slow or mismatched in device placement
    torch.manual_seed(seed)
    x = x.cpu()
    shifted = x - x.max(dim=-1, keepdim=True).values
    exp_x = shifted.exp()
    soft = exp_x / exp_x.sum(dim=-1, keepdim=True)
    mask = (torch.rand_like(soft) > p).float()
    return (soft * mask / (1 - p)).to(device='cuda')

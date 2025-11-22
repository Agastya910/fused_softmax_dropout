import torch
import torch.nn.functional as F

def fused_softmax_dropout_ref(x: torch.Tensor, p: float, seed: int = 0) -> torch.Tensor:
    """
    Pure PyTorch reference implementation (oracle).
    This is the ground truth used for numeric comparison.
    """
    torch.manual_seed(seed)

    # Numerically stable softmax
    x_max = x.max(dim=-1, keepdim=True).values
    shifted = x - x_max
    exp_x = shifted.exp()
    softmax_out = exp_x / exp_x.sum(dim=-1, keepdim=True)

    # Dropout (seeded for reproducibility)
    mask = (torch.rand_like(softmax_out) > p).float()
    return softmax_out * mask / (1 - p)

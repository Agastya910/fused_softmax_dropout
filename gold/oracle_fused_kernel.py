import torch
from fused_softmax_dropout.utils.reference import fused_softmax_dropout_ref

def fused_softmax_dropout(x: torch.Tensor, p: float, seed: int = 0) -> torch.Tensor:
    """
    Oracle implementation for the grader.
    Computes the exact reference output using PyTorch.
    """
    if not x.is_cuda:
        x = x.cuda()
    return fused_softmax_dropout_ref(x, p, seed)

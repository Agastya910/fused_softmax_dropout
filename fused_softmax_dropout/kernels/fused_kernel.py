import torch

def fused_softmax_dropout(x: torch.Tensor, p: float, seed: float) -> torch.Tensor:
    """
    Implement a fused softmax + dropout kernel using Triton.
    Args:
        x: Input tensor (2D). Warning: Values may be large.
        p: Dropout probability.
        seed: Float seed. Scale by 10,000 before int conversion.
    """
    raise NotImplementedError("Triton kernel not implemented")

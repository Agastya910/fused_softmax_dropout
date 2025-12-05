import torch

def fused_softmax_dropout(x: torch.Tensor, p: float, seed: float = 0.0) -> torch.Tensor:
    """
    Implement a fused softmax + dropout kernel using Triton.
    
    Args:
        x: Input tensor (2D) on CUDA
        p: Dropout probability (0 <= p < 1)
        seed: Float seed for deterministic dropout
    
    Returns:
        Output tensor with fused softmax + dropout applied
    
    Notes:
        - Must handle numerical stability for large input values
        - Seed is a float and must be properly converted for RNG
    """
    raise NotImplementedError("Triton kernel not implemented")

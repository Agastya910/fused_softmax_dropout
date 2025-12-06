import torch

def fused_softmax_dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """
    Implement a fused softmax + dropout kernel using Triton.
    
    Args:
        x: Input tensor (2D) on CUDA.
           Warning: Values may be large (requires numerical stability).
        p: Dropout probability (0 <= p < 1)
        seed: Integer seed for deterministic dropout.
              
    Returns:
        Output tensor with fused softmax + dropout applied
    """
    raise NotImplementedError("Triton kernel not implemented")

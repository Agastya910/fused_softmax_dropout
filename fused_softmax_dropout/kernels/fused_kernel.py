import torch

def fused_softmax_dropout(x: torch.Tensor, p: float, seed: float) -> torch.Tensor:
    """
    Implement a fused softmax + dropout kernel using Triton.
    
    Args:
        x (torch.Tensor): 2D Input tensor on CUDA. 
                          Note: Values can be large/unnormalized.
                          Note: Stride may not be contiguous.
        p (float): Dropout probability (0.0 to 1.0).
        seed (float): High-precision seed for deterministic dropout.
                      Ensure the full precision of the float is used 
                      to generate distinct masks for small seed increments.
    
    Returns:
        torch.Tensor: Output tensor with softmax and dropout applied.
    """
    raise NotImplementedError("Triton kernel not implemented")

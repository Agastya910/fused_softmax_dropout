import torch

def fused_softmax_dropout(x: torch.Tensor, p: float, seed: float) -> torch.Tensor:
    """
    Implement a fused softmax + dropout kernel using Triton.
    
    Args:
        x (torch.Tensor): 2D Input tensor on CUDA. 
                          Note: Values can be large (requires numerical stability).
                          Note: Stride may not be contiguous.
        p (float): Dropout probability (0.0 to 1.0).
        seed (float): Seed for deterministic dropout.
                      HINT: The hardware RNG requires integers. You should scale 
                      this float seed by a LARGE constant (e.g. sufficient to 
                      distinguish 1.2345 from 1.2346) before casting to int.
    
    Returns:
        torch.Tensor: Output tensor with softmax and dropout applied.
    """
    raise NotImplementedError("Triton kernel not implemented")

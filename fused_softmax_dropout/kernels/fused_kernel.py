import torch

def fused_softmax_dropout(x: torch.Tensor, p: float, seed: float) -> torch.Tensor:
    """
    Implement a fused softmax + dropout kernel using Triton.
    
    Args:
        x: Input tensor (2D) on CUDA.
        p: Dropout probability (0 <= p < 1)
        seed: Float seed.
        
    CRITICAL IMPLEMENTATION REQUIREMENTS:
    1. Numerical Stability: Inputs may be large (e.g., 10,000). You must handle potential 
       overflows in the softmax calculation.
       
    2. Seed Sensitivity: The seed is provided as a float. You must ensure that the 
       generated dropout mask changes even for small increments in the seed value 
       (e.g., changes of 0.0001). Simply casting to int is insufficient.
       
    3. Memory Layout: Do not assume the input is contiguous. You must handle the 
       strides correctly.
    """
    raise NotImplementedError("Triton kernel not implemented")

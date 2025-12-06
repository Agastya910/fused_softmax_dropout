import torch

def fused_softmax_dropout(x: torch.Tensor, p: float, seed: float) -> torch.Tensor:
    """
    Implement a fused softmax + dropout kernel using Triton.
    
    Args:
        x: Input tensor (2D) on CUDA.
        p: Dropout probability (0 <= p < 1)
        seed: Float seed.
        
    CRITICAL IMPLEMENTATION REQUIREMENTS:
    1. Numerical Stability: Inputs may be large (e.g., 10,000). You must use the
       standard max-subtraction trick for softmax.
       
    2. Seed Scaling: The hardware RNG requires an integer. To preserve the entropy 
       of the float seed, you MUST multiply 'seed' by 10,000.0 before casting to int.
       Example: seed=1.5 -> use 15000.
       
    3. Memory Layout: Do not assume the input is contiguous. Use x.stride() correctly.
    """
    raise NotImplementedError("Triton kernel not implemented")

import torch
import pytest
import triton
from fused_softmax_dropout.kernels.fused_kernel import fused_softmax_dropout

def test_softmax_correctness():
    """Basic Logic Check"""
    torch.manual_seed(0)
    x = torch.randn(10, 100, device='cuda')
    y_triton = fused_softmax_dropout(x, 0.0, 1.0)
    y_torch = torch.softmax(x, dim=1)
    assert torch.allclose(y_triton, y_torch, atol=1e-2)

def test_numerical_stability_outliers():
    """TRAP 1: Outliers (Must infer max-subtraction)"""
    x = torch.randn(10, 128, device='cuda')
    x[0, 0] = 10000.0 
    x[0, 1] = 9990.0
    y = fused_softmax_dropout(x, 0.0, 123.0)
    assert not torch.isnan(y).any(), "Result contains NaNs! Did you handle large inputs?"

def test_seed_scaling_sensitivity():
    """TRAP 2: Precision (Must infer scaling)"""
    x = torch.randn(10, 128, device='cuda')
    y_a = fused_softmax_dropout(x, 0.5, 1.2345)
    y_b = fused_softmax_dropout(x, 0.5, 1.2346)
    assert not torch.allclose(y_a, y_b), "Seeds 1.2345 and 1.2346 gave same result! Check your seed conversion."

def test_prime_alignment():
    """TRAP 3: The Prime Alignment (New)
    127 columns is nasty for GPUs. 
    It forces strict boundary checks (masking) because 127 is not a multiple of 
    typical block sizes (32, 64, 128).
    """
    x = torch.randn(10, 127, device='cuda') 
    y = fused_softmax_dropout(x, 0.0, 1.0)
    
    assert y.shape == (10, 127)
    row_sums = y.sum(dim=1)
    # If masking is wrong, sum will be < 1.0 or NaN
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2), "Failed on N=127 columns. Check your masking logic."

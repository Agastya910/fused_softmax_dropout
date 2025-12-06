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
    """TRAP 1: Outliers"""
    x = torch.randn(10, 128, device='cuda')
    x[0, 0] = 10000.0 
    x[0, 1] = 9990.0
    y = fused_softmax_dropout(x, 0.0, 123.0)
    assert not torch.isnan(y).any(), "Result contains NaNs! Subtract max(x)."

def test_seed_scaling_sensitivity():
    """TRAP 2: Precision"""
    x = torch.randn(10, 128, device='cuda')
    # If they use int(seed), these are both 1 -> Fail.
    # If they use int(seed * scale), these are different -> Pass.
    y_a = fused_softmax_dropout(x, 0.5, 1.2345)
    y_b = fused_softmax_dropout(x, 0.5, 1.2346)
    assert not torch.allclose(y_a, y_b), "Seeds 1.2345 and 1.2346 gave same result! Did you scale the seed?"

def test_strided_inputs():
    """TRAP 3: Non-Contiguous Memory"""
    full = torch.randn(20, 200, device='cuda')
    x = full[:, :100] # Slice creates non-contiguous stride
    
    y = fused_softmax_dropout(x, 0.0, 1.0)
    
    assert y.shape == (20, 100)
    row_sums = y.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)

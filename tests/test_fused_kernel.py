import torch
import pytest
import triton
from fused_softmax_dropout.kernels.fused_kernel import fused_softmax_dropout

def test_softmax_correctness():
    """1. Basic Functionality Check"""
    torch.manual_seed(0)
    x = torch.randn(10, 100, device='cuda')
    p = 0.0 
    seed = 123
    y_triton = fused_softmax_dropout(x, p, seed)
    y_torch = torch.softmax(x, dim=1)
    assert torch.allclose(y_triton, y_torch, atol=1e-2)

def test_numerical_stability_outliers():
    """
    2. THE REMAINING TRAP (Stability)
    Feed massive inputs (10,000). 
    Naive exp(10000) -> Inf. 
    Safe exp(10000 - 10000) -> 1.
    """
    x = torch.randn(10, 128, device='cuda')
    x[0, 0] = 10000.0  # Massive outlier
    x[0, 1] = 9990.0
    
    y = fused_softmax_dropout(x, 0.0, 123)
    
    # Fail if we see NaNs (means they used naive exp)
    assert not torch.isnan(y).any(), "Result contains NaNs! Did you subtract max(x) before exp()?"
    
    # Verify rows still sum to 1
    row_sums = y.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)

def test_determinism():
    """3. Determinism Check"""
    x = torch.randn(10, 128, device='cuda')
    y1 = fused_softmax_dropout(x, 0.5, 42)
    y2 = fused_softmax_dropout(x, 0.5, 42)
    assert torch.allclose(y1, y2), "Non-deterministic output for same seed!"

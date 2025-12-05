import torch
import pytest
import triton
from fused_softmax_dropout.kernels.fused_kernel import fused_softmax_dropout

SEED_SCALE_FACTOR = 10000.0

def reference_implementation(x, p, seed_float):
    """PyTorch reference that matches the expected Triton behavior."""
    real_seed = int(seed_float * SEED_SCALE_FACTOR)
    softmax = torch.softmax(x, dim=-1)
    return softmax

def test_softmax_correctness():
    """Verify softmax calculation matches PyTorch (Basic functionality)"""
    torch.manual_seed(0)
    x = torch.randn(10, 100, device='cuda')
    p = 0.0  # No dropout to test softmax pure
    seed = 1.0
    y_triton = fused_softmax_dropout(x, p, seed)
    y_torch = torch.softmax(x, dim=1)
    assert torch.allclose(y_triton, y_torch, atol=1e-2)

def test_numerical_stability_outliers():
    """
    TRAP 1: Numerical Stability
    Inputs with values like 10,000 will result in exp(10000) = inf
    unless max-subtraction is used.
    """
    torch.manual_seed(0)
    x = torch.randn(10, 128, device='cuda')
    x[0, 0] = 10000.0  # Outlier!
    x[0, 1] = 9990.0
    p = 0.0
    seed = 1.0
    y_triton = fused_softmax_dropout(x, p, seed)
    assert not torch.isnan(y_triton).any(), "Kernel produced NaNs on large inputs - implementation lacks numerical stability (max subtraction)."
    row_sums = y_triton.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)

def test_seed_scaling_sensitivity():
    """
    TRAP 2: Float Seed Scaling
    We verify that seed=1.0 and seed=1.0001 produce DIFFERENT results.
    If the user implemented `int(seed)`, these would be identical.
    If the user implements `int(seed * 10000)`, they will be different.
    """
    x = torch.randn(10, 128, device='cuda')
    p = 0.5
    seed_a = 12.3456
    seed_b = 12.3457  # Delta is 0.0001
    y_a = fused_softmax_dropout(x, p, seed_a)
    y_b = fused_softmax_dropout(x, p, seed_b)
    are_identical = torch.allclose(y_a, y_b)
    assert not are_identical, "Close float seeds produced identical results! Did you forget to scale the seed before casting to int? (e.g. seed * 10000)"

def test_determinism():
    """Verify that same float seed gives same result"""
    x = torch.randn(10, 128, device='cuda')
    p = 0.5
    seed = 42.5
    y1 = fused_softmax_dropout(x, p, seed)
    y2 = fused_softmax_dropout(x, p, seed)
    assert torch.allclose(y1, y2), "Kernel is not deterministic for the same seed!"

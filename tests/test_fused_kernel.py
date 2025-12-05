import torch
import pytest
import triton
from fused_softmax_dropout.kernels.fused_kernel import fused_softmax_dropout

SEED_SCALE_FACTOR = 10000.0

def test_softmax_correctness():
    torch.manual_seed(0)
    x = torch.randn(10, 100, device='cuda')
    y_triton = fused_softmax_dropout(x, 0.0, 1.0)
    y_torch = torch.softmax(x, dim=1)
    assert torch.allclose(y_triton, y_torch, atol=1e-2)

def test_numerical_stability_outliers():
    x = torch.randn(10, 128, device='cuda')
    x[0, 0] = 10000.0
    x[0, 1] = 9990.0
    y = fused_softmax_dropout(x, 0.0, 1.0)
    assert not torch.isnan(y).any(), "Result contains NaNs! Did you subtract max(x)?"

def test_seed_scaling_sensitivity():
    x = torch.randn(10, 128, device='cuda')
    y_a = fused_softmax_dropout(x, 0.5, 12.3456)
    y_b = fused_softmax_dropout(x, 0.5, 12.3457)
    assert not torch.allclose(y_a, y_b), "Seeds 12.3456 and 12.3457 gave identical results! Did you forget to scale the seed?"

def test_determinism():
    x = torch.randn(10, 128, device='cuda')
    y1 = fused_softmax_dropout(x, 0.5, 42.5)
    y2 = fused_softmax_dropout(x, 0.5, 42.5)
    assert torch.allclose(y1, y2), "Non-deterministic output!"

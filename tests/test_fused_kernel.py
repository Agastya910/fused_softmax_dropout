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
    """TRAP 2: Precision (The Main Filter)"""
    x = torch.randn(10, 128, device='cuda')
    # If they use int(seed), these become identical -> Fail.
    # If they scale by 100, these become identical -> Fail.
    # They must scale by at least ~10,000 to pass.
    y_a = fused_softmax_dropout(x, 0.5, 1.2345)
    y_b = fused_softmax_dropout(x, 0.5, 1.2346)
    assert not torch.allclose(y_a, y_b), "Dropout patterns are identical for close seeds! Precision loss detected."

def test_odd_dimension_masking():
    """TRAP 3: Odd Dimensions (The Masking Filter)
    Many LLMs assume block_size is always a power of 2 and inputs are perfectly aligned.
    This test uses a prime number dimension (317) to force correct masking.
    """
    torch.manual_seed(0)
    # 317 is prime, ensuring it doesn't align with standard block sizes (128, 256)
    x = torch.randn(10, 317, device='cuda') 
    y_triton = fused_softmax_dropout(x, 0.0, 1.0)
    y_torch = torch.softmax(x, dim=1)
    
    # If they forgot 'mask=offset<n_cols' in load/store, this will likely have garbage values at the end
    # or crash with an Illegal Memory Access before getting here.
    assert torch.allclose(y_triton, y_torch, atol=1e-2), "Kernel failed on odd-sized dimension (317). Did you handle masking?"

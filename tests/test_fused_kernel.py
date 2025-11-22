import torch
import torch.nn.functional as F
from fused_softmax_dropout.utils.reference import fused_softmax_dropout_ref
from fused_softmax_dropout.kernels.fused_kernel import fused_softmax_dropout

def test_correctness_small():
    x = torch.randn(4, 128, device='cuda')
    p = 0.1
    out = fused_softmax_dropout(x, p, seed=42)
    ref = fused_softmax_dropout_ref(x, p, 42)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)

def test_correctness_large():
    x = torch.randn(64, 2048, device='cuda')
    p = 0.2
    out = fused_softmax_dropout(x, p, seed=123)
    ref = fused_softmax_dropout_ref(x, p, 123)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)

def test_numerical_stability():
    x = torch.randn(8, 512, device='cuda') * 100
    out = fused_softmax_dropout(x, 0.0, seed=7)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)

def test_reproducibility():
    x = torch.randn(16, 256, device='cuda')
    out1 = fused_softmax_dropout(x, 0.3, seed=999)
    out2 = fused_softmax_dropout(x, 0.3, seed=999)
    assert torch.equal(out1, out2)

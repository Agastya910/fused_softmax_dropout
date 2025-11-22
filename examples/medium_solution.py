# Medium solution using PyTorch (for reference only)
import torch
from fused_softmax_dropout.utils.reference import fused_softmax_dropout_ref
def fused_softmax_dropout(x, p, seed=0):
    return fused_softmax_dropout_ref(x, p, seed)

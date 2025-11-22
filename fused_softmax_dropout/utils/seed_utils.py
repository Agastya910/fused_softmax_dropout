import torch

def make_seed(seed: int):
    torch.manual_seed(seed)
    return seed

import torch
from torch import nn
import torch.nn.functional as F


if __name__ == '__main__':
    device = 'cuda:0'
    n, num_class, c = 400000, 19, 32
    q = torch.randn((1, n, c), device=device, requires_grad=True)
    k = torch.randn((num_class, n, c), device=device, requires_grad=True)
    v = torch.randn((num_class, n, c), device=device, requires_grad=True)
    
    msa = nn.MultiheadAttention(c, num_heads=1, device=device)
    
    res = msa(q, k, v)
    
    print(torch.cuda.max_memory_reserved(device=device))

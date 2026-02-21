"""
Implementation of corase layer
"""
import torch
import torch.nn as nn

class PerChannelDenseEinsum(nn.Module):
    def __init__(self, dim_k, dense_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_k, dense_dim, dense_dim) * 0.02)
        self.bias = nn.Parameter(torch.randn(dim_k, dense_dim)) if bias else None

    def forward(self, x):
        out = torch.einsum('bkd,kdf->bkf', x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    
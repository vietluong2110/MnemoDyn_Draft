"""
Implementation of corase layer
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import List, Tuple, Optional
import torch.utils.checkpoint as checkpoint
import torch.utils.benchmark as benchmark

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
    
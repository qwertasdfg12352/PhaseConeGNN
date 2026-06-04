# coding=utf-8
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ContextGraphNorm(nn.Module):
    """GraphNorm with graph-level context conditioned affine parameters."""

    def __init__(self, in_channels: int, context_dim: int = 8, hidden_dim: int | None = None, eps: float = 1e-5):
        super().__init__()
        hidden_dim = hidden_dim or max(16, in_channels // 4)
        self.in_channels = in_channels
        self.context_dim = context_dim
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(in_channels))
        self.bias = nn.Parameter(torch.empty(in_channels))
        self.mean_scale = nn.Parameter(torch.empty(in_channels))
        self.context_affine = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * in_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.mean_scale)
        for module in self.context_affine:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        last = self.context_affine[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: Tensor, graph_context: Tensor) -> Tensor:
        if graph_context.dim() == 1:
            graph_context = graph_context.unsqueeze(0)

        mean = x.mean(dim=0, keepdim=True)
        out = x - mean * self.mean_scale
        var = out.pow(2).mean(dim=0, keepdim=True)
        out = out / torch.sqrt(var + self.eps)

        gamma_delta, beta_delta = self.context_affine(graph_context).chunk(2, dim=-1)
        gamma = self.weight.unsqueeze(0) + gamma_delta
        beta = self.bias.unsqueeze(0) + beta_delta
        return out * gamma + beta

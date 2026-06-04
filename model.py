# coding=utf-8
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from .layers import (
        PhaseStateInitializer,
        PhaseTransitionBlock,
        GatedResidual,
        LogicAwareSPPHead,
        CircuitLayerMoE,
    )
    from .normalization import ContextGraphNorm
    from .utils import build_graph_context, create_spectral_features, split_signed_edges
except ImportError:
    from layers import (
        PhaseStateInitializer,
        PhaseTransitionBlock,
        GatedResidual,
        LogicAwareSPPHead,
        CircuitLayerMoE,
    )
    from normalization import ContextGraphNorm
    from utils import build_graph_context, create_spectral_features, split_signed_edges


class PhaseConeGNN(nn.Module):
    """
    Minimal PhaseConeGNN.

    Components:
    - Phase-state transition message passing.
    - Context-conditioned GraphNorm.
    - Graph-context gated residual propagation.
    - Logic-aware probability readout.
    """

    def __init__(
        self,
        args,
        node_num: int,
        device: torch.device,
        in_dim: int = 64,
        out_dim: int = 64,
        layer_num: int = 2,
        lamb: float = 5,
        norm_emb: bool = False,
        context_dim: int = 8,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if out_dim % 2 != 0:
            raise ValueError("PhaseConeGNN requires out_dim to be even.")

        self.args = args
        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.state_dim = out_dim // 2
        self.lamb = lamb
        self.device = device
        self.norm_emb = norm_emb
        self.context_dim = context_dim
        self.layer_num = layer_num
        use_backward_arg = getattr(args, "use_backward", -1) if args is not None else -1
        if use_backward_arg < 0:
            self.use_backward = getattr(args, "task_type", "prob") != "prob"
        else:
            self.use_backward = bool(use_backward_arg)
        self.use_layer_moe = bool(getattr(args, "use_layer_moe", 1)) if args is not None else True

        self.pos_edge_index = None
        self.neg_edge_index = None
        self.x = None
        self.graph_context = None
        self.last_moe_weights = None

        self.phase_init = PhaseStateInitializer(in_dim, self.state_dim, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                PhaseTransitionBlock(self.state_dim, context_dim=context_dim, dropout=dropout)
                for _ in range(layer_num)
            ]
        )
        self.norms = nn.ModuleList(
            [ContextGraphNorm(out_dim, context_dim=context_dim) for _ in range(layer_num)]
        )
        self.residuals = nn.ModuleList(
            [GatedResidual(out_dim, context_dim=context_dim) for _ in range(layer_num)]
        )
        self.layer_moe = CircuitLayerMoE(
            out_dim,
            num_experts=layer_num + 1,
            context_dim=context_dim,
            hidden_dim=out_dim,
            dropout=dropout,
        )

        self.output_proj = nn.Linear(out_dim, out_dim)
        self.readout_prob = LogicAwareSPPHead(
            out_dim,
            context_dim=context_dim,
            hidden_dim=out_dim,
            dropout=dropout,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.phase_init.reset_parameters()
        for block, norm, residual in zip(self.blocks, self.norms, self.residuals):
            block.reset_parameters()
            norm.reset_parameters()
            residual.reset_parameters()
        self.layer_moe.reset_parameters()
        self.output_proj.reset_parameters()
        self.readout_prob.reset_parameters()

    def get_x_edge_index(self, init_emb: Tensor | None, edge_index_s: Tensor):
        self.pos_edge_index, self.neg_edge_index = split_signed_edges(edge_index_s)
        if init_emb is not None:
            node_num = int(init_emb.size(0))
        elif edge_index_s.numel() > 0:
            node_num = int(edge_index_s[:, :2].max().item()) + 1
        else:
            node_num = self.node_num
        if init_emb is None:
            init_emb = create_spectral_features(
                pos_edge_index=self.pos_edge_index,
                neg_edge_index=self.neg_edge_index,
                node_num=node_num,
                dim=self.in_dim,
            ).to(self.device)
        self.x = init_emb
        self.node_num = int(init_emb.size(0))
        self.graph_context = build_graph_context(edge_index_s, self.node_num)

    def forward(self, init_emb: Tensor | None, edge_index_s: Tensor) -> Tuple[Tensor, Tensor]:
        self.get_x_edge_index(init_emb, edge_index_s)

        z = self.phase_init(self.x)
        layer_states = [z]
        denom = max(self.layer_num - 1, 1)
        for layer_idx, (block, norm, residual) in enumerate(zip(self.blocks, self.norms, self.residuals)):
            proposal = block(
                z,
                self.pos_edge_index,
                self.neg_edge_index,
                self.graph_context,
                use_backward=self.use_backward,
            )
            proposal = norm(proposal, self.graph_context)
            proposal = torch.tanh(proposal)
            z = residual(z, proposal, self.graph_context, layer_ratio=layer_idx / denom)
            layer_states.append(z)

        if self.use_layer_moe:
            z, moe_weights = self.layer_moe(layer_states, self.graph_context)
            self.last_moe_weights = moe_weights.detach()
        else:
            self.last_moe_weights = None
        z = torch.tanh(self.output_proj(z))
        if self.norm_emb:
            z = F.normalize(z, p=2, dim=-1)

        prob = self.readout_prob(z, edge_index_s, self.graph_context)
        return z, prob

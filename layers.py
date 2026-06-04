# coding=utf-8
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _aggregate_mean(messages: Tensor, dst: Tensor, num_nodes: int) -> Tensor:
    out = messages.new_zeros((num_nodes, messages.size(-1)))
    if messages.numel() == 0:
        return out
    out.index_add_(0, dst, messages)
    count = messages.new_zeros((num_nodes, 1))
    count.index_add_(0, dst, messages.new_ones((messages.size(0), 1)))
    return out / count.clamp_min(1.0)


def signed_logic_prior(prob: Tensor, edge_index_s: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Build a differentiable signed-fanin SPP prior from predicted probabilities."""
    num_nodes = prob.size(0)
    prior = prob
    mask = torch.zeros((num_nodes, 1), dtype=torch.bool, device=prob.device)
    if edge_index_s.numel() == 0:
        return prior, mask

    edge_index = edge_index_s[:, :2].long()
    sign = edge_index_s[:, 2]
    src = edge_index[:, 0]
    dst = edge_index[:, 1]

    src_prob = prob[src].clamp(eps, 1.0 - eps)
    msg_prob = torch.where(sign.view(-1, 1) > 0, src_prob, 1.0 - src_prob)

    log_sum = prob.new_zeros((num_nodes, 1))
    count = prob.new_zeros((num_nodes, 1))
    log_sum.index_add_(0, dst, torch.log(msg_prob))
    count.index_add_(0, dst, prob.new_ones((dst.size(0), 1)))

    mask = count > 0
    logic = torch.exp(log_sum).clamp(eps, 1.0 - eps)
    prior = torch.where(mask, logic, prob)
    return prior, mask


class PhaseStateInitializer(nn.Module):
    """Initializes original-phase and inverse-phase node states."""

    def __init__(self, in_dim: int, state_dim: int, dropout: float = 0.0):
        super().__init__()
        self.same_proj = nn.Linear(in_dim, state_dim)
        self.inv_proj = nn.Linear(in_dim, state_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.same_proj.reset_parameters()
        self.inv_proj.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        same = torch.tanh(self.same_proj(x))
        inv = torch.tanh(self.inv_proj(x))
        return self.dropout(torch.cat([same, inv], dim=-1))


class PhaseTransitionConv(nn.Module):
    """
    Phase-state transition message passing.

    Positive edges keep phase:
        same -> same, inv -> inv
    Negative edges flip phase:
        same -> inv, inv -> same
    """

    def __init__(self, state_dim: int, context_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim

        self.message_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.self_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * state_dim + context_dim, state_dim),
            nn.Sigmoid(),
        )
        self.update_cell = nn.GRUCell(state_dim, state_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.message_proj.reset_parameters()
        self.self_proj.reset_parameters()
        for module in self.edge_gate:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.update_cell.reset_parameters()

    def _edge_messages(self, src_state: Tensor, dst_state: Tensor, edge_index: Tensor, graph_context: Tensor) -> Tensor:
        num_nodes = dst_state.size(0)
        if edge_index.numel() == 0:
            return dst_state.new_zeros((num_nodes, self.state_dim))

        src, dst = edge_index[0], edge_index[1]
        msg = self.message_proj(src_state[src])
        ctx = graph_context.view(1, -1).expand(msg.size(0), -1)
        gate = self.edge_gate(torch.cat([msg, dst_state[dst], ctx], dim=-1))
        msg = self.dropout(msg * gate)
        return _aggregate_mean(msg, dst, num_nodes)

    def forward(
        self,
        phase_state: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
        graph_context: Tensor,
        reverse: bool = False,
    ) -> Tensor:
        if reverse:
            pos_edge_index = pos_edge_index.flip(0)
            neg_edge_index = neg_edge_index.flip(0)

        same, inv = phase_state.chunk(2, dim=-1)

        same_msg = self._edge_messages(same, same, pos_edge_index, graph_context)
        same_msg = same_msg + self._edge_messages(inv, same, neg_edge_index, graph_context)
        same_msg = same_msg + self.self_proj(same)

        inv_msg = self._edge_messages(inv, inv, pos_edge_index, graph_context)
        inv_msg = inv_msg + self._edge_messages(same, inv, neg_edge_index, graph_context)
        inv_msg = inv_msg + self.self_proj(inv)

        next_same = self.update_cell(same_msg, same)
        next_inv = self.update_cell(inv_msg, inv)
        return torch.cat([next_same, next_inv], dim=-1)


class PhaseTransitionBlock(nn.Module):
    """Forward fanin transition plus backward fanout transition."""

    def __init__(self, state_dim: int, context_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        self.forward_conv = PhaseTransitionConv(state_dim, context_dim=context_dim, dropout=dropout)
        self.backward_conv = PhaseTransitionConv(state_dim, context_dim=context_dim, dropout=dropout)
        phase_dim = 2 * state_dim
        self.fusion = nn.Sequential(
            nn.Linear(3 * phase_dim + context_dim, phase_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(phase_dim, phase_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.forward_conv.reset_parameters()
        self.backward_conv.reset_parameters()
        for module in self.fusion:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(
        self,
        phase_state: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
        graph_context: Tensor,
        use_backward: bool = True,
    ) -> Tensor:
        fwd = self.forward_conv(phase_state, pos_edge_index, neg_edge_index, graph_context, reverse=False)
        if use_backward:
            bwd = self.backward_conv(phase_state, pos_edge_index, neg_edge_index, graph_context, reverse=True)
        else:
            bwd = phase_state.new_zeros(phase_state.size())
        ctx = graph_context.view(1, -1).expand(phase_state.size(0), -1)
        return self.fusion(torch.cat([phase_state, fwd, bwd, ctx], dim=-1))


class GatedResidual(nn.Module):
    """Graph-context controlled residual update for stable deep propagation."""

    def __init__(self, feature_dim: int, context_dim: int = 8, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(16, feature_dim // 4)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * feature_dim + context_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.gate_net:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        last = self.gate_net[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, -2.0)

    def forward(self, current: Tensor, proposal: Tensor, graph_context: Tensor, layer_ratio: float) -> Tensor:
        pooled_current = current.mean(dim=0, keepdim=True)
        pooled_proposal = proposal.mean(dim=0, keepdim=True)
        ctx = graph_context.view(1, -1)
        layer = current.new_tensor([[float(layer_ratio)]])
        gate = torch.sigmoid(self.gate_net(torch.cat([pooled_current, pooled_proposal, ctx, layer], dim=-1)))
        return current + gate * (proposal - current)


class LogicAwareSPPHead(nn.Module):
    """Blend neural SPP prediction with a signed-fanin logic prior."""

    def __init__(self, feature_dim: int, context_dim: int = 8, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.neural_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.mix_gate = nn.Sequential(
            nn.Linear(feature_dim + context_dim + 2, max(16, hidden_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.neural_head:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        for module in self.mix_gate:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        last = self.mix_gate[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 1.0)

    def forward(self, z: Tensor, edge_index_s: Tensor, graph_context: Tensor) -> Tensor:
        neural_prob = torch.sigmoid(self.neural_head(z))
        logic_prob, logic_mask = signed_logic_prior(neural_prob, edge_index_s)

        ctx = graph_context.view(1, -1).expand(z.size(0), -1)
        gate_input = torch.cat([z, neural_prob, logic_prob, ctx], dim=-1)
        gate = torch.sigmoid(self.mix_gate(gate_input))

        mixed = gate * neural_prob + (1.0 - gate) * logic_prob
        return torch.where(logic_mask, mixed, neural_prob).clamp(1e-6, 1.0 - 1e-6)


class CircuitLayerMoE(nn.Module):
    """
    Circuit-context router over layer-wise node representations.

    The experts consume states from different propagation depths. The router uses
    graph-level circuit statistics and the current representation summary to
    decide how much each depth should contribute.
    """

    def __init__(
        self,
        feature_dim: int,
        num_experts: int,
        context_dim: int = 8,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.feature_dim = feature_dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim, feature_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.router = nn.Sequential(
            nn.Linear(feature_dim + context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for expert in self.experts:
            for module in expert:
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
        for module in self.router:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.output_norm.reset_parameters()

    def forward(self, states: list[Tensor], graph_context: Tensor) -> tuple[Tensor, Tensor]:
        if len(states) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} layer states, got {len(states)}.")

        expert_outputs = torch.stack(
            [expert(state) for expert, state in zip(self.experts, states)],
            dim=0,
        )

        state_summary = torch.stack([state.mean(dim=0) for state in states], dim=0).mean(dim=0, keepdim=True)
        ctx = graph_context.view(1, -1)
        weights = torch.softmax(self.router(torch.cat([state_summary, ctx], dim=-1)), dim=-1)
        fused = torch.sum(expert_outputs * weights.view(self.num_experts, 1, 1), dim=0)

        # Keep the strongest shallow/deep signal available while letting experts refine it.
        residual = torch.sum(
            torch.stack(states, dim=0) * weights.view(self.num_experts, 1, 1),
            dim=0,
        )
        return self.output_norm(fused + residual), weights.squeeze(0)

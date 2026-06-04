# coding=utf-8
from __future__ import annotations

import torch


def create_spectral_features(
    pos_edge_index: torch.LongTensor,
    neg_edge_index: torch.LongTensor,
    node_num: int,
    dim: int,
) -> torch.FloatTensor:
    try:
        import scipy.sparse as sp
        from sklearn.decomposition import TruncatedSVD
    except ImportError as exc:
        raise ImportError(
            "Spectral features require scipy and scikit-learn. "
            "Use --feature_type one-hot to avoid these optional dependencies."
        ) from exc

    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_index = edge_index.to(torch.device("cpu"))

    pos_val = torch.full((pos_edge_index.size(1),), 2, dtype=torch.float)
    neg_val = torch.full((neg_edge_index.size(1),), 0, dtype=torch.float)
    val = torch.cat([pos_val, neg_val], dim=0)

    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    val = torch.cat([val, val], dim=0)

    sparse = torch.sparse_coo_tensor(edge_index, val, (node_num, node_num)).coalesce()
    edge_index = sparse.indices().detach().numpy()
    val = (sparse.values() - 1).detach().numpy()
    matrix = sp.coo_matrix((val, edge_index), shape=(node_num, node_num))
    svd = TruncatedSVD(n_components=dim, n_iter=128)
    svd.fit(matrix)
    x = svd.components_.T
    return torch.from_numpy(x).to(torch.float)


def split_signed_edges(edge_index_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = edge_index_s.device
    empty = torch.empty((2, 0), dtype=torch.long, device=device)
    if edge_index_s.numel() == 0:
        return empty, empty

    edge_index = edge_index_s[:, :2].long()
    sign = edge_index_s[:, 2]
    pos_edge_index = edge_index[sign > 0].t().contiguous()
    neg_edge_index = edge_index[sign < 0].t().contiguous()

    if pos_edge_index.numel() == 0:
        pos_edge_index = empty
    if neg_edge_index.numel() == 0:
        neg_edge_index = empty
    return pos_edge_index, neg_edge_index


def build_graph_context(edge_index_s: torch.Tensor, num_nodes: int) -> torch.Tensor:
    device = edge_index_s.device
    dtype = torch.float32
    num_nodes_f = torch.tensor(float(max(num_nodes, 1)), dtype=dtype, device=device)
    num_edges_f = torch.tensor(float(edge_index_s.size(0)), dtype=dtype, device=device)

    if edge_index_s.numel() == 0:
        return torch.stack(
            [
                torch.log1p(num_nodes_f),
                torch.zeros((), dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
            ]
        )

    sign = edge_index_s[:, 2]
    edge_index = edge_index_s[:, :2].long()
    src, dst = edge_index[:, 0], edge_index[:, 1]

    pos_count = (sign > 0).sum().to(dtype)
    neg_count = (sign < 0).sum().to(dtype)
    denom = num_edges_f.clamp_min(1.0)
    pos_ratio = pos_count / denom
    neg_ratio = neg_count / denom
    edge_density = num_edges_f / num_nodes_f

    ones = torch.ones(edge_index.size(0), dtype=dtype, device=device)
    fanin = torch.zeros(num_nodes, dtype=dtype, device=device)
    fanout = torch.zeros(num_nodes, dtype=dtype, device=device)
    fanin.index_add_(0, dst, ones)
    fanout.index_add_(0, src, ones)

    fanin_mean = fanin.mean() if num_nodes > 0 else torch.zeros((), dtype=dtype, device=device)
    fanout_mean = fanout.mean() if num_nodes > 0 else torch.zeros((), dtype=dtype, device=device)
    fanin_std = fanin.std(unbiased=False) if num_nodes > 1 else torch.zeros((), dtype=dtype, device=device)

    return torch.stack(
        [
            torch.log1p(num_nodes_f),
            torch.log1p(num_edges_f),
            pos_ratio,
            neg_ratio,
            edge_density,
            fanin_mean,
            fanout_mean,
            fanin_std,
        ]
    )

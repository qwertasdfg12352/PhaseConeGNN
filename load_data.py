# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


@dataclass
class GraphSample:
    graph_name: str
    graph_dir: Path
    edge_index_s: torch.Tensor
    pair_index: torch.Tensor
    node_features: torch.Tensor | None
    node_labels: torch.Tensor
    tt_dis: torch.Tensor
    eq_sim: torch.Tensor


def _read_csv_array(path: Path, delimiter: str | None = ",") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    arr = np.genfromtxt(path, delimiter=delimiter)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(arr)


def _read_vector(path: Path) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=None)
    return np.atleast_1d(np.asarray(arr, dtype=np.float32))


def _build_node_map(edge_rows: np.ndarray, raw_node_count: int) -> dict[int, int]:
    node_map: dict[int, int] = {}
    for row in edge_rows:
        src = int(row[0])
        dst = int(row[1])
        if src not in node_map:
            node_map[src] = len(node_map)
        if dst not in node_map:
            node_map[dst] = len(node_map)

    # Keep isolated or label-only nodes if they exist in raw feature/prob files.
    for old_id in range(raw_node_count):
        if old_id not in node_map:
            node_map[old_id] = len(node_map)
    return node_map


def _load_existing_node_map(graph_dir: Path) -> dict[int, int] | None:
    map_path = graph_dir / "processed" / "node_id_map.json"
    if not map_path.exists():
        return None
    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(old): int(new) for old, new in raw.items()}


def _remap_rows(raw: np.ndarray, node_map: dict[int, int], width: int | None = None) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.ndim == 1:
        raw = raw[:, None]
    if width is None:
        width = raw.shape[1]

    out = np.zeros((len(node_map), width), dtype=np.float32)
    for old_id, new_id in node_map.items():
        if 0 <= old_id < raw.shape[0]:
            row = raw[old_id]
            out[new_id, : min(width, row.shape[0])] = row[:width]
    return out


def _remap_vector(raw: np.ndarray, node_map: dict[int, int]) -> np.ndarray:
    out = np.zeros((len(node_map),), dtype=np.float32)
    for old_id, new_id in node_map.items():
        if 0 <= old_id < raw.shape[0]:
            out[new_id] = raw[old_id]
    return out


def _one_hot_gate(raw_features: np.ndarray, num_gate_types: int) -> np.ndarray:
    if raw_features.ndim == 1:
        raw_features = raw_features[:, None]
    gate_col = 1 if raw_features.shape[1] > 1 else 0
    gate_ids = raw_features[:, gate_col].astype(np.int64)
    if gate_ids.size > 0 and (gate_ids.min() < 0 or gate_ids.max() >= num_gate_types):
        raise ValueError(
            f"Gate id out of one-hot range [0, {num_gate_types}). "
            f"Observed min={gate_ids.min()} max={gate_ids.max()}."
        )
    out = np.zeros((raw_features.shape[0], num_gate_types), dtype=np.float32)
    if gate_ids.size > 0:
        out[np.arange(gate_ids.shape[0]), gate_ids] = 1.0
    return out


def _remap_edge_index_s(edge_rows: np.ndarray, node_map: dict[int, int], device: torch.device) -> torch.Tensor:
    if edge_rows.ndim == 1:
        edge_rows = edge_rows[None, :]
    remapped = []
    for row in edge_rows:
        src = node_map[int(row[0])]
        dst = node_map[int(row[1])]
        sign = 1 if float(row[2]) > 0 else -1
        remapped.append((src, dst, sign))
    if not remapped:
        return torch.empty((0, 3), dtype=torch.long, device=device)
    return torch.tensor(remapped, dtype=torch.long, device=device)


def _remap_pair_index(pair_index: Any, node_map: dict[int, int], device: torch.device) -> torch.Tensor:
    pair = np.asarray(pair_index)
    if pair.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    pair = np.atleast_2d(pair).astype(np.int64)

    remapped = []
    for src, dst in pair[:, :2]:
        if int(src) in node_map and int(dst) in node_map:
            remapped.append((node_map[int(src)], node_map[int(dst)]))
    if not remapped:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(remapped, dtype=torch.long, device=device).t().contiguous()


def _resolve_graph_dirs(data_root_path: Path, split_file: str, split_name: str) -> list[Path]:
    split_path = data_root_path / "split" / split_file / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    graph_dirs = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            path = Path(item)
            graph_dirs.append(path if path.is_absolute() else data_root_path / path)
    random.shuffle(graph_dirs)
    return graph_dirs


def _load_tt_labels(data_root_path: Path) -> dict[str, Any]:
    label_path = data_root_path / "npz" / "labels.npz"
    if not label_path.exists():
        return {}
    return np.load(label_path, allow_pickle=True)["labels"].item()


def load_graph_sample(args, graph_dir: Path, tt_pair_dict: dict[str, Any]) -> GraphSample:
    graph_dir = Path(graph_dir)
    graph_name = graph_dir.name
    raw_dir = graph_dir / "raw"

    edge_rows = _read_csv_array(raw_dir / "signed_edge.csv", delimiter=",")
    edge_rows = np.atleast_2d(edge_rows).astype(np.float64)

    raw_features = _read_csv_array(raw_dir / "node-feat.csv", delimiter=",")
    raw_features = np.atleast_2d(raw_features).astype(np.float32)
    raw_labels = _read_vector(raw_dir / "prob.csv")

    raw_node_count = max(raw_features.shape[0], raw_labels.shape[0])
    node_map = _load_existing_node_map(graph_dir)
    if node_map is None:
        node_map = _build_node_map(edge_rows, raw_node_count)

    edge_index_s = _remap_edge_index_s(edge_rows, node_map, args.device)

    if args.feature_type == "one-hot":
        raw_x = _one_hot_gate(raw_features, args.in_dim)
        x = _remap_rows(raw_x, node_map, width=args.in_dim)
        node_features = torch.from_numpy(x).float().to(args.device)
    elif args.feature_type == "raw":
        x = _remap_rows(raw_features, node_map, width=args.in_dim)
        node_features = torch.from_numpy(x).float().to(args.device)
    elif args.feature_type == "spectral":
        node_features = None
    else:
        raise ValueError(f"Unsupported feature_type: {args.feature_type}")

    labels = _remap_vector(raw_labels, node_map)
    node_labels = torch.from_numpy(labels).float().to(args.device)

    tt_info = tt_pair_dict.get(graph_name, {})
    pair_index = _remap_pair_index(tt_info.get("tt_pair_index", []), node_map, args.device)
    tt_dis_np = np.asarray(tt_info.get("tt_dis", []), dtype=np.float32)
    if pair_index.size(1) != tt_dis_np.shape[0]:
        tt_dis_np = tt_dis_np[: pair_index.size(1)]
    tt_dis = torch.from_numpy(tt_dis_np).float().to(args.device)
    eq_sim = 1.0 - tt_dis

    return GraphSample(
        graph_name=graph_name,
        graph_dir=graph_dir,
        edge_index_s=edge_index_s,
        pair_index=pair_index,
        node_features=node_features,
        node_labels=node_labels,
        tt_dis=tt_dis,
        eq_sim=eq_sim,
    )


def load_split(args, split_name: str, tt_pair_dict: dict[str, Any] | None = None) -> list[GraphSample]:
    tt_pair_dict = tt_pair_dict if tt_pair_dict is not None else _load_tt_labels(args.data_root_path)
    graph_dirs = _resolve_graph_dirs(args.data_root_path, args.split_file, split_name)
    if len(graph_dirs) == 0:
        return []

    num_workers = max(1, int(getattr(args, "num_workers", 1)))
    desc = f"Loading {split_name}"

    if num_workers == 1:
        return [
            load_graph_sample(args, graph_dir, tt_pair_dict)
            for graph_dir in tqdm(graph_dirs, desc=desc, unit="graph", dynamic_ncols=True)
        ]

    samples: list[GraphSample | None] = [None] * len(graph_dirs)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(load_graph_sample, args, graph_dir, tt_pair_dict): idx
            for idx, graph_dir in enumerate(graph_dirs)
        }
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(graph_dirs),
            desc=desc,
            unit="graph",
            dynamic_ncols=True,
        ):
            idx = future_to_idx[future]
            samples[idx] = future.result()

    return [sample for sample in samples if sample is not None]


def load_train_valid(args):
    tt_pair_dict = _load_tt_labels(args.data_root_path)
    train_data = load_split(args, "train", tt_pair_dict)
    valid_data = load_split(args, "valid", tt_pair_dict)
    return train_data, valid_data, tt_pair_dict


def load_test(args, tt_pair_dict: dict[str, Any] | None = None):
    tt_pair_dict = tt_pair_dict if tt_pair_dict is not None else _load_tt_labels(args.data_root_path)
    return load_split(args, "test", tt_pair_dict)

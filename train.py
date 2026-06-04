# coding=utf-8
from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from load_data import GraphSample, load_test, load_train_valid
from model import PhaseConeGNN


def get_logger(name, logfile=None):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ForgeEDA_proxy_processed")
    parser.add_argument("--data_root_path", type=str, default="")
    parser.add_argument("--task_type", type=str, default="prob", choices=["prob", "eq", "tt"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=None, help="Alias for --epochs.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="PhaseConeGNN")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--in_dim", type=int, default=3)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--eval_step", type=int, default=1)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--layer_num", type=int, default=3)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--device_backend", type=str, default="legacy",
                        choices=["legacy", "auto", "cpu", "cuda", "mps"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel graph-loading workers.")
    parser.add_argument("--split_file", type=str, default="0.05-0.05-0.9")
    parser.add_argument("--feature_type", type=str, default="one-hot",
                        choices=["one-hot", "raw", "spectral"])
    parser.add_argument("--loss_type", type=str, default="mae", choices=["mae", "mse"])
    parser.add_argument("--name_others", type=str, default="")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_backward", type=int, default=-1, choices=[-1, 0, 1],
                        help="-1 auto: prob disables backward, eq/tt enables backward; 0 off; 1 on.")
    parser.add_argument("--use_layer_moe", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lambda_not", type=float, default=0.05)
    parser.add_argument("--lambda_and", type=float, default=0.02)

    args = parser.parse_args()
    if args.epoch is not None:
        args.epochs = args.epoch

    if args.data_root_path:
        args.data_root_path = Path(args.data_root_path).expanduser().resolve()
    else:
        local_default = Path(__file__).resolve().parent.parent / "AIGDataset" / args.dataset
        home_default = Path.home() / "AIGDataset" / args.dataset
        args.data_root_path = local_default if local_default.exists() else home_default

    args.device = resolve_device(args.device_backend, args.device)

    os.makedirs(Path.cwd() / "ft_saved", exist_ok=True)
    os.makedirs(Path.cwd() / "results", exist_ok=True)
    suffix = f"{args.task_type}_{args.dataset}_{args.model}_{args.layer_num}"
    if args.name_others:
        suffix = f"{suffix}_{args.name_others}"
    args.ft_model_path = Path.cwd() / "ft_saved" / f"{suffix}_state_dict.pth"
    return args


def has_mps():
    return (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(device_backend, device_index):
    if device_backend == "legacy":
        if torch.cuda.is_available() and device_index >= 0:
            return torch.device(f"cuda:{device_index}")
        return torch.device("cpu")
    if device_backend == "auto":
        if torch.cuda.is_available():
            resolved_index = device_index if device_index >= 0 else 0
            return torch.device(f"cuda:{resolved_index}")
        if has_mps():
            return torch.device("mps")
        return torch.device("cpu")
    if device_backend == "cpu":
        return torch.device("cpu")
    if device_backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but CUDA is not available.")
        resolved_index = device_index if device_index >= 0 else 0
        return torch.device(f"cuda:{resolved_index}")
    if device_backend == "mps":
        if not has_mps():
            raise RuntimeError("MPS backend requested but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unsupported device backend: {device_backend}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(args):
    model = PhaseConeGNN(
        args=args,
        node_num=0,
        device=args.device,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        layer_num=args.layer_num,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def zero_loss_like(out_ref):
    return out_ref.sum() * 0.0


def loss_fn(pred, target, out_ref, typ="mae"):
    if pred.numel() == 0 or target.numel() == 0:
        return zero_loss_like(out_ref)
    if typ == "mae":
        return F.l1_loss(pred, target)
    return F.mse_loss(pred, target)


def zero_normalization(x):
    if x.numel() <= 1:
        return x
    return (x - torch.mean(x)) / (torch.std(x) + 1e-8)


def compute_pair_outputs(out_emb, pair_index):
    if pair_index.numel() == 0:
        return out_emb.new_zeros((0,)), out_emb.new_zeros((0,))
    node_a = out_emb[pair_index[0]]
    node_b = out_emb[pair_index[1]]
    cos_sim = torch.cosine_similarity(node_a, node_b, eps=1e-8)
    emb_dis = 1.0 - cos_sim
    return cos_sim, emb_dis


def compute_task_losses(args, prob, node_labels, out_emb, pair_index, tt_dis, eq_sim):
    prob_loss = loss_fn(prob, node_labels, prob, typ=args.loss_type)
    cos_sim, emb_dis = compute_pair_outputs(out_emb, pair_index)
    tt_loss = loss_fn(zero_normalization(emb_dis), zero_normalization(tt_dis), prob, typ=args.loss_type)
    eq_loss = loss_fn(cos_sim, eq_sim, prob, typ=args.loss_type)
    return prob_loss, eq_loss, tt_loss


def compute_train_losses(args, prob, node_labels, out_emb, pair_index, tt_dis, eq_sim, edge_index_s):
    if args.task_type == "prob":
        prob_loss = loss_fn(prob, node_labels, prob, typ=args.loss_type)
        not_loss = AIG_Not_Edge_Loss(edge_index_s, prob, typ=args.loss_type)
        and_loss = AIG_AND_Product_Loss(edge_index_s, prob, typ=args.loss_type)
        logic_loss = args.lambda_not * not_loss + args.lambda_and * and_loss
        zero = zero_loss_like(prob)
        return prob_loss, zero, zero, logic_loss, prob_loss + logic_loss

    if args.task_type == "eq":
        cos_sim, _ = compute_pair_outputs(out_emb, pair_index)
        eq_loss = loss_fn(cos_sim, eq_sim, prob, typ=args.loss_type)
        zero = zero_loss_like(prob)
        return zero, eq_loss, zero, zero, eq_loss

    _, emb_dis = compute_pair_outputs(out_emb, pair_index)
    tt_loss = loss_fn(zero_normalization(emb_dis), zero_normalization(tt_dis), prob, typ=args.loss_type)
    zero = zero_loss_like(prob)
    return zero, zero, tt_loss, zero, tt_loss


def AIG_Not_Edge_Loss(edge_index_s, prob, typ="mae"):
    not_edges = edge_index_s[edge_index_s[:, 2] < 0]
    if not_edges.numel() == 0:
        return zero_loss_like(prob)
    input_prob = prob[not_edges[:, 0]]
    output_prob = prob[not_edges[:, 1]]
    target = torch.ones_like(input_prob)
    if typ == "mae":
        return F.l1_loss(input_prob + output_prob, target)
    return F.mse_loss(input_prob + output_prob, target)


def AIG_AND_Edge_Loss(edge_index_s, prob, typ="mae"):
    sorted_indices = torch.argsort(edge_index_s[:, 1])
    edge_index_s = edge_index_s[sorted_indices]
    and_edges = edge_index_s[edge_index_s[:, 2] > 0]
    complete_edge_cnt = (and_edges.shape[0] // 2) * 2
    if complete_edge_cnt == 0:
        return zero_loss_like(prob)
    and_edges = and_edges[:complete_edge_cnt]
    inputs = prob[and_edges[:, 0]].view(-1, 2)
    outputs = prob[and_edges[:, 1]].view(-1, 2)
    min_inputs = torch.min(inputs, dim=1)[0]
    if typ == "mae":
        return F.l1_loss(outputs[:, 0], min_inputs)
    return F.mse_loss(outputs[:, 0], min_inputs)


def AIG_AND_Product_Loss(edge_index_s, prob, typ="mae", eps=1e-6):
    pos_edges = edge_index_s[edge_index_s[:, 2] > 0]
    if pos_edges.numel() == 0:
        return zero_loss_like(prob)

    src = pos_edges[:, 0].long()
    dst = pos_edges[:, 1].long()
    num_nodes = prob.size(0)
    src_prob = prob[src].clamp(eps, 1.0 - eps)

    log_sum = prob.new_zeros((num_nodes, 1))
    count = prob.new_zeros((num_nodes, 1))
    log_sum.index_add_(0, dst, torch.log(src_prob))
    count.index_add_(0, dst, prob.new_ones((dst.size(0), 1)))

    mask = count.squeeze(-1) >= 2
    if not torch.any(mask):
        return zero_loss_like(prob)

    target = torch.exp(log_sum[mask]).clamp(eps, 1.0 - eps)
    pred = prob[mask]
    if typ == "mae":
        return F.l1_loss(pred, target)
    return F.mse_loss(pred, target)


def AIG_PI_AND_Edge_Loss(edge_index_s, prob, typ="mae"):
    edge_index = edge_index_s[:, :2]
    edge_sign = edge_index_s[:, 2]
    neg_nodes = edge_index[edge_sign < 0][:, 1]
    pos_mask = torch.ones(prob.size(0), dtype=torch.bool, device=prob.device)
    if neg_nodes.numel() > 0:
        pos_mask[neg_nodes] = False
    loss_nodes = prob[pos_mask]
    if loss_nodes.numel() == 0:
        return zero_loss_like(prob)
    if typ == "mae":
        return F.l1_loss(loss_nodes, torch.zeros_like(loss_nodes))
    return F.mse_loss(loss_nodes, torch.zeros_like(loss_nodes))


@torch.no_grad()
def test(args, model, dataset: list[GraphSample]):
    model.eval()
    results = {
        "prob": 0.0,
        "not": 0.0,
        "and": 0.0,
        "tt": 0.0,
        "eq": 0.0,
        "pi_and": 0.0,
        "level": defaultdict(lambda: {"value": 0.0, "cnt": 0.0}),
    }
    if len(dataset) == 0:
        return results

    for sample in dataset:
        labels = sample.node_labels.unsqueeze(1)
        out_emb, prob = model(sample.node_features, sample.edge_index_s)

        results["not"] += AIG_Not_Edge_Loss(sample.edge_index_s, prob, typ=args.loss_type).item()
        results["and"] += AIG_AND_Edge_Loss(sample.edge_index_s, prob, typ=args.loss_type).item()
        results["pi_and"] += AIG_PI_AND_Edge_Loss(sample.edge_index_s, prob, typ=args.loss_type).item()

        prob_loss, eq_loss, tt_loss = compute_task_losses(
            args, prob, labels, out_emb, sample.pair_index, sample.tt_dis, sample.eq_sim
        )
        results["prob"] += prob_loss.item()
        results["eq"] += eq_loss.item()
        results["tt"] += tt_loss.item()

    for key in ["prob", "not", "and", "tt", "eq", "pi_and"]:
        results[key] /= len(dataset)
    return results


def train(args, model, optimizer, train_data, valid_data, logger):
    logger.info("*********** Start PhaseConeGNN Train ***********")
    best_loss = float("inf")
    best_model_info = None
    patience = args.patience
    total_time = 0.0
    cnt_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_prob_loss = 0.0
        total_tt_loss = 0.0
        total_eq_loss = 0.0
        total_logic_loss = 0.0
        random.shuffle(train_data)

        for step, sample in enumerate(train_data, 1):
            labels = sample.node_labels.unsqueeze(1)
            out_emb, prob = model(sample.node_features, sample.edge_index_s)
            prob_loss, eq_loss, tt_loss, logic_loss, loss = compute_train_losses(
                args,
                prob,
                labels,
                out_emb,
                sample.pair_index,
                sample.tt_dis,
                sample.eq_sim,
                sample.edge_index_s,
            )

            total_prob_loss += prob_loss.item()
            total_eq_loss += eq_loss.item()
            total_tt_loss += tt_loss.item()
            total_logic_loss += logic_loss.item()

            loss.backward()

            if step % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_time += time.time() - start
        cnt_epoch += 1

        if epoch % args.eval_step == 0:
            valid_results = test(args, model, valid_data)
            total_prob_loss /= max(len(train_data), 1)
            total_eq_loss /= max(len(train_data), 1)
            total_tt_loss /= max(len(train_data), 1)
            total_logic_loss /= max(len(train_data), 1)

            logger.info(
                "Epoch: {:03d} | [Train] Prob: {:.4f} Eq: {:.4f} TT: {:.4f} Logic: {:.4f} | "
                "[Valid] Prob: {:.4f} Eq: {:.4f} TT: {:.4f} | "
                "PI_AND: {:.4f} AND: {:.4f} NOT: {:.4f}".format(
                    epoch,
                    total_prob_loss,
                    total_eq_loss,
                    total_tt_loss,
                    total_logic_loss,
                    valid_results["prob"],
                    valid_results["eq"],
                    valid_results["tt"],
                    valid_results["pi_and"],
                    valid_results["and"],
                    valid_results["not"],
                )
            )

            if valid_results[args.task_type] < best_loss:
                best_loss = valid_results[args.task_type]
                best_model_info = {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "best_epoch": epoch,
                    "best_valid": best_loss,
                }
                patience = args.patience

            patience -= 1
            if patience <= 0:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    if best_model_info is None:
        best_model_info = {
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "best_epoch": cnt_epoch,
            "best_valid": best_loss,
        }
    torch.save(best_model_info, args.ft_model_path)
    return total_time / max(cnt_epoch, 1)


def main():
    args = parameter_parser()
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    if args.name_others:
        logname = f"{args.task_type}_{args.dataset}_{args.model}_{args.layer_num}__{args.split_file}_{args.name_others}_{timestamp}.log"
    else:
        logname = f"{args.task_type}_{args.dataset}_{args.model}_{args.layer_num}__{args.split_file}_{timestamp}.log"
    logfile = Path("results") / logname
    logger = get_logger(__name__, logfile=logfile)

    logger.info(f"Model: {args.model}")
    logger.info(f"Device: backend={args.device_backend} resolved={args.device}")
    logger.info(f"Data root: {args.data_root_path}")
    logger.info(f"Split: {args.split_file}")
    logger.info("*********** Load Train/Valid Data ***********")
    train_data, valid_data, tt_pair_dict = load_train_valid(args)
    logger.info(f"Graphs: train={len(train_data)} valid={len(valid_data)}")

    model, optimizer = load_model(args)
    avg_train_time = train(args, model, optimizer, train_data, valid_data, logger)

    logger.info("*********** Load Best Checkpoint ***********")
    checkpoint = torch.load(args.ft_model_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("*********** Load Test Data ***********")
    test_data = load_test(args, tt_pair_dict)
    logger.info(f"Graphs: test={len(test_data)}")

    logger.info("*********** Test Result ***********")
    test_results = test(args, model, test_data)
    logger.info(
        "[Test] Prob: {:.4f} Eq: {:.4f} TT: {:.4f} | "
        "PI_AND: {:.4f} AND: {:.4f} NOT: {:.4f} | Avg.Train_time: {:.4f}".format(
            test_results["prob"],
            test_results["eq"],
            test_results["tt"],
            test_results["pi_and"],
            test_results["and"],
            test_results["not"],
            avg_train_time,
        )
    )


if __name__ == "__main__":
    main()

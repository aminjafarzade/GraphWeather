from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F


def _to_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(dtype=torch.float32)


def flatten_embeddings(tensor: torch.Tensor) -> torch.Tensor:
    """Return an embedding matrix [M, C] from common graph/grid activation shapes."""
    x = _to_float_tensor(tensor)
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x.reshape(-1, x.shape[-1])
    if x.dim() == 4:
        # Weather grid tensors are [B, C, H, W].
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    if x.dim() > 4:
        return x.reshape(-1, x.shape[-1])
    raise ValueError(f"Expected activation with at least 2 dimensions, got {tuple(x.shape)}")


def sample_rows(x: torch.Tensor, max_rows: int | None) -> torch.Tensor:
    if max_rows is None or int(max_rows) <= 0 or x.shape[0] <= int(max_rows):
        return x
    idx = torch.linspace(0, x.shape[0] - 1, int(max_rows), device=x.device).long()
    return x.index_select(0, idx)


def embedding_variance(embeddings: torch.Tensor, eps: float = 1.0e-12) -> float:
    x = flatten_embeddings(embeddings)
    if x.shape[0] < 2:
        return float("nan")
    del eps
    value = torch.var(x, dim=0, unbiased=False).mean()
    return float(value.item())


def pairwise_cosine_stats(
    embeddings: torch.Tensor,
    max_rows: int | None = 1024,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    x = sample_rows(flatten_embeddings(embeddings), max_rows)
    if x.shape[0] < 2:
        return {"cosine_mean": float("nan"), "cosine_std": float("nan"), "mad_cosine": float("nan")}
    z = F.normalize(x, p=2, dim=-1, eps=eps)
    cosine = z @ z.transpose(0, 1)
    mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
    off_diag = cosine[mask]
    return {
        "cosine_mean": float(off_diag.mean().item()),
        "cosine_std": float(off_diag.std(unbiased=False).item()),
        "mad_cosine": float((1.0 - off_diag).mean().item()),
    }


def mean_average_distance(
    embeddings: torch.Tensor,
    max_rows: int | None = 1024,
    eps: float = 1.0e-12,
) -> float:
    return pairwise_cosine_stats(embeddings, max_rows=max_rows, eps=eps)["mad_cosine"]


def effective_rank(
    embeddings: torch.Tensor,
    max_rows: int | None = 2048,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    x = sample_rows(flatten_embeddings(embeddings), max_rows)
    if x.shape[0] < 2 or x.shape[1] < 1:
        return {
            "effective_rank": float("nan"),
            "effective_rank_norm": float("nan"),
            "stable_rank": float("nan"),
        }
    x = x - x.mean(dim=0, keepdim=True)
    try:
        singular_values = torch.linalg.svdvals(x)
    except RuntimeError:
        singular_values = torch.linalg.svdvals(x.cpu()).to(device=x.device)
    singular_sum = singular_values.sum()
    if float(singular_sum.item()) <= eps:
        return {
            "effective_rank": 0.0,
            "effective_rank_norm": 0.0,
            "stable_rank": 0.0,
        }
    p = singular_values / (singular_sum + eps)
    entropy = -(p * torch.log(p + eps)).sum()
    eff_rank = torch.exp(entropy)
    denom = float(min(x.shape[0], x.shape[1]))
    stable = (x.square().sum() / (singular_values[0].square() + eps)) if singular_values.numel() else x.new_tensor(float("nan"))
    return {
        "effective_rank": float(eff_rank.item()),
        "effective_rank_norm": float((eff_rank / max(denom, eps)).item()),
        "stable_rank": float(stable.item()),
    }


def embedding_stats(
    embeddings: torch.Tensor,
    embedding_sample_nodes: int | None = 2048,
    pairwise_sample_nodes: int | None = 1024,
) -> dict[str, float]:
    sampled = sample_rows(flatten_embeddings(embeddings), embedding_sample_nodes)
    stats = {"embedding_variance": embedding_variance(sampled)}
    stats.update(pairwise_cosine_stats(sampled, max_rows=pairwise_sample_nodes))
    stats.update(effective_rank(sampled, max_rows=embedding_sample_nodes))
    return stats


def _attention_stats_fixed_k(attn: torch.Tensor, eps: float) -> dict[str, float]:
    # Expected local GAT shape: [B, N, K, H] or [N, K, H].
    p = _to_float_tensor(attn)
    if p.dim() == 3:
        p = p.unsqueeze(0)
    if p.dim() != 4:
        raise ValueError(f"Expected fixed-k attention [B,N,K,H], got {tuple(attn.shape)}")
    degree = int(p.shape[2])
    if degree <= 0:
        return {
            "entropy": float("nan"),
            "entropy_norm": float("nan"),
            "max_weight_mean": float("nan"),
            "max_weight_std": float("nan"),
            "degree_mean": float("nan"),
            "uniform_baseline_max_weight": float("nan"),
        }
    p = p.clamp_min(0.0)
    p = p / (p.sum(dim=2, keepdim=True) + eps)
    entropy = -(p * torch.log(p + eps)).sum(dim=2)
    max_weight = p.max(dim=2).values
    log_degree = math.log(max(float(degree), 1.0) + eps)
    entropy_norm = entropy / max(log_degree, eps)
    return {
        "entropy": float(entropy.mean().item()),
        "entropy_norm": float(entropy_norm.mean().item()),
        "max_weight_mean": float(max_weight.mean().item()),
        "max_weight_std": float(max_weight.std(unbiased=False).item()),
        "degree_mean": float(degree),
        "uniform_baseline_max_weight": float(1.0 / float(degree)),
    }


def _grouped_attention_stats(attn: torch.Tensor, edge_index: torch.Tensor, eps: float) -> dict[str, float]:
    p = _to_float_tensor(attn)
    if p.dim() == 1:
        p = p[:, None]
    if p.dim() == 2:
        p = p.unsqueeze(0)
    if p.dim() != 3:
        raise ValueError(f"Expected edge attention [E,H] or [B,E,H], got {tuple(attn.shape)}")
    if edge_index is None or edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index [2,E] is required for edge-list attention diagnostics.")
    dst = edge_index[1].detach().to(device=p.device, dtype=torch.long)
    if dst.numel() != p.shape[1]:
        raise ValueError(f"edge_index has {dst.numel()} edges but attention has {p.shape[1]}")
    order = torch.argsort(dst)
    dst_sorted = dst.index_select(0, order)
    p_sorted = p.index_select(1, order).clamp_min(0.0)
    unique_dst, counts = torch.unique_consecutive(dst_sorted, return_counts=True)
    del unique_dst
    entropies = []
    max_weights = []
    uniform = []
    start = 0
    for count in counts.tolist():
        group = p_sorted[:, start : start + count, :]
        group = group / (group.sum(dim=1, keepdim=True) + eps)
        entropies.append((-(group * torch.log(group + eps)).sum(dim=1)).reshape(-1))
        max_weights.append(group.max(dim=1).values.reshape(-1))
        uniform.append(1.0 / float(max(count, 1)))
        start += count
    if not entropies:
        return {
            "entropy": float("nan"),
            "entropy_norm": float("nan"),
            "max_weight_mean": float("nan"),
            "max_weight_std": float("nan"),
            "degree_mean": float("nan"),
            "uniform_baseline_max_weight": float("nan"),
        }
    entropy = torch.cat(entropies)
    max_weight = torch.cat(max_weights)
    degrees = counts.to(dtype=torch.float32, device=p.device)
    entropy_norm_values = []
    start = 0
    for count in counts.tolist():
        group_entropy = entropies[start]
        entropy_norm_values.append(group_entropy / max(math.log(max(float(count), 1.0) + eps), eps))
        start += 1
    entropy_norm = torch.cat(entropy_norm_values)
    return {
        "entropy": float(entropy.mean().item()),
        "entropy_norm": float(entropy_norm.mean().item()),
        "max_weight_mean": float(max_weight.mean().item()),
        "max_weight_std": float(max_weight.std(unbiased=False).item()),
        "degree_mean": float(degrees.mean().item()),
        "uniform_baseline_max_weight": float(sum(uniform) / max(len(uniform), 1)),
    }


def attention_entropy_stats(
    attn: torch.Tensor,
    edge_index: torch.Tensor | None = None,
    num_nodes: int | None = None,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    del num_nodes
    if attn.dim() in {3, 4}:
        if attn.dim() == 4 or (attn.dim() == 3 and edge_index is None):
            return _attention_stats_fixed_k(attn, eps)
    return _grouped_attention_stats(attn, edge_index, eps)


def global_gradient_norm(model: torch.nn.Module, eps: float = 1.0e-12) -> float:
    del eps
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        value = float(param.grad.detach().float().norm(2).item())
        total += value * value
    return float(math.sqrt(total))


def _gradient_group_name(name: str) -> str:
    parts = name.split(".")
    if not parts:
        return "model"
    if parts[0] in {"embed", "head"}:
        return parts[0]
    if parts[0] in {"encoder", "decoder"}:
        return parts[0]
    if parts[0] == "processor":
        if len(parts) >= 2:
            return f"processor.{parts[1]}"
        return "processor"
    return parts[0]


def gradient_norms_by_block(model: torch.nn.Module, eps: float = 1.0e-12) -> dict[str, float]:
    del eps
    totals: dict[str, float] = defaultdict(float)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        group = _gradient_group_name(name)
        value = float(param.grad.detach().float().norm(2).item())
        totals[group] += value * value
    return {name: float(math.sqrt(value)) for name, value in sorted(totals.items())}


def gradient_metrics(model: torch.nn.Module) -> dict[str, float]:
    metrics = {"global": global_gradient_norm(model)}
    metrics.update(gradient_norms_by_block(model))
    return metrics


def dirichlet_energy_stats(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    """Graph Dirichlet energy of node features h over the (directed kNN) edges.

    E(h) = sum_{(u,v) in E} ||h_u - h_v||^2, reported per-node (/N), per-edge (/E),
    and scale-invariant normalized (/ sum_v ||h_v||^2) -- the last is the oversmoothing
    indicator. node_features must be [B,N,C] or [N,C] with N == number of graph nodes
    (do NOT row-sample before calling; rows must stay aligned to edge_index).
    """
    x = node_features.detach().to(dtype=torch.float32)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"Expected node features [B,N,C] or [N,C], got {tuple(node_features.shape)}")
    if edge_index is None or edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index [2,E] is required for Dirichlet energy.")
    _, num_nodes, _ = x.shape
    src = edge_index[0].to(device=x.device, dtype=torch.long)
    dst = edge_index[1].to(device=x.device, dtype=torch.long)
    if int(src.max()) >= num_nodes or int(dst.max()) >= num_nodes:
        raise ValueError(
            f"edge_index indexes {int(max(src.max(), dst.max())) + 1} nodes but features have {num_nodes}."
        )
    num_edges = int(edge_index.shape[1])
    diff = x[:, dst, :] - x[:, src, :]                    # [B, E, C]
    edge_energy = diff.square().sum(dim=-1).sum(dim=1)     # [B]  sum over edges
    mass = x.square().sum(dim=-1).sum(dim=-1)              # [B]  sum_v ||h_v||^2
    return {
        "dirichlet_energy": float((edge_energy / float(max(num_nodes, 1))).mean().item()),
        "dirichlet_energy_per_edge": float((edge_energy / float(max(num_edges, 1))).mean().item()),
        "dirichlet_energy_norm": float((edge_energy / (mass + eps)).mean().item()),
    }

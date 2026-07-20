from __future__ import annotations

import numpy as np


def _sym_adjacency(edge_index, num_nodes: int):
    import scipy.sparse as sp

    ei = edge_index.detach().cpu().long().numpy()
    src, dst = ei[0], ei[1]
    A = sp.coo_matrix((np.ones(src.shape[0], dtype=np.float64), (dst, src)),
                      shape=(num_nodes, num_nodes)).tocsr()
    A = A.maximum(A.T)          # undirected 0/1 adjacency
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def spectral_gap(A) -> float:
    """lambda_2 of the symmetric-normalized Laplacian (normalized algebraic connectivity)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = A.shape[0]
    dinv = 1.0 / np.sqrt(np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1e-12))
    Lsym = sp.identity(n, format="csr") - sp.diags(dinv) @ A @ sp.diags(dinv)
    k = min(6, max(2, n - 2))
    try:
        vals = spla.eigsh(Lsym, k=k, which="SA", return_eigenvectors=False)
    except Exception:
        vals = np.linalg.eigvalsh(Lsym.toarray())
    vals = np.sort(np.real(vals))
    return float(vals[1]) if vals.size > 1 else float("nan")


def mean_effective_resistance(A, max_dense_nodes: int = 20000) -> tuple[float, float]:
    """Mean pairwise effective resistance via Kirchhoff index from the combinatorial-Laplacian
    spectrum: Kf = N * sum_{k>=2} 1/mu_k ; mean R = Kf / C(N,2). Returns (mean_R, Kf).
    Uses dense eigvalsh; returns (nan, nan) when N > max_dense_nodes."""
    import scipy.sparse as sp

    n = A.shape[0]
    if n > int(max_dense_nodes):
        return float("nan"), float("nan")
    L = (sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A).toarray().astype(np.float64)
    mu = np.sort(np.real(np.linalg.eigvalsh(L)))
    nz = mu[1:]                      # drop the single ~0 eigenvalue (assumes connected)
    kirchhoff = float(n * np.sum(1.0 / np.maximum(nz, 1e-9)))
    mean_r = float(kirchhoff / (n * (n - 1) / 2.0)) if n > 1 else float("nan")
    return mean_r, kirchhoff


def compute_graph_structure_metrics(graph, max_dense_nodes: int = 20000) -> list[dict]:
    """Per graph-level spectral gap + mean effective resistance. `graph` is a GraphBundle
    exposing .L0/.L1/.L2/.L3(/.L4), each a GraphLevel with .num_nodes and .edge_index [2,E]."""
    rows: list[dict] = []
    for name in ("L0", "L1", "L2", "L3", "L4"):
        level = getattr(graph, name, None)
        if level is None:
            continue
        num_nodes = int(level.num_nodes)
        edge_index = level.edge_index
        A = _sym_adjacency(edge_index, num_nodes)
        lambda2 = spectral_gap(A)
        mean_r, kirchhoff = mean_effective_resistance(A, max_dense_nodes=max_dense_nodes)
        rows.append({
            "level": name,
            "num_nodes": num_nodes,
            "num_edges": int(edge_index.shape[1]),
            "avg_degree": float(A.sum() / max(num_nodes, 1)),
            "spectral_gap_lambda2": lambda2,
            "mean_effective_resistance": mean_r,
            "kirchhoff_index": kirchhoff,
        })
    return rows

"""Time the LocalGraphAttention contraction formulations at production shapes.

Backs the ``ATTENTION_IMPL_DEFAULT`` choice in src/layers.py. All variants compute
identical values; they differ only in tensor layout and in which op performs the two
contractions (scores over head_dim, output over neighbours). Default shapes are the
real 2.5-degree blocks: batch 12, dim 160, 5 heads.

    python scripts/dev/bench_attention_impl.py            # eager variants
    python scripts/dev/bench_attention_impl.py --compile  # + torch.compile

Headline result on a B200 (L0, bf16, forward+backward):
    eager  elementwise    7.84 ms  3.25 GB   (default)
    eager  matmul        17.27 ms  2.97 GB   0.45x
    compiled elementwise  1.82 ms  1.36 GB   4.31x
The matmul contraction is a batched gemv (m=1) over B*N*H matrices, so no
tensor-core tile applies; fusing the elementwise form is what actually wins.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

DEVICE = "cuda"


def make_inputs(bsz, num_nodes, k, heads, head_dim, dtype):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    q = torch.randn(bsz, num_nodes, heads, head_dim, device=DEVICE, dtype=dtype, generator=g, requires_grad=True)
    key = torch.randn(bsz, num_nodes, heads, head_dim, device=DEVICE, dtype=dtype, generator=g, requires_grad=True)
    val = torch.randn(bsz, num_nodes, heads, head_dim, device=DEVICE, dtype=dtype, generator=g, requires_grad=True)
    src = torch.randint(0, num_nodes, (num_nodes, k), device=DEVICE, generator=g)
    # neighbour-major edge terms [N,k,H,d]; head-major variants permute them once
    # (cheap: they are per-node constants, not per-batch)
    ek = torch.randn(num_nodes, k, heads, head_dim, device=DEVICE, dtype=dtype, generator=g, requires_grad=True)
    ev = torch.randn(num_nodes, k, heads, head_dim, device=DEVICE, dtype=dtype, generator=g, requires_grad=True)
    eb = torch.randn(num_nodes, k, heads, device=DEVICE, dtype=dtype, generator=g, requires_grad=True)
    return q, key, val, src, ek, ev, eb


def v_reference(q, key, val, src, ek, ev, eb, scale):
    """[B,N,k,H,d] elementwise, the pre-refactor implementation."""
    k_src = key[:, src]
    v_src = val[:, src]
    scores = (q[:, :, None] * (k_src + ek[None])).sum(dim=-1) / scale + eb[None]
    attn = torch.softmax(scores, dim=2)
    return (attn[..., None] * (v_src + ev[None])).sum(dim=2)


def v_matmul_transpose(q, key, val, src, ek, ev, eb, scale):
    """Head-major via transpose(2,3) on the gathered tensors -- what shipped."""
    ekh, evh, ebh = ek.transpose(1, 2), ev.transpose(1, 2), eb.transpose(1, 2)
    k_src = key[:, src].transpose(2, 3) + ekh
    scores = torch.matmul(q.unsqueeze(-2), k_src.transpose(-1, -2)).squeeze(-2) / scale + ebh
    attn = torch.softmax(scores, dim=-1)
    v_src = val[:, src].transpose(2, 3) + evh
    return torch.matmul(attn.unsqueeze(-2), v_src).squeeze(-2)


def v_matmul_permute(q, key, val, src, ek, ev, eb, scale):
    """Head-major built by permuting BEFORE the gather, so no big transposed add."""
    ekh, evh, ebh = ek.permute(2, 0, 1, 3), ev.permute(2, 0, 1, 3), eb.permute(2, 0, 1)
    kp = key.permute(0, 2, 1, 3)                      # [B,H,N,d]
    vp = val.permute(0, 2, 1, 3)
    qp = q.permute(0, 2, 1, 3)
    k_src = kp[:, :, src] + ekh                       # [B,H,N,k,d]
    scores = torch.matmul(qp.unsqueeze(-2), k_src.transpose(-1, -2)).squeeze(-2) / scale + ebh
    attn = torch.softmax(scores, dim=-1)
    v_src = vp[:, :, src] + evh
    out = torch.matmul(attn.unsqueeze(-2), v_src).squeeze(-2)   # [B,H,N,d]
    return out.permute(0, 2, 1, 3)


def v_einsum_neighbour(q, key, val, src, ek, ev, eb, scale):
    """Keep the original layout; let einsum pick the contraction."""
    k_src = key[:, src] + ek[None]
    scores = torch.einsum("bnhd,bnkhd->bnkh", q, k_src) / scale + eb[None]
    attn = torch.softmax(scores, dim=2)
    v_src = val[:, src] + ev[None]
    return torch.einsum("bnkh,bnkhd->bnhd", attn, v_src)


def v_elementwise_headmajor(q, key, val, src, ek, ev, eb, scale):
    """Head-major layout but the elementwise reduction -- isolates layout from matmul."""
    ekh, evh, ebh = ek.transpose(1, 2), ev.transpose(1, 2), eb.transpose(1, 2)
    k_src = key[:, src].transpose(2, 3) + ekh
    scores = (q.unsqueeze(-2) * k_src).sum(dim=-1) / scale + ebh
    attn = torch.softmax(scores, dim=-1)
    v_src = val[:, src].transpose(2, 3) + evh
    return (attn.unsqueeze(-1) * v_src).sum(dim=-2)


VARIANTS = [
    ("reference   [B,N,k,H,d] mul+sum", v_reference),
    ("matmul      transpose (shipped)", v_matmul_transpose),
    ("matmul      permute-then-gather", v_matmul_permute),
    ("einsum      neighbour-major", v_einsum_neighbour),
    ("mul+sum     head-major", v_elementwise_headmajor),
]


def bench(fn, args, *, backward, iters=30, warmup=10):
    for _ in range(warmup):
        out = fn(*args)
        if backward:
            out.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        if backward:
            out.sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3, torch.cuda.max_memory_allocated() / 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="also time torch.compile'd variants")
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    variants = list(VARIANTS)
    if args.compile:
        variants += [
            ("compiled   reference", torch.compile(v_reference, dynamic=False)),
            ("compiled   matmul", torch.compile(v_matmul_transpose, dynamic=False)),
        ]
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    for shape_name, bsz, num_nodes, k in (("L0", args.batch, 10368, 8), ("L3", args.batch, 162, 24)):
        heads, head_dim = 5, 32
        scale = math.sqrt(head_dim)
        for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
            args = make_inputs(bsz, num_nodes, k, heads, head_dim, dtype)
            reference_out = v_reference(*args, scale).float()
            print(f"\n=== {shape_name} {dtype_name} | B={bsz} N={num_nodes} k={k} H={heads} d={head_dim} ===")
            print(f"{'variant':36} {'fwd ms':>8} {'f+b ms':>8} {'peak GB':>8} {'max|err|':>10}")
            base_fb = None
            for label, fn in variants:
                try:
                    out = fn(*args, scale).float()
                    err = (out - reference_out).abs().max().item()
                    f_ms, _ = bench(fn, (*args, scale), backward=False)
                    fb_ms, mem = bench(fn, (*args, scale), backward=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"{label:36} {'OOM':>8}")
                    torch.cuda.empty_cache()
                    continue
                base_fb = base_fb if base_fb is not None else fb_ms
                print(f"{label:36} {f_ms:8.3f} {fb_ms:8.3f} {mem:8.2f} {err:10.2e}"
                      f"   {base_fb / fb_ms:5.2f}x vs reference")
                torch.cuda.empty_cache()
            del args
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

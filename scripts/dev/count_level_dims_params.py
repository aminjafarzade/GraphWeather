#!/usr/bin/env python
"""Analytic trainable-parameter counter for per-level widths (no torch needed).

Mirrors the module inventory that src/models.py + src/processor.py build for the
standard 4-level L3 layout in grid mode (edge_encoding off, l0_refine attention,
plain Linear embed/head), using the per-block closed form

    block(d, h) = 12*d^2 + 27*d + 7*h

(q/k/v/out projections 4*(d^2+d), edge_k/edge_v 2*(6d+d), edge_bias 6h+h,
two LayerNorms 2*2d, MLP 4d^2+4d + 4d^2+d).

Examples (the two counts pinned by tests/test_level_dims.py):
    python scripts/dev/count_level_dims_params.py 192 96 48 24 --heads 6 6 6 6
    -> 2,852,257
    python scripts/dev/count_level_dims_params.py 160 160 160 160 --heads 5 5 5 5
    -> 3,844,932
"""

from __future__ import annotations

import argparse


def attention_block(dim: int, heads: int) -> int:
    return 12 * dim * dim + 27 * dim + 7 * heads


def mean_max_pool(fine_dim: int, coarse_dim: int) -> int:
    # proj: Linear(2*fine, coarse)
    return 2 * fine_dim * coarse_dim + coarse_dim


def parent_unpool_fuse(fine_dim: int, coarse_dim: int) -> int:
    # fuse: Linear(fine + coarse, fine) -> GELU -> Linear(fine, fine)
    return (fine_dim + coarse_dim) * fine_dim + fine_dim + fine_dim * fine_dim + fine_dim


def count_params(
    dims: list[int],
    heads: list[int],
    *,
    in_features: int,
    out_channels: int,
    encoder_blocks: int,
    decoder_blocks: int,
    l0_blocks: int,
    l1_blocks: int,
    l2_blocks: int,
    l3_blocks: int,
    l2_refine_after_l3_blocks: int,
    l1_refine_blocks: int,
    l0_refine_blocks: int,
) -> int:
    d0, d1, d2, d3 = dims
    h0, h1, h2, h3 = heads
    blocks_per_level = [
        # encoder/decoder run on L0 alongside the L0 processor blocks.
        (d0, h0, encoder_blocks + decoder_blocks + l0_blocks + l0_refine_blocks),
        (d1, h1, l1_blocks + l1_refine_blocks),
        (d2, h2, l2_blocks + l2_refine_after_l3_blocks),
        (d3, h3, l3_blocks),
    ]
    total = sum(attention_block(d, h) * n for d, h, n in blocks_per_level)
    total += mean_max_pool(d0, d1) + mean_max_pool(d1, d2) + mean_max_pool(d2, d3)
    total += parent_unpool_fuse(d2, d3) + parent_unpool_fuse(d1, d2) + parent_unpool_fuse(d0, d1)
    total += in_features * d0 + d0  # embed: Linear(in_features, d0)
    total += d0 * out_channels + out_channels  # head: Linear(d0, out_channels)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dims", type=int, nargs=4, metavar="DIM", help="widths for L0 L1 L2 L3")
    parser.add_argument("--heads", type=int, nargs=4, required=True, metavar="H", help="heads for L0 L1 L2 L3")
    parser.add_argument("--in-features", type=int, default=134, help="embed input channels (default: 134)")
    parser.add_argument("--out-channels", type=int, default=67, help="head output channels (default: 67)")
    parser.add_argument("--encoder-blocks", type=int, default=1)
    parser.add_argument("--decoder-blocks", type=int, default=1)
    parser.add_argument("--l0-blocks", type=int, default=2)
    parser.add_argument("--l1-blocks", type=int, default=2)
    parser.add_argument("--l2-blocks", type=int, default=1)
    parser.add_argument("--l3-blocks", type=int, default=1)
    parser.add_argument("--l2-refine-after-l3-blocks", type=int, default=1)
    parser.add_argument("--l1-refine-blocks", type=int, default=1)
    parser.add_argument("--l0-refine-blocks", type=int, default=1)
    args = parser.parse_args()

    for level_idx, (dim, head_count) in enumerate(zip(args.dims, args.heads)):
        if dim < 1 or head_count < 1 or dim % head_count != 0:
            parser.error(
                f"level {level_idx}: dim={dim} must be a positive multiple of heads={head_count}"
            )

    total = count_params(
        args.dims,
        args.heads,
        in_features=args.in_features,
        out_channels=args.out_channels,
        encoder_blocks=args.encoder_blocks,
        decoder_blocks=args.decoder_blocks,
        l0_blocks=args.l0_blocks,
        l1_blocks=args.l1_blocks,
        l2_blocks=args.l2_blocks,
        l3_blocks=args.l3_blocks,
        l2_refine_after_l3_blocks=args.l2_refine_after_l3_blocks,
        l1_refine_blocks=args.l1_refine_blocks,
        l0_refine_blocks=args.l0_refine_blocks,
    )
    head_dims = ", ".join(str(d // h) for d, h in zip(args.dims, args.heads))
    print(f"dims={args.dims} heads={args.heads} (head_dim {head_dims})")
    print(f"trainable parameters: {total:,} ({total})")


if __name__ == "__main__":
    main()

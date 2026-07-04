#!/usr/bin/env python3
"""Synthetic masked-attention fixture for the #46 deploy contract (stdlib only, no torch).

Hand-authors an ncnn graph containing a single `MultiHeadAttention` layer with an additive
attention-mask input (param 5=1) and computes golden outputs with a pure-Python replica of
ncnn's exact forward loops (thirdparty/ncnn/src/layer/multiheadattention.cpp): torch-style
row-major weights, scale on the Q affine, per-head split over embed rows, 2D (src x dst)
additive mask shared across heads.

This is the ncnn-side half of #258's round-trip spike, runnable without torch/pnnx: it proves
the runtime executes masked attention and that `NcnnRunner.run_inference_multi` deploys it.
(What it deliberately does NOT prove: that pnnx emits this graph from a torch module — that
half of the spike still needs the Python toolchain.)

Cases:
  1. all_present    — zero mask, base entities.
  2. padded_junk_a  — slot 2 masked out (-1e9 column), slot 2 content = junk A.
  3. padded_junk_b  — same mask, slot 2 content = junk B. Rows 0..1 of the output must match
                      case 2 exactly: mask invariance, asserted by the GDScript golden test.

Regenerate: python3 scripts/make_synthetic_attention.py
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

N = 3          # entities (seq len)
D = 4          # embed_dim
HEADS = 2
DHEAD = D // HEADS
SCALE = 1.0 / math.sqrt(DHEAD)
NEG = -1e9

OUT_PARAM = Path("models/synthetic_attention.ncnn.param")
OUT_BIN = Path("models/synthetic_attention.ncnn.bin")
OUT_GOLDEN = Path("models/synthetic_attention_golden.json")


def lcg(seed: int):
    state = seed
    while True:
        state = (state * 1103515245 + 12345) % (1 << 31)
        yield ((state >> 8) % 65 - 32) / 64.0  # multiples of 1/64 in [-0.5, 0.5] — fp32-exact


def take(gen, n: int) -> list[float]:
    return [next(gen) for _ in range(n)]


def f32(x: float) -> float:
    """Round-trip through float32 so the golden matches ncnn's storage precision."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def affine(x_rows: list[list[float]], w: list[float], b: list[float], scale: float) -> list[list[float]]:
    """ncnn layout: result[j][i] = (b[j] + sum_k x[i][k]*w[j*len(x_row)+k]) * scale."""
    in_dim = len(x_rows[0])
    out = [[0.0] * len(x_rows) for _ in range(D)]
    for i, row in enumerate(x_rows):
        for j in range(D):
            s = b[j] + sum(row[k] * w[j * in_dim + k] for k in range(in_dim))
            out[j][i] = s * scale
    return out


def mha_forward(x_rows, mask_rows, qw, qb, kw, kb, vw, vb, ow, ob):
    q_aff = affine(x_rows, qw, qb, SCALE)
    k_aff = affine(x_rows, kw, kb, 1.0)
    v_aff = affine(x_rows, vw, vb, 1.0)
    ctx = [[0.0] * D for _ in range(N)]
    for h in range(HEADS):
        rows = range(h * DHEAD, (h + 1) * DHEAD)
        for i in range(N):
            scores = []
            for j in range(N):
                s = sum(q_aff[l][i] * k_aff[l][j] for l in rows) + mask_rows[i][j]
                scores.append(s)
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            tot = sum(exps)
            attn = [e / tot for e in exps]
            for li, l in enumerate(rows):
                ctx[i][h * DHEAD + li] = sum(attn[j] * v_aff[l][j] for j in range(N))
    out = []
    for i in range(N):
        row = []
        for j in range(D):  # qdim == D here
            row.append(ob[j] + sum(ctx[i][k] * ow[j * D + k] for k in range(D)))
        out.append(row)
    return out


def main() -> None:
    gen = lcg(20260704)
    qw, qb = take(gen, D * D), take(gen, D)
    kw, kb = take(gen, D * D), take(gen, D)
    vw, vb = take(gen, D * D), take(gen, D)
    ow, ob = take(gen, D * D), take(gen, D)

    base = [take(gen, D) for _ in range(N)]
    junk_a = base[0:2] + [[9.0, -9.0, 9.0, -9.0]]
    junk_b = base[0:2] + [[-7.0, 7.0, -7.0, 7.0]]
    zero_mask = [[0.0] * N for _ in range(N)]
    pad2_mask = [[0.0, 0.0, NEG] for _ in range(N)]

    cases = []
    for name, x_rows, mask_rows in [
        ("all_present", base, zero_mask),
        ("padded_junk_a", junk_a, pad2_mask),
        ("padded_junk_b", junk_b, pad2_mask),
    ]:
        expected = mha_forward(x_rows, mask_rows, qw, qb, kw, kb, vw, vb, ow, ob)
        cases.append({
            "name": name,
            "x": [f32(v) for row in x_rows for v in row],
            "mask": [f32(v) for row in mask_rows for v in row],
            "expected": [f32(v) for row in expected for v in row],
        })

    param = "\n".join([
        "7767517",
        "3 3",
        "Input x 0 1 x 0=%d 1=%d" % (D, N),
        "Input mask 0 1 mask 0=%d 1=%d" % (N, N),
        "MultiHeadAttention mha 2 1 x mask out 0=%d 1=%d 2=%d 5=1" % (D, HEADS, D * D),
    ]) + "\n"
    OUT_PARAM.write_text(param)

    def packed(data, tagged):
        blob = b"" if not tagged else struct.pack("<I", 0)
        return blob + struct.pack("<%df" % len(data), *data)

    bin_blob = b"".join([
        packed(qw, True), packed(qb, False),
        packed(kw, True), packed(kb, False),
        packed(vw, True), packed(vb, False),
        packed(ow, True), packed(ob, False),
    ])
    OUT_BIN.write_bytes(bin_blob)

    OUT_GOLDEN.write_text(json.dumps({
        "n": N, "embed_dim": D, "num_heads": HEADS,
        "x_shape": [D, N], "mask_shape": [N, N], "out_shape": [D, N],
        "cases": cases,
    }, indent=1))
    print("Wrote:", OUT_PARAM, OUT_BIN, OUT_GOLDEN)


if __name__ == "__main__":
    main()

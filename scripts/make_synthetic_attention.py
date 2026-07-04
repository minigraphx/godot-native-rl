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


# ---------------------------------------------------------------------------
# Fixture 2 (#46 single-input contract): the FULL entity encoder as ONE graph —
# flat obs in, pooled embedding out. Crop splits entities/flags, Reshape rows,
# Gemm(+ReLU) is the EntityEmbedding, the additive mask is BUILT IN-GRAPH from
# the presence flags ((1-p)*-1e9, tiled to src x dst), MultiHeadAttention runs
# masked, and the masked mean-pool is p^T*att (2-input Gemm) / max(sum p, 1).
# Proves the spec's PREFERRED deploy shape (single run_inference call) without
# pnnx — and doubles as the direct-export prototype for the encoder.
# ---------------------------------------------------------------------------

F = 2  # per-entity features in the encoder fixture

ENC_PARAM = Path("models/synthetic_entity_encoder.ncnn.param")
ENC_BIN = Path("models/synthetic_entity_encoder.ncnn.bin")
ENC_GOLDEN = Path("models/synthetic_entity_encoder_golden.json")


def encoder_forward(flat, emb_w, emb_b, qw, qb, kw, kb, vw, vb, ow, ob):
    ents = [flat[i * F:(i + 1) * F] for i in range(N)]
    flags = flat[N * F:N * F + N]
    emb = []
    for row in ents:
        out_row = []
        for j in range(D):
            s = emb_b[j] + sum(row[k] * emb_w[j * F + k] for k in range(F))
            out_row.append(max(s, 0.0))
        emb.append(out_row)
    mask_row = [(1.0 - p) * NEG for p in flags]
    mask_rows = [mask_row for _ in range(N)]
    att = mha_forward(emb, mask_rows, qw, qb, kw, kb, vw, vb, ow, ob)
    denom = max(sum(flags), 1.0)
    return [sum(flags[j] * att[j][k] for j in range(N)) / denom for k in range(D)]


def write_encoder_fixture(gen) -> None:
    emb_w, emb_b = take(gen, D * F), take(gen, D)
    qw, qb = take(gen, D * D), take(gen, D)
    kw, kb = take(gen, D * D), take(gen, D)
    vw, vb = take(gen, D * D), take(gen, D)
    ow, ob = take(gen, D * D), take(gen, D)

    def flat_obs(ent_rows, flags):
        padded = ent_rows + [[0.0] * F] * (N - len(ent_rows))
        return [v for row in padded for v in row] + flags

    real2 = [take(gen, F), take(gen, F)]
    cases = []
    for name, rows, flags in [
        ("all_present", real2 + [take(gen, F)], [1.0, 1.0, 1.0]),
        ("two_present", real2, [1.0, 1.0, 0.0]),
        # Same two real entities, junk in the padded slot: output must equal two_present.
        ("two_present_junk", real2 + [[9.0, -9.0]], [1.0, 1.0, 0.0]),
    ]:
        flat = flat_obs(rows, flags)
        cases.append({
            "name": name,
            "obs": [f32(v) for v in flat],
            "expected": [f32(v) for v in encoder_forward(flat, emb_w, emb_b, qw, qb, kw, kb, vw, vb, ow, ob)],
        })

    nf = N * F
    # ncnn graph rule (pnnx always honors it): ONE consumer per blob — every fan-out goes
    # through an explicit Split layer, or lightmode blob recycling breaks the forward chain.
    lines = [
        "7767517",
        "0 0",
        "Input flat 0 1 flat 0=%d" % (nf + N),
        "Split flat_split 1 2 flat flat0 flat1",
        "Crop entflat 1 1 flat0 entflat 0=0 3=%d" % nf,
        "Crop flags 1 1 flat1 flags 0=%d 3=%d" % (nf, N),
        "Split flags_split 1 2 flags flags0 flags1",
        "Reshape ent 1 1 entflat ent 0=%d 1=%d" % (F, N),
        "Gemm emb0 1 1 ent emb0 2=0 3=1 4=0 5=1 6=1 8=%d 9=%d 10=4" % (D, F),
        "ReLU emb 1 1 emb0 emb",
        "Reshape flagrow 1 1 flags0 flagrow 0=%d 1=1" % N,
        "Split flagrow_split 1 2 flagrow flagrow0 flagrow1",
        "BinaryOp inv 1 1 flagrow0 inv 0=7 1=1 2=1.0",
        "BinaryOp negmask 1 1 inv negmask 0=2 1=1 2=-1000000000.0",
        "Tile mask 1 1 negmask mask 0=0 1=%d" % N,
        "MultiHeadAttention att 2 1 emb mask att 0=%d 1=%d 2=%d 5=1" % (D, HEADS, D * D),
        "Gemm pool 2 1 flagrow1 att pool 1=0.0",
        "Reduction denom0 1 1 flags1 denom0 0=0 1=1",
        "BinaryOp denom 1 1 denom0 denom 0=4 1=1 2=1.0",
        "BinaryOp out 2 1 pool denom out 0=3",
    ]
    n_layers = len(lines) - 2
    blobs = set()
    for l in lines[2:]:
        toks = l.split()
        nin, nout = int(toks[2]), int(toks[3])
        blobs.update(toks[4 + nin:4 + nin + nout])
    lines[1] = "%d %d" % (n_layers, len(blobs))
    ENC_PARAM.write_text("\n".join(lines) + "\n")

    def packed(data, tagged):
        blob = b"" if not tagged else struct.pack("<I", 0)
        return blob + struct.pack("<%df" % len(data), *data)

    ENC_BIN.write_bytes(b"".join([
        packed(emb_w, True), packed(emb_b, True),  # Gemm B (transB: KxN load) + C (type 4) — both tagged
        packed(qw, True), packed(qb, False),
        packed(kw, True), packed(kb, False),
        packed(vw, True), packed(vb, False),
        packed(ow, True), packed(ob, False),
    ]))

    ENC_GOLDEN.write_text(json.dumps({
        "n": N, "f": F, "embed_dim": D, "obs_size": nf + N,
        "cases": cases,
    }, indent=1))
    print("Wrote:", ENC_PARAM, ENC_BIN, ENC_GOLDEN)


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
    write_encoder_fixture(gen)


if __name__ == "__main__":
    main()

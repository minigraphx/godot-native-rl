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
  2. padded_junk_a  — slot 2 masked out (NEG column), slot 2 content = junk A.
  3. padded_junk_b  — same mask, slot 2 content = junk B. Rows 0..1 of the output must match
                      case 2 exactly: mask invariance, asserted by the GDScript golden test.

Regenerate: python3 scripts/make_synthetic_attention.py — and the committed fixtures are
byte-pinned by test/python/test_export_statedict_to_ncnn.py (regenerates into a tempdir and
compares), so generator edits can't silently drift from the committed artifacts (#319). The
two fixtures use INDEPENDENT LCG seeds, so editing one never changes the other's bytes.
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_statedict_to_ncnn as sd  # the production writer — single source of graph truth

N = 3          # entities (seq len)
D = 4          # embed_dim
HEADS = 2
DHEAD = D // HEADS
SCALE = 1.0 / math.sqrt(DHEAD)
NEG = sd.FP16_SAFE_NEG_MASK  # -6e4, fp16-finite (#322): -1e9 turns -inf on ARM's default fp16 path

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
        # EVERY entity absent (#322): a representable EntitySensor obs (empty group). The whole
        # mask row is NEG; with the fp16-finite constant the softmax stays defined (uniform) and
        # the clamped-denominator pool yields exact zeros — pinned on the real ncnn forward so a
        # regression back to an overflowing constant would NaN this case on fp16 targets.
        ("none_present", [take(gen, F)], [0.0, 0.0, 0.0]),
    ]:
        flat = flat_obs(rows, flags)
        cases.append({
            "name": name,
            "obs": [f32(v) for v in flat],
            "expected": [f32(v) for v in encoder_forward(flat, emb_w, emb_b, qw, qb, kw, kb, vw, vb, ow, ob)],
        })

    nf = N * F
    weights = {
        "emb_w": emb_w, "emb_b": emb_b,
        "q_w": qw, "q_b": qb, "k_w": kw, "k_b": kb,
        "v_w": vw, "v_b": vb, "out_w": ow, "out_b": ob,
    }
    layers = sd.attention_policy_layers(N, F, D, HEADS, weights)
    ENC_PARAM.write_text(sd.ncnn_param_text(layers))

    ENC_BIN.write_bytes(sd.ncnn_bin_bytes(layers))
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

    # The production writers are the single source of ncnn-format truth (#316) — no hand-rolled
    # magic/tag/param text here.
    layers = [
        sd.input_layer_2d("x", D, N),
        sd.input_layer_2d("mask", N, N),
        sd.multihead_attention_layer("mha", "x", "mask", "out", D, HEADS,
                                     qw, qb, kw, kb, vw, vb, ow, ob),
    ]
    sd.validate_single_consumer(layers)
    OUT_PARAM.write_text(sd.ncnn_param_text(layers))
    OUT_BIN.write_bytes(sd.ncnn_bin_bytes(layers))

    OUT_GOLDEN.write_text(json.dumps({
        "n": N, "embed_dim": D, "num_heads": HEADS,
        "x_shape": [D, N], "mask_shape": [N, N], "out_shape": [D, N],
        "cases": cases,
    }, indent=1))
    print("Wrote:", OUT_PARAM, OUT_BIN, OUT_GOLDEN)
    # Independent seed (#319): fixture 2 must not draw from fixture 1's stream tail, or editing
    # fixture 1 silently changes fixture 2's regenerated bytes.
    write_encoder_fixture(lcg(20260705))


if __name__ == "__main__":
    main()

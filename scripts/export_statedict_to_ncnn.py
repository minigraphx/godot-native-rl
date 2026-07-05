#!/usr/bin/env python3
"""Direct PyTorch module -> ncnn (.param/.bin) for simple feed-forward policies.

The dependency-free escape hatch: convert an MLP policy straight to ncnn's native
format with NO ONNX, NO TorchScript, and NO pnnx -- just introspect the module and
write the two files. Immune to the dynamo-exporter / torch.jit / pnnx toolchains (and
their deprecation churn), at the cost of only covering layer types we map by hand.

v1 scope (feed-forward stacks): Input, Linear (-> InnerProduct), ReLU, Tanh, Sigmoid,
Flatten. Anything else raises (fail loud) -- use the ONNX/pnnx path for it. Conv/pool
are a planned follow-up. NOT a general pnnx replacement.

The mapping is deterministic (a layer-type -> ncnn-layer lookup + bit-exact weight
copy), so correctness is a unit-testable property, NOT something to approximate -- the
format writer below is pure and fully tested; `verify_ncnn_parity.py` gates the real
round-trip. Run under .venv-train (torch). The module walk targets a flat Sequential-
style stack; composing an SB3 actor into one is version-sensitive (see main()).

Usage (library): export_module_to_ncnn(model, input_dim, "models/policy")
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Iterable

NCNN_MAGIC = "7767517"
_FP32_TAG = struct.pack("<I", 0)  # ncnn ModelBin type-0 weight tag: plain float32 follows


# ---- pure format writer (no torch; the format-critical, fully-tested core) ----

def fmt_param(value) -> str:
    """Format an ncnn `.param` value: ints bare, floats with a decimal/exponent.

    ncnn parses a token as float only if it looks like one ('.' or 'e'), so a float
    param (e.g. a ReLU slope) must keep its dot -- `0` would be read as an int.
    """
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def count_blobs(layers: list[dict]) -> int:
    """Number of distinct blob names across all layers' bottoms + tops."""
    blobs: set[str] = set()
    for layer in layers:
        blobs.update(layer["bottoms"])
        blobs.update(layer["tops"])
    return len(blobs)


def ncnn_param_text(layers: list[dict]) -> str:
    """Render the ncnn `.param` text for a list of layer dicts.

    Each layer dict: {type, name, bottoms, tops, params{id: value}}. Header is the
    magic, then `layer_count blob_count`, then one line per layer:
    `type name n_in n_out in_blobs... out_blobs... id=value...` (params sorted by id).
    """
    lines = [NCNN_MAGIC, f"{len(layers)} {count_blobs(layers)}"]
    for layer in layers:
        parts = [layer["type"], layer["name"], str(len(layer["bottoms"])), str(len(layer["tops"]))]
        parts += list(layer["bottoms"]) + list(layer["tops"])
        for pid in sorted(layer.get("params", {})):
            parts.append(f"{pid}={fmt_param(layer['params'][pid])}")
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def _pack_floats(data: Iterable[float]) -> bytes:
    floats = [float(x) for x in data]
    return struct.pack(f"<{len(floats)}f", *floats)


def ncnn_bin_bytes(layers: list[dict]) -> bytes:
    """Render the ncnn `.bin` for a list of layer dicts.

    Concatenates each layer's weight blobs in order. A blob marked `tagged` (ncnn
    ModelBin type-0, e.g. InnerProduct/Convolution weights) gets the 4-byte fp32 tag
    prefix; untagged blobs (type-1, e.g. bias) are raw float32. Layout per weight
    array must already match ncnn's expectation (Linear weight is [out, in], same as
    torch -- copied verbatim).
    """
    out = bytearray()
    for layer in layers:
        for data, tagged in layer.get("weights", []):
            if tagged:
                out += _FP32_TAG
            out += _pack_floats(data)
    return bytes(out)


# ---- pure layer builders ----

def input_layer(blob: str, width: int) -> dict:
    return {"type": "Input", "name": blob, "bottoms": [], "tops": [blob], "params": {0: int(width)}}


def input_layer_2d(blob: str, width: int, height: int) -> dict:
    """2D Mat input (w=features, h=sequence) — MultiHeadAttention's x/mask blob shape."""
    return {"type": "Input", "name": blob, "bottoms": [], "tops": [blob],
            "params": {0: int(width), 1: int(height)}}


def _expect_len(what: str, data, expected: int) -> None:
    """ncnn's ModelBin reads blobs sequentially by DECLARED size, so a mis-sized list shifts
    every later blob and the net loads fine but computes garbage (#323). Fail loud instead."""
    if len(data) != expected:
        raise ValueError("%s has %d floats, expected %d — the .bin would load but every "
                         "subsequent blob would be misaligned (silent wrong logits)"
                         % (what, len(data), expected))


def linear_layer(name: str, bottom: str, top: str, weight: list[float], bias: list[float] | None,
                 in_features: int, out_features: int) -> dict:
    """Build an InnerProduct layer. `weight` is the flat [out, in] row-major list."""
    _expect_len("linear '%s' weight" % name, weight, in_features * out_features)
    weights = [(weight, True)]
    if bias is not None:
        _expect_len("linear '%s' bias" % name, bias, out_features)
        weights.append((bias, False))
    return {
        "type": "InnerProduct", "name": name, "bottoms": [bottom], "tops": [top],
        "params": {0: int(out_features), 1: 1 if bias is not None else 0, 2: in_features * out_features},
        "weights": weights,
    }


def activation_layer(ncnn_type: str, name: str, bottom: str, top: str, params: dict | None = None) -> dict:
    return {"type": ncnn_type, "name": name, "bottoms": [bottom], "tops": [top], "params": params or {}}


# ---- attention-encoder layer builders (#46; the pnnx-independent export path) ----
# Param IDs and weight layouts mirror the vendored ncnn sources exactly; the whole graph shape
# below is byte-pinned to committed fixtures that the GDScript golden suite executes through the
# real NcnnRunner at zero error (test_attention_golden_inference.gd).

def split_layer(name: str, bottom: str, tops: list[str]) -> dict:
    """ncnn permits ONE consumer per blob: every fan-out must pass through a Split (pnnx inserts
    them silently; hand-authored graphs that skip this die at forward with no log)."""
    return {"type": "Split", "name": name, "bottoms": [bottom], "tops": list(tops), "params": {}}


def crop_layer(name: str, bottom: str, top: str, woffset: int, outw: int) -> dict:
    return {"type": "Crop", "name": name, "bottoms": [bottom], "tops": [top],
            "params": {0: int(woffset), 3: int(outw)}}


def reshape_layer(name: str, bottom: str, top: str, w: int, h: int) -> dict:
    return {"type": "Reshape", "name": name, "bottoms": [bottom], "tops": [top],
            "params": {0: int(w), 1: int(h)}}


def gemm_const_b_layer(name: str, bottom: str, top: str, weight: list[float], bias: list[float],
                       in_features: int, out_features: int) -> dict:
    """A(M,in) x W(out,in)^T + bias-row -> (M,out). transB=1 with torch-layout weights; bias is a
    constant C of broadcast type 4 (one row over M). BOTH constant blobs are fp32-tagged."""
    _expect_len("gemm '%s' weight" % name, weight, in_features * out_features)
    _expect_len("gemm '%s' bias" % name, bias, out_features)
    return {
        "type": "Gemm", "name": name, "bottoms": [bottom], "tops": [top],
        "params": {2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 8: int(out_features), 9: int(in_features), 10: 4},
        "weights": [(weight, True), (bias, True)],
    }


def gemm_two_input_layer(name: str, bottom_a: str, bottom_b: str, top: str) -> dict:
    """Plain A x B on two runtime blobs (beta=0: no C term)."""
    return {"type": "Gemm", "name": name, "bottoms": [bottom_a, bottom_b], "tops": [top],
            "params": {1: 0.0}}


def binaryop_scalar_layer(name: str, bottom: str, top: str, op: int, scalar: float) -> dict:
    """One-blob BinaryOp with scalar operand. op: 0 ADD 1 SUB 2 MUL 3 DIV 4 MAX 5 MIN 7 RSUB."""
    return {"type": "BinaryOp", "name": name, "bottoms": [bottom], "tops": [top],
            "params": {0: int(op), 1: 1, 2: float(scalar)}}


def binaryop_layer(name: str, bottom_a: str, bottom_b: str, top: str, op: int) -> dict:
    return {"type": "BinaryOp", "name": name, "bottoms": [bottom_a, bottom_b], "tops": [top],
            "params": {0: int(op)}}


def tile_layer(name: str, bottom: str, top: str, axis: int, tiles: int) -> dict:
    return {"type": "Tile", "name": name, "bottoms": [bottom], "tops": [top],
            "params": {0: int(axis), 1: int(tiles)}}


def reduction_sum_all_layer(name: str, bottom: str, top: str) -> dict:
    return {"type": "Reduction", "name": name, "bottoms": [bottom], "tops": [top],
            "params": {0: 0, 1: 1}}


def multihead_attention_layer(name: str, bottom_x: str, bottom_mask: str, top: str,
                              embed_dim: int, num_heads: int,
                              q_w: list[float], q_b: list[float], k_w: list[float], k_b: list[float],
                              v_w: list[float], v_b: list[float], out_w: list[float], out_b: list[float]) -> dict:
    """Self-attention with an additive attn_mask bottom (param 5=1). Weight order/layout per
    MultiHeadAttention::load_model: q/k/v/out as (out x in) row-major fp32-tagged weights, each
    followed by its RAW (untagged) bias."""
    if int(embed_dim) % int(num_heads) != 0:
        raise ValueError("embed_dim %d not divisible by num_heads %d — ncnn truncates "
                         "embed_dim_per_head and silently drops dimensions from attention"
                         % (embed_dim, num_heads))
    for label, w, b in [("q", q_w, q_b), ("k", k_w, k_b), ("v", v_w, v_b), ("out", out_w, out_b)]:
        _expect_len("mha '%s' %s weight" % (name, label), w, int(embed_dim) * int(embed_dim))
        _expect_len("mha '%s' %s bias" % (name, label), b, int(embed_dim))
    return {
        "type": "MultiHeadAttention", "name": name, "bottoms": [bottom_x, bottom_mask], "tops": [top],
        "params": {0: int(embed_dim), 1: int(num_heads), 2: int(embed_dim) * int(embed_dim), 5: 1},
        "weights": [(q_w, True), (q_b, False), (k_w, True), (k_b, False),
                    (v_w, True), (v_b, False), (out_w, True), (out_b, False)],
    }


def validate_single_consumer(layers: list[dict]) -> None:
    """Fail loud on the silent killer: a blob consumed by more than one layer (needs a Split)."""
    consumers: dict[str, list[str]] = {}
    for layer in layers:
        for b in layer["bottoms"]:
            consumers.setdefault(b, []).append(layer["name"])
    multi = {b: c for b, c in consumers.items() if len(c) > 1}
    if multi:
        raise ValueError(
            "blob(s) with multiple consumers (insert Split layers): %s" % multi)


# fp16-FINITE mask constant (#322): -1e9 overflows fp16 (max ±65504) to -inf, and a softmax row
# that is ALL -inf (every entity absent — a representable EntitySensor obs) computes
# exp(-inf - -inf) = NaN. ncnn enables fp16 storage/arithmetic by default on ARM, so the NaN
# only appears on the mobile/edge deploys CI's fp32-x86 goldens can't see. -6e4 stays finite in
# fp16 while still dominating any post-embedding logit scale.
FP16_SAFE_NEG_MASK = -6e4


def attention_policy_layers(n_entities: int, feat: int, embed_dim: int, num_heads: int,
                            weights: dict, head_dims: list[int] | None = None,
                            neg_mask: float = FP16_SAFE_NEG_MASK) -> list[dict]:
    """The proven single-input entity-encoder graph (#46), optionally followed by an MLP head.

    Input blob `in0` = the EntitySensor flat obs [n*feat entities][n presence flags]; output
    blob `out0` = the pooled embedding, or the MLP head's logits when head_dims is given.

    `weights` keys (plain float lists, torch .tolist() layout):
      emb_w (embed_dim x feat), emb_b, q_w/q_b/k_w/k_b/v_w/v_b/out_w/out_b (embed_dim square),
      and for each head layer i: head{i}_w (out x in), head{i}_b. head_dims are the head's
      OUTPUT sizes; ReLU between head layers, none after the last (logits).
    """
    n, f, d = int(n_entities), int(feat), int(embed_dim)
    nf = n * f
    layers = [
        input_layer("in0", nf + n),
        split_layer("in0_split", "in0", ["in0_a", "in0_b"]),
        crop_layer("entflat", "in0_a", "entflat", 0, nf),
        crop_layer("flags", "in0_b", "flags", nf, n),
        split_layer("flags_split", "flags", ["flags_a", "flags_b"]),
        reshape_layer("ent", "entflat", "ent", f, n),
        gemm_const_b_layer("emb0", "ent", "emb0", weights["emb_w"], weights["emb_b"], f, d),
        activation_layer("ReLU", "emb", "emb0", "emb", {0: 0.0}),
        reshape_layer("flagrow", "flags_a", "flagrow", n, 1),
        split_layer("flagrow_split", "flagrow", ["flagrow_a", "flagrow_b"]),
        binaryop_scalar_layer("inv", "flagrow_a", "inv", 7, 1.0),          # 1 - p
        binaryop_scalar_layer("negmask", "inv", "negmask", 2, neg_mask),   # * neg_mask (fp16-finite)
        tile_layer("mask", "negmask", "mask", 0, n),                       # (n,1) -> (n,n)
        multihead_attention_layer("att", "emb", "mask", "att", d, num_heads,
                                  weights["q_w"], weights["q_b"], weights["k_w"], weights["k_b"],
                                  weights["v_w"], weights["v_b"], weights["out_w"], weights["out_b"]),
        gemm_two_input_layer("pool", "flagrow_b", "att", "pool"),          # p^T x att -> (1,d)
        reduction_sum_all_layer("denom0", "flags_b", "denom0"),
        binaryop_scalar_layer("denom", "denom0", "denom", 4, 1.0),         # max(sum p, 1)
        binaryop_layer("pooled", "pool", "denom", "pooled", 3),            # divide
    ]
    prev = "pooled"
    for i, out_dim in enumerate(head_dims or []):
        in_dim = d if i == 0 else int((head_dims or [])[i - 1])
        name = "head%d" % i
        layers.append(linear_layer(name, prev, name, weights["%s_w" % name], weights["%s_b" % name],
                                   in_dim, int(out_dim)))
        prev = name
        if i < len(head_dims or []) - 1:
            act = "head%d_relu" % i
            layers.append(activation_layer("ReLU", act, prev, act, {0: 0.0}))
            prev = act
    layers[-1]["tops"] = ["out0"]
    validate_single_consumer(layers)
    return layers


# ---- torch-facing module walk (lazy torch import) ----

def module_to_layers(model, input_dim: int) -> list[dict]:
    """Walk a flat feed-forward `nn.Module` into ncnn layer dicts (input blob `in0`).

    Handles Linear/ReLU/Tanh/Sigmoid/Flatten/Identity over `model.children()` in order
    (a bare module with no children is treated as the single layer). Raises ValueError on
    any unmapped layer type -- fail loud, route it through the ONNX/pnnx path instead.
    The final layer's output blob is renamed `out0` (matches the NcnnRunner convention).
    """
    import torch.nn as nn

    layers = [input_layer("in0", input_dim)]
    prev = "in0"
    idx = 0
    children = list(model.children()) or [model]
    for child in children:
        if isinstance(child, nn.Linear):
            name = f"fc{idx}"
            weight = child.weight.detach().cpu().contiguous().view(-1).tolist()
            bias = child.bias.detach().cpu().view(-1).tolist() if child.bias is not None else None
            layers.append(linear_layer(name, prev, name, weight, bias,
                                       child.in_features, child.out_features))
            prev = name
            idx += 1
        elif isinstance(child, nn.ReLU):
            name = f"act{idx}"
            layers.append(activation_layer("ReLU", name, prev, name, {0: 0.0}))
            prev = name
            idx += 1
        elif isinstance(child, nn.Tanh):
            name = f"act{idx}"
            layers.append(activation_layer("TanH", name, prev, name))
            prev = name
            idx += 1
        elif isinstance(child, nn.Sigmoid):
            name = f"act{idx}"
            layers.append(activation_layer("Sigmoid", name, prev, name))
            prev = name
            idx += 1
        elif isinstance(child, nn.Flatten):
            name = f"flat{idx}"
            layers.append(activation_layer("Flatten", name, prev, name))
            prev = name
            idx += 1
        elif isinstance(child, nn.Identity):
            continue
        else:
            raise ValueError(
                f"no ncnn mapping for layer {type(child).__name__!r}; route this model "
                "through the ONNX/pnnx path (scripts/export_to_ncnn.py)"
            )
    layers[-1]["tops"] = ["out0"]
    return layers


def export_module_to_ncnn(model, input_dim: int, out_stem: str) -> tuple[Path, Path]:
    """Convert a feed-forward module to `<out_stem>.ncnn.{param,bin}`. Returns the paths."""
    layers = module_to_layers(model, input_dim)
    param_path = Path(f"{out_stem}.ncnn.param")
    bin_path = Path(f"{out_stem}.ncnn.bin")
    param_path.parent.mkdir(parents=True, exist_ok=True)
    param_path.write_text(ncnn_param_text(layers))
    bin_path.write_bytes(ncnn_bin_bytes(layers))
    return param_path, bin_path


def main() -> None:
    import torch.nn as nn
    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SB3 PPO checkpoint .zip")
    parser.add_argument("--out_stem", default="models/policy",
                        help="output stem -> <stem>.ncnn.{param,bin}")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint)
    policy = model.policy.to("cpu")
    policy.eval()

    # Compose the default-MLP actor stack into one flat Sequential. NOTE: this reaches
    # into SB3 internals (mlp_extractor.policy_net + action_net) and assumes an identity
    # feature extractor (true for a single Box "obs"); validate parity for your model.
    obs_space = model.observation_space
    spaces = getattr(obs_space, "spaces", None)
    box = spaces["obs"] if isinstance(spaces, dict) and "obs" in spaces else obs_space
    input_dim = int(box.shape[0])
    stack = nn.Sequential(*list(policy.mlp_extractor.policy_net.children()), policy.action_net)

    param_path, bin_path = export_module_to_ncnn(stack, input_dim, args.out_stem)
    print("Wrote:", param_path)
    print("Wrote:", bin_path)
    print("VALIDATE: run scripts/verify_ncnn_parity.py against the source model before deploy.")


if __name__ == "__main__":
    main()

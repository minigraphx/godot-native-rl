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


def linear_layer(name: str, bottom: str, top: str, weight: list[float], bias: list[float] | None,
                 in_features: int, out_features: int) -> dict:
    """Build an InnerProduct layer. `weight` is the flat [out, in] row-major list."""
    weights = [(weight, True)]
    if bias is not None:
        weights.append((bias, False))
    return {
        "type": "InnerProduct", "name": name, "bottoms": [bottom], "tops": [top],
        "params": {0: int(out_features), 1: 1 if bias is not None else 0, 2: in_features * out_features},
        "weights": weights,
    }


def activation_layer(ncnn_type: str, name: str, bottom: str, top: str, params: dict | None = None) -> dict:
    return {"type": ncnn_type, "name": name, "bottoms": [bottom], "tops": [top], "params": params or {}}


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

#!/usr/bin/env python3
"""Verify ncnn argmax parity with a TorchScript policy over random observations.

The TorchScript counterpart of `verify_ncnn_parity.py`: instead of running the
source model through onnxruntime, it loads the `.pt` with `torch.jit.load` and
diffs against ncnn. Same argmax/logit/degenerate decision rules and `atol=1e-2`
tolerance (reused via `parity_summary` / `VerifyResult` from `verify_ncnn_parity`).

A `.pt` carries no readable input-shape metadata, so the observation dim is parsed
from the pnnx `inputshape` string (the same one used to convert the model).

Usage:
    verify_torchscript_parity.py <pt> <ncnn.param> <ncnn.bin> <in_blob> <out_blob> <inputshape>
Exits 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import re
import sys
from typing import NoReturn

from verify_ncnn_parity import VerifyResult, parity_summary

_FIRST_GROUP = re.compile(r"\[([^\[\]]*)\]")


def obs_dim_from_inputshape(inputshape: str) -> int:
    """Return the observation dim N from a pnnx `inputshape` string.

    Reads the last integer of the FIRST `[...]` group, e.g. `[1,5]` or `[1,5],[1]`
    -> 5, ` [1, 8] ` -> 8. Raises ValueError when no leading `[...]` group with a
    positive trailing integer can be parsed.
    """
    m = _FIRST_GROUP.search(inputshape)
    if m is None:
        raise ValueError(f"no '[...]' group in inputshape {inputshape!r}")
    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"empty input-shape group in {inputshape!r}")
    try:
        n = int(parts[-1])
    except ValueError:
        raise ValueError(
            f"could not parse obs dim from inputshape {inputshape!r} "
            f"(last dim {parts[-1]!r} is not an int)"
        ) from None
    if n <= 0:
        raise ValueError(f"obs dim must be positive, got {n} from {inputshape!r}")
    return n


def verify_torchscript_parity(
    pt_path: str,
    param_path: str,
    bin_path: str,
    in_blob: str,
    out_blob: str,
    inputshape: str,
    *,
    n_samples: int = 50,
    seed: int = 0,
) -> VerifyResult:
    import numpy as np
    import torch
    import ncnn

    obs_dim = obs_dim_from_inputshape(inputshape)
    rng = np.random.default_rng(seed)

    model = torch.jit.load(pt_path)
    model.eval()

    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)

    argmax_mismatches = 0
    value_mismatches = 0
    seen_actions: set[int] = set()

    with torch.no_grad():
        for _ in range(n_samples):
            obs = rng.uniform(-1.0, 1.0, size=(1, obs_dim)).astype(np.float32)
            torch_out = model(torch.from_numpy(obs))
            if isinstance(torch_out, (tuple, list)):
                torch_out = torch_out[0]
            torch_logits = np.ravel(torch_out.detach().numpy())
            torch_arg = int(np.argmax(torch_logits))

            ex = net.create_extractor()
            ex.input(in_blob, ncnn.Mat(obs.reshape(obs_dim)))
            _, out = ex.extract(out_blob)
            ncnn_logits = np.array(out, dtype=np.float32)
            ncnn_arg = int(np.argmax(ncnn_logits))

            if torch_arg != ncnn_arg:
                argmax_mismatches += 1
            if not np.allclose(torch_logits, ncnn_logits, atol=1e-2):
                value_mismatches += 1
            seen_actions.add(ncnn_arg)

    distinct = len(seen_actions)
    ok, summary = parity_summary(argmax_mismatches, value_mismatches, distinct, n_samples)
    return VerifyResult(ok, argmax_mismatches, value_mismatches, distinct, n_samples, summary)


def fail(msg: str) -> NoReturn:
    print(f"PARITY FAILED: {msg}")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 7:
        print(
            "Usage: verify_torchscript_parity.py "
            "<pt> <ncnn.param> <ncnn.bin> <in_blob> <out_blob> <inputshape>"
        )
        sys.exit(2)

    pt_path, param_path, bin_path, in_blob, out_blob, inputshape = sys.argv[1:7]
    result = verify_torchscript_parity(
        pt_path, param_path, bin_path, in_blob, out_blob, inputshape
    )
    if not result.ok:
        fail(result.summary)
    print(f"PARITY OK: {result.summary}")


if __name__ == "__main__":
    main()

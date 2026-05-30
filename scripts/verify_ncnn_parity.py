#!/usr/bin/env python3
"""Verify ncnn argmax parity with the ONNX policy over random observations.

Usage: verify_ncnn_parity.py <onnx> <ncnn.param> <ncnn.bin> <in_blob> <out_blob>
Exits 0 if all checks pass, 1 otherwise.

Checks:
  - Argmax matches on all samples (required: conversion preserved the policy).
  - Logit values are numerically close (atol 1e-2) — catches drift that argmax masks.
  - At least 2 distinct actions appear — catches a degenerate all-zeros model.
"""
import sys
from dataclasses import dataclass
from typing import NoReturn


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    argmax_mismatches: int
    value_mismatches: int
    distinct_actions: int
    n_samples: int
    summary: str


def parity_summary(
    argmax_mismatches: int,
    value_mismatches: int,
    distinct_actions: int,
    n_samples: int,
) -> tuple[bool, str]:
    """Pure decision: given the counts, return (ok, human-readable summary)."""
    if argmax_mismatches:
        return False, f"{argmax_mismatches}/{n_samples} argmax mismatches"
    if value_mismatches:
        return False, f"{value_mismatches}/{n_samples} samples exceed atol=1e-2 logit tolerance"
    if distinct_actions < 2:
        return False, f"only {distinct_actions} distinct action(s) seen — model may be degenerate"
    return True, (
        f"{n_samples}/{n_samples} argmax match, logits within atol=1e-2, "
        f"{distinct_actions} distinct actions seen"
    )


def verify_parity(
    onnx_path: str,
    param_path: str,
    bin_path: str,
    in_blob: str,
    out_blob: str,
    *,
    n_samples: int = 50,
    seed: int = 0,
) -> VerifyResult:
    import numpy as np
    import onnxruntime as ort
    import ncnn

    rng = np.random.default_rng(seed)
    sess = ort.InferenceSession(onnx_path)
    onnx_input_names = {i.name for i in sess.get_inputs()}
    obs_dim = sess.get_inputs()[0].shape[-1]

    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)

    argmax_mismatches = 0
    value_mismatches = 0
    seen_actions: set[int] = set()

    for _ in range(n_samples):
        obs = rng.uniform(-1.0, 1.0, size=(1, obs_dim)).astype(np.float32)
        feeds: dict[str, np.ndarray] = {"obs": obs}
        if "state_ins" in onnx_input_names:
            feeds["state_ins"] = np.zeros((1,), dtype=np.float32)
        onnx_logits = np.ravel(sess.run(None, feeds)[0])
        onnx_arg = int(np.argmax(onnx_logits))

        ex = net.create_extractor()
        ex.input(in_blob, ncnn.Mat(obs.reshape(obs_dim)))
        _, out = ex.extract(out_blob)
        ncnn_logits = np.array(out, dtype=np.float32)
        ncnn_arg = int(np.argmax(ncnn_logits))

        if onnx_arg != ncnn_arg:
            argmax_mismatches += 1
        if not np.allclose(onnx_logits, ncnn_logits, atol=1e-2):
            value_mismatches += 1
        seen_actions.add(ncnn_arg)

    distinct = len(seen_actions)
    ok, summary = parity_summary(argmax_mismatches, value_mismatches, distinct, n_samples)
    return VerifyResult(ok, argmax_mismatches, value_mismatches, distinct, n_samples, summary)


def fail(msg: str) -> NoReturn:
    print(f"PARITY FAILED: {msg}")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 6:
        print("Usage: verify_ncnn_parity.py <onnx> <ncnn.param> <ncnn.bin> <in_blob> <out_blob>")
        sys.exit(2)

    onnx_path, param_path, bin_path, in_blob, out_blob = sys.argv[1:6]
    result = verify_parity(onnx_path, param_path, bin_path, in_blob, out_blob)
    if not result.ok:
        fail(result.summary)
    print(f"PARITY OK: {result.summary}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify ncnn argmax parity with the ONNX policy over random observations.

Usage: verify_ncnn_parity.py <onnx> <ncnn.param> <ncnn.bin> <in_blob> <out_blob>
Exits 0 if all checks pass, 1 otherwise.

Checks:
  - Argmax matches on all 50 samples (required: conversion preserved the policy).
  - Logit values are numerically close (atol 1e-2) — catches drift that argmax masks.
    (torch dynamo exporter and ncnn's InnerProduct differ by ~1e-3 to 5e-3 in float32;
    1e-2 is tight enough to catch real conversion failures while tolerating expected noise.)
  - At least 2 distinct actions appear — catches a degenerate all-zeros model that
    trivially matches argmax while being completely wrong.
"""
import sys
from typing import NoReturn

import numpy as np
import onnxruntime as ort
import ncnn


def fail(msg: str) -> NoReturn:
    print(f"PARITY FAILED: {msg}")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 6:
        print("Usage: verify_ncnn_parity.py <onnx> <ncnn.param> <ncnn.bin> <in_blob> <out_blob>")
        sys.exit(2)

    onnx_path, param_path, bin_path, in_blob, out_blob = sys.argv[1:6]
    rng = np.random.default_rng(0)

    sess = ort.InferenceSession(onnx_path)
    onnx_input_names = {i.name for i in sess.get_inputs()}
    obs_dim = sess.get_inputs()[0].shape[-1]

    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)

    argmax_mismatches = 0
    value_mismatches = 0
    seen_actions: set[int] = set()

    n_samples = 50
    for _ in range(n_samples):
        obs = rng.uniform(-1.0, 1.0, size=(1, obs_dim)).astype(np.float32)

        # ONNX inference.
        feeds: dict[str, np.ndarray] = {"obs": obs}
        if "state_ins" in onnx_input_names:
            feeds["state_ins"] = np.zeros((1,), dtype=np.float32)
        onnx_logits = np.ravel(sess.run(None, feeds)[0])
        onnx_arg = int(np.argmax(onnx_logits))

        # ncnn inference.
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

    if argmax_mismatches:
        fail(f"{argmax_mismatches}/{n_samples} argmax mismatches")
    if value_mismatches:
        fail(f"{value_mismatches}/{n_samples} samples exceed atol=1e-3 logit tolerance")
    if len(seen_actions) < 2:
        fail(f"only {len(seen_actions)} distinct action(s) seen — model may be degenerate")

    print(
        f"PARITY OK: {n_samples}/{n_samples} argmax match, "
        f"logits within atol=1e-2, "
        f"{len(seen_actions)} distinct actions seen"
    )


if __name__ == "__main__":
    main()

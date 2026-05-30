#!/usr/bin/env python3
"""Verify ncnn argmax parity with the ONNX policy over random observations.

Usage: verify_ncnn_parity.py <onnx> <ncnn.param> <ncnn.bin> <in_blob> <out_blob>
Exits 0 if argmax matches on all samples, 1 otherwise.
"""
import sys

import numpy as np
import onnxruntime as ort
import ncnn


def main() -> None:
    onnx_path, param_path, bin_path, in_blob, out_blob = sys.argv[1:6]
    rng = np.random.default_rng(0)

    sess = ort.InferenceSession(onnx_path)
    onnx_inputs = {i.name: i for i in sess.get_inputs()}

    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)

    mismatches = 0
    for _ in range(50):
        obs = rng.uniform(-1.0, 1.0, size=(1, 5)).astype(np.float32)

        feeds = {"obs": obs}
        if "state_ins" in onnx_inputs:
            feeds["state_ins"] = np.zeros((1,), dtype=np.float32)
        onnx_out = sess.run(None, feeds)
        onnx_logits = np.ravel(onnx_out[0])
        onnx_arg = int(np.argmax(onnx_logits))

        ex = net.create_extractor()
        ex.input(in_blob, ncnn.Mat(obs.reshape(5)))
        _, out = ex.extract(out_blob)
        ncnn_logits = np.array(out)
        ncnn_arg = int(np.argmax(ncnn_logits))

        if onnx_arg != ncnn_arg:
            mismatches += 1

    if mismatches:
        print(f"PARITY FAILED: {mismatches}/50 argmax mismatches")
        sys.exit(1)
    print("PARITY OK: 50/50 argmax match between ONNX and ncnn")


if __name__ == "__main__":
    main()

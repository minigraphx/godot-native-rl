#!/usr/bin/env python3
"""Verify INT8 ncnn argmax agreement against the fp32 ncnn baseline.

INT8 trades precision for size/speed, so logit closeness fails by design; we judge argmax
agreement rate over seeded sampled inputs instead. Comparing INT8 vs the fp32 *ncnn* model
(not ONNX) isolates quantization error from any conversion error.

Usage:
    verify_int8_parity.py <fp32.param> <fp32.bin> <int8.param> <int8.bin> \
        <in_blob> <out_blob> <W> <H> <C>
Exits 0 if parity passes, 1 otherwise.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

DEFAULT_THRESHOLD = 0.9
DEFAULT_N_SAMPLES = 50


@dataclass(frozen=True)
class Int8VerifyResult:
    ok: bool
    agreement_rate: float
    distinct_actions: int
    n_samples: int
    summary: str


def int8_parity_summary(
    agreement_rate: float, distinct_actions: int, n_samples: int, threshold: float
) -> tuple[bool, str]:
    """Pure decision: pass iff agreement >= threshold AND at least 2 distinct actions."""
    if agreement_rate < threshold:
        return False, (
            f"argmax agreement {agreement_rate:.3f} < threshold {threshold:.3f} "
            f"over {n_samples} samples"
        )
    if distinct_actions < 2:
        return False, f"only {distinct_actions} distinct action(s) — model may be degenerate"
    return True, (
        f"argmax agreement {agreement_rate:.3f} >= {threshold:.3f}, "
        f"{distinct_actions} distinct actions over {n_samples} samples"
    )


def _argmax_image(net, img_hwc_uint8, width: int, height: int, in_blob: str, out_blob: str) -> int:
    """Run one image through an ncnn net the same way NcnnRunner.run_inference_image does."""
    import numpy as np
    import ncnn

    flat = np.ascontiguousarray(img_hwc_uint8).reshape(-1)
    mat = ncnn.Mat.from_pixels(flat, ncnn.Mat.PixelType.PIXEL_RGB, width, height)
    mat.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
    ex = net.create_extractor()
    ex.input(in_blob, mat)
    _, out = ex.extract(out_blob)
    return int(np.argmax(np.array(out).reshape(-1)))


def verify_int8_parity(
    fp32_param: str,
    fp32_bin: str,
    int8_param: str,
    int8_bin: str,
    in_blob: str,
    out_blob: str,
    width: int,
    height: int,
    channels: int,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 0,
) -> Int8VerifyResult:
    import numpy as np
    import ncnn

    rng = np.random.default_rng(seed)
    fp32 = ncnn.Net()
    fp32.load_param(fp32_param)
    fp32.load_model(fp32_bin)
    int8 = ncnn.Net()
    int8.load_param(int8_param)
    int8.load_model(int8_bin)

    agree = 0
    seen: set[int] = set()
    for _ in range(n_samples):
        img = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
        a32 = _argmax_image(fp32, img, width, height, in_blob, out_blob)
        a8 = _argmax_image(int8, img, width, height, in_blob, out_blob)
        if a32 == a8:
            agree += 1
        seen.add(a8)

    rate = agree / n_samples if n_samples else 0.0
    ok, summary = int8_parity_summary(rate, len(seen), n_samples, threshold)
    return Int8VerifyResult(ok, rate, len(seen), n_samples, summary)


def main() -> None:
    if len(sys.argv) < 10:
        print(
            "Usage: verify_int8_parity.py <fp32.param> <fp32.bin> <int8.param> <int8.bin> "
            "<in_blob> <out_blob> <W> <H> <C>"
        )
        sys.exit(2)
    fp32_param, fp32_bin, int8_param, int8_bin, in_blob, out_blob = sys.argv[1:7]
    width, height, channels = (int(x) for x in sys.argv[7:10])
    result = verify_int8_parity(
        fp32_param, fp32_bin, int8_param, int8_bin, in_blob, out_blob, width, height, channels
    )
    if not result.ok:
        print(f"INT8 PARITY FAILED: {result.summary}")
        sys.exit(1)
    print(f"INT8 PARITY OK: {result.summary}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate INT8 calibration data (.npy tensors + list file) for ncnn2table.

ncnn2table with `type=1` reads float32 .npy tensors. For an image policy the tensor must
be CHW and normalized exactly as the deploy path normalizes (NcnnRunner.run_inference_image
with normalize_to_zero_one=true divides by 255), so calibration activations match what the
game feeds at inference.

The .npy on disk is stored CHW (shape [C,H,W]); ncnn2table is told `shape=[W,H,C]` and
reverses the dims internally (see read_npy in tools/quantize/ncnn2table.cpp).

NOTE: this samples a synthetic distribution for the regression fixture. Real policies should
calibrate on captured game frames representative of deployment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_N_SAMPLES = 256


def table_shape_arg(width: int, height: int, channels: int) -> str:
    """ncnn2table `shape=` arg, given in W,H,C order."""
    return f"[{width},{height},{channels}]"


def sample_images(n: int, width: int, height: int, channels: int, seed: int):
    """Return n seeded uniform-random uint8 images, shape (n, H, W, C)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, height, width, channels), dtype=np.uint8)


def image_to_chw_float(img_hwc):
    """HWC uint8 -> contiguous CHW float32 normalized to [0,1] (matches run_inference_image)."""
    import numpy as np

    chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw)


def write_calibration_set(images, outdir) -> Path:
    """Write each image as a CHW float32 .npy and a newline list file. Returns the list path."""
    import numpy as np

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, img in enumerate(images):
        npy_path = out / f"calib_{i:04d}.npy"
        np.save(npy_path, image_to_chw_float(img))
        lines.append(str(npy_path))
    list_path = out / "calib_list.txt"
    list_path.write_text("\n".join(lines) + "\n")
    return list_path


def generate(outdir, *, n_samples=DEFAULT_N_SAMPLES, width=8, height=8, channels=3, seed=0) -> Path:
    """Sample + write a calibration set in one call. Returns the list-file path."""
    imgs = sample_images(n_samples, width, height, channels, seed)
    return write_calibration_set(imgs, outdir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate INT8 calibration .npy set for ncnn2table.")
    p.add_argument("outdir")
    p.add_argument("--samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--height", type=int, default=8)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    list_path = generate(
        a.outdir, n_samples=a.samples, width=a.width, height=a.height,
        channels=a.channels, seed=a.seed,
    )
    print(f"OK: {a.samples} calibration tensors, list at {list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

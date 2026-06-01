"""Generate a tiny seeded CNN and an ncnn golden fixture for image-inference tests.

Run under .venv-train (torch + onnxruntime + ncnn; shells out to .venv pnnx via
scripts/export_to_ncnn.py). Writes models/synthetic_cnn.ncnn.{param,bin} and
models/synthetic_cnn_golden.json (a fixed 8x8x3 image + golden logits/argmax) used by
test/unit/test_image_inference_golden.gd.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_cnn.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
WIDTH = HEIGHT = 8
CHANNELS = 3
N_ACTIONS = 4
SEED = 42


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(CHANNELS, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * HEIGHT * WIDTH, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)


def fixed_image_bytes() -> bytes:
    # Deterministic ramp; 8*8*3 = 192 distinct values, all < 256.
    return bytes(range(WIDTH * HEIGHT * CHANNELS))


def to_chw_normalized(img_bytes: bytes) -> np.ndarray:
    hwc = np.frombuffer(img_bytes, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)
    chw = hwc.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw[None, :, :, :]  # [1, 3, 8, 8]


def main() -> int:
    torch.manual_seed(SEED)
    model = TinyCNN().eval()
    MODELS.mkdir(exist_ok=True)

    img_bytes = fixed_image_bytes()
    chw = to_chw_normalized(img_bytes)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "synthetic_cnn.onnx"
        dummy = torch.zeros(1, CHANNELS, HEIGHT, WIDTH)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"], opset_version=13,
            dynamo=False,
        )
        sess = ort.InferenceSession(str(onnx_path))
        in_name = sess.get_inputs()[0].name
        logits_onnx = np.array(sess.run(None, {in_name: chw})[0]).reshape(-1)

        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_to_ncnn.py"),
             str(onnx_path), "--outdir", str(MODELS), "--skip-verify",
             "--inputshape", "[1,3,8,8]"],
            check=False,
        ).returncode
        if rc != 0:
            print("export_to_ncnn failed", file=sys.stderr)
            return 1

    param = MODELS / "synthetic_cnn.ncnn.param"
    bin_ = MODELS / "synthetic_cnn.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    # Best-effort early cross-check of the C++ deploy path's preprocessing via the ncnn
    # python package. The authoritative parity gate is the GDScript golden test (Task 5),
    # which runs the real C++ run_inference_image; this block only gives an early signal,
    # so API quirks here must not block generation — hence the try/except.
    try:
        import ncnn
        net = ncnn.Net()
        net.load_param(str(param))
        net.load_model(str(bin_))
        ex = net.create_extractor()
        mat = ncnn.Mat.from_pixels(
            np.frombuffer(img_bytes, dtype=np.uint8),
            ncnn.Mat.PixelType.PIXEL_RGB, WIDTH, HEIGHT,
        )
        mat.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
        ex.input("in0", mat)
        _, out = ex.extract("out0")
        logits_ncnn = np.array(out).reshape(-1)
        max_diff = float(np.max(np.abs(logits_onnx - logits_ncnn)))
        print(f"onnx vs ncnn(python) max abs diff: {max_diff:.5f}")
    except Exception as exc:  # noqa: BLE001 — early signal only; GDScript golden is the gate
        print(f"ncnn(python) cross-check skipped ({exc}); GDScript golden remains the gate")

    golden = {
        "width": WIDTH,
        "height": HEIGHT,
        "channels": CHANNELS,
        "image_bytes": list(img_bytes),
        "logits": [float(x) for x in logits_onnx],
        "argmax": int(np.argmax(logits_onnx)),
    }
    (MODELS / "synthetic_cnn_golden.json").write_text(json.dumps(golden, indent=2))
    print("golden logits:", golden["logits"], "argmax:", golden["argmax"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

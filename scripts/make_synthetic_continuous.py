"""Generate a tiny seeded MLP and an ncnn golden fixture for continuous-action decode tests.

Run under .venv-train (torch + onnxruntime + ncnn; shells out to .venv pnnx via
scripts/export_to_ncnn.py). Writes models/synthetic_continuous.ncnn.{param,bin} and
models/synthetic_continuous_golden.json (a fixed obs vector + golden raw output) used by
test/unit/test_action_decode_golden.gd to verify run_inference numerical closeness (atol=1e-2)
and the continuous decode path.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_continuous.py
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
OBS_DIM = 5
OUT_DIM = 3  # a single continuous action key of size 3 (mean vector)
SEED = 7


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def fixed_obs() -> np.ndarray:
    # Deterministic, non-trivial obs vector.
    return np.array([[0.5, -0.25, 0.1, 0.75, -0.6]], dtype=np.float32)


def main() -> int:
    torch.manual_seed(SEED)
    model = TinyMLP().eval()
    MODELS.mkdir(exist_ok=True)

    obs = fixed_obs()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "synthetic_continuous.onnx"
        dummy = torch.zeros(1, OBS_DIM)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"], opset_version=13,
            dynamo=False,
        )
        sess = ort.InferenceSession(str(onnx_path))
        in_name = sess.get_inputs()[0].name
        out_onnx = np.array(sess.run(None, {in_name: obs})[0]).reshape(-1)

        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_to_ncnn.py"),
             str(onnx_path), "--outdir", str(MODELS), "--skip-verify",
             "--inputshape", "[1,5]"],
            check=False,
        ).returncode
        if rc != 0:
            print("export_to_ncnn failed", file=sys.stderr)
            return 1

    param = MODELS / "synthetic_continuous.ncnn.param"
    bin_ = MODELS / "synthetic_continuous.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    golden = {
        "obs": [float(x) for x in obs.reshape(-1)],
        "output": [float(x) for x in out_onnx],
        "squashed": [float(np.tanh(x)) for x in out_onnx],
    }
    (MODELS / "synthetic_continuous_golden.json").write_text(json.dumps(golden, indent=2))
    print(f"wrote {param.name}, {bin_.name}, synthetic_continuous_golden.json")
    print(f"golden output: {golden['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

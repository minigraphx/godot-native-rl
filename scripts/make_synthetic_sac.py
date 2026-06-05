"""Generate a tiny seeded MLP continuous actor + ncnn golden fixture for algorithm-agnostic tests (#45).

A SAC-style continuous actor: obs -> hidden -> ACT_DIM raw means (pre-tanh). SAC squashes the mean
with tanh at deploy (not in the network), so the deterministic deploy action is tanh(mean) — applied
game-side by ActionDecode via the per-key "squash" flag. This is a separate, SAC-named, self-contained
guard (the generic synthetic_continuous test already covers squash; this asserts the contract for SAC
explicitly by name — decision 2026-06-05, see the design spec).

Run under .venv-train (torch + onnxruntime; shells out to .venv pnnx via scripts/export_to_ncnn.py).
Writes models/synthetic_sac.ncnn.{param,bin} and models/synthetic_sac_golden.json (fixed obs, golden
raw means, expected tanh(mean)) used by test/unit/test_algorithm_agnostic_golden_inference.gd.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_sac.py
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
ACT_DIM = 2  # a single continuous action key of size 2 (mean vector, pre-tanh)
SEED = 23


class TinyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, ACT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def fixed_obs() -> np.ndarray:
    # Deterministic, non-trivial obs vector.
    return np.array([[0.5, -0.25, 0.1, 0.75, -0.6]], dtype=np.float32)


def main() -> int:
    torch.manual_seed(SEED)
    model = TinyActor().eval()
    MODELS.mkdir(exist_ok=True)

    obs = fixed_obs()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "synthetic_sac.onnx"
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

    param = MODELS / "synthetic_sac.ncnn.param"
    bin_ = MODELS / "synthetic_sac.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    golden = {
        "obs": [float(x) for x in obs.reshape(-1)],
        "output": [float(x) for x in out_onnx],
        "squashed": [float(np.tanh(x)) for x in out_onnx],
    }
    (MODELS / "synthetic_sac_golden.json").write_text(json.dumps(golden, indent=2))
    print(f"wrote {param.name}, {bin_.name}, synthetic_sac_golden.json")
    print(f"golden means: {golden['output']}  squashed: {golden['squashed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

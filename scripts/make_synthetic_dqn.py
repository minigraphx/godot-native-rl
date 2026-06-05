"""Generate a tiny seeded MLP Q-net + ncnn golden fixture for algorithm-agnostic decode tests (#45).

A DQN-style discrete Q-network: obs -> hidden -> N action-value estimates. Weights/biases are
scaled so the outputs are clearly UNBOUNDED Q-values (magnitude ~tens), distinct from small PPO
logits, to prove argmax survives the real fp32 ncnn pipeline end-to-end.

Run under .venv-train (torch + onnxruntime; shells out to .venv pnnx via scripts/export_to_ncnn.py).
Writes models/synthetic_dqn.ncnn.{param,bin} and models/synthetic_dqn_golden.json (fixed obs,
golden Q-values, expected argmax) used by test/unit/test_algorithm_agnostic_golden_inference.gd.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_dqn.py
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
N_ACTIONS = 4  # a single discrete action key of size 4 (Q-value per action)
SEED = 11


class TinyQNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def fixed_obs() -> np.ndarray:
    # Deterministic, non-trivial obs vector.
    return np.array([[0.5, -0.25, 0.1, 0.75, -0.6]], dtype=np.float32)


def main() -> int:
    torch.manual_seed(SEED)
    model = TinyQNet().eval()
    # Scale the output head so Q-values are clearly unbounded (~tens), with distinct per-action
    # biases guaranteeing a stable, unique argmax independent of fp32 drift.
    with torch.no_grad():
        model.fc2.weight *= 8.0
        model.fc2.bias.copy_(torch.tensor([2.0, 25.0, 9.0, -3.0]))
    MODELS.mkdir(exist_ok=True)

    obs = fixed_obs()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "synthetic_dqn.onnx"
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

    param = MODELS / "synthetic_dqn.ncnn.param"
    bin_ = MODELS / "synthetic_dqn.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    golden = {
        "obs": [float(x) for x in obs.reshape(-1)],
        "output": [float(x) for x in out_onnx],
        "argmax": int(np.argmax(out_onnx)),
    }
    (MODELS / "synthetic_dqn_golden.json").write_text(json.dumps(golden, indent=2))
    print(f"wrote {param.name}, {bin_.name}, synthetic_dqn_golden.json")
    print(f"golden Q-values: {golden['output']}  argmax: {golden['argmax']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

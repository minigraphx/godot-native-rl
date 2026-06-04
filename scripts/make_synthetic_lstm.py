"""Generate a tiny seeded LSTM + an ncnn golden fixture for recurrent-deploy tests.

Run under .venv-train (torch + ncnn; shells to .venv pnnx via scripts/export_to_ncnn.py).
Writes models/synthetic_lstm.ncnn.{param,bin}, models/synthetic_lstm.recurrent.json, and
models/synthetic_lstm_golden.json (a fixed obs SEQUENCE + torch-reference actions/state per
step, zero-init start) used by test/unit/test_recurrent_golden_inference.gd.

Resolves the spec's feasibility unknown: does pnnx preserve the 3-in/3-out LSTM state blobs?
If the ncnn(python) cross-check below fails to bind in1/in2 or out1/out2, the fallback is to
hand-author the .param LSTM wiring (see DEVELOPMENT.md). The script writes the EXACT blob names
and Mat shapes it verified into the sidecar, so the GDScript side stays shape-agnostic.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_lstm.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OBS_SIZE = 5
HIDDEN = 8
N_ACTIONS = 4
SEQ_STEPS = 4
# Seed chosen so every step's top-2 logit margin is ~0.5 (>>the ~0.01 float32 drift the
# LSTM accumulates across the sequence in ONNX->ncnn conversion), keeping the golden
# argmax decision stable end-to-end. (Seed 7 landed on a near-tie at step 2 that the
# conversion drift flipped.) See the parity print below.
SEED = 45


class TinyLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # batch_first=False: input (seq=1, batch=1, OBS_SIZE); state (num_layers=1, batch=1, HIDDEN)
        self.lstm = nn.LSTM(OBS_SIZE, HIDDEN, num_layers=1)
        self.fc = nn.Linear(HIDDEN, N_ACTIONS)

    def forward(self, obs, h_in, c_in):
        out, (h_out, c_out) = self.lstm(obs, (h_in, c_in))
        action = self.fc(out)
        return action, h_out, c_out


def main() -> int:
    torch.manual_seed(SEED)
    model = TinyLSTM().eval()
    MODELS.mkdir(exist_ok=True)

    # Fixed obs sequence (deterministic seeded Gaussian samples), zero initial state.
    rng = np.random.default_rng(SEED)
    obs_seq = [rng.standard_normal(OBS_SIZE).astype(np.float32) for _ in range(SEQ_STEPS)]

    # Torch reference: carry (h, c) across the sequence from zeros.
    h = torch.zeros(1, 1, HIDDEN)
    c = torch.zeros(1, 1, HIDDEN)
    ref_steps = []
    with torch.no_grad():
        for obs in obs_seq:
            obs_t = torch.from_numpy(obs).reshape(1, 1, OBS_SIZE)
            action, h, c = model(obs_t, h, c)
            logits = action.reshape(-1).numpy()
            ref_steps.append({
                "obs": [float(x) for x in obs],
                "logits": [float(x) for x in logits],
                "argmax": int(np.argmax(logits)),
            })

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "synthetic_lstm.onnx"
        dummy_obs = torch.zeros(1, 1, OBS_SIZE)
        dummy_h = torch.zeros(1, 1, HIDDEN)
        dummy_c = torch.zeros(1, 1, HIDDEN)
        torch.onnx.export(
            model, (dummy_obs, dummy_h, dummy_c), str(onnx_path),
            input_names=["obs", "h_in", "c_in"],
            output_names=["action", "h_out", "c_out"],
            opset_version=13, dynamo=False,
        )
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_to_ncnn.py"),
             str(onnx_path), "--outdir", str(MODELS), "--skip-verify",
             "--inputshape", "[1,1,5],[1,1,8],[1,1,8]"],
            check=False,
        ).returncode
        if rc != 0:
            print("export_to_ncnn failed", file=sys.stderr)
            return 1

    param = MODELS / "synthetic_lstm.ncnn.param"
    bin_ = MODELS / "synthetic_lstm.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    # Discover the ncnn Mat shapes ncnn actually expects, and verify parity by replaying the
    # sequence through ncnn(python) with state fed back. The blob names follow pnnx ordering:
    # inputs obs=in0, h_in=in1, c_in=in2; outputs action=out0, h_out=out1, c_out=out2.
    # If these bindings fail, fall back to hand-authoring (see module docstring).
    import ncnn
    OBS_SHAPE = [OBS_SIZE]
    STATE_SHAPE = [HIDDEN]
    net = ncnn.Net()
    net.load_param(str(param))
    net.load_model(str(bin_))
    h_n = np.zeros(HIDDEN, dtype=np.float32)
    c_n = np.zeros(HIDDEN, dtype=np.float32)
    max_diff = 0.0
    for step in ref_steps:
        ex = net.create_extractor()
        ex.input("in0", ncnn.Mat(np.array(step["obs"], dtype=np.float32).reshape(OBS_SHAPE).copy()))
        ex.input("in1", ncnn.Mat(h_n.reshape(STATE_SHAPE).copy()))
        ex.input("in2", ncnn.Mat(c_n.reshape(STATE_SHAPE).copy()))
        _, out0 = ex.extract("out0")
        _, out1 = ex.extract("out1")
        _, out2 = ex.extract("out2")
        logits_ncnn = np.array(out0).reshape(-1)
        h_n = np.array(out1, dtype=np.float32).reshape(-1)
        c_n = np.array(out2, dtype=np.float32).reshape(-1)
        max_diff = max(max_diff, float(np.max(np.abs(logits_ncnn - np.array(step["logits"])))))
    print(f"torch vs ncnn(python) max abs diff over sequence: {max_diff:.5f}")
    if max_diff > 1e-2:
        print("PARITY FAIL — state blobs likely not preserved; hand-author fallback needed", file=sys.stderr)
        return 1

    sidecar = {
        "obs_input": "in0",
        "obs_shape": OBS_SHAPE,
        "action_output": "out0",
        "state_pairs": [
            {"in": "in1", "out": "out1", "shape": STATE_SHAPE},
            {"in": "in2", "out": "out2", "shape": STATE_SHAPE},
        ],
    }
    (MODELS / "synthetic_lstm.recurrent.json").write_text(json.dumps(sidecar, indent=2))
    golden = {"obs_size": OBS_SIZE, "hidden": HIDDEN, "n_actions": N_ACTIONS, "steps": ref_steps}
    (MODELS / "synthetic_lstm_golden.json").write_text(json.dumps(golden, indent=2))
    print("wrote sidecar + golden;", SEQ_STEPS, "steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SampleFactory Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third training backend — SampleFactory async PPO over the godot_rl bridge — that trains the chase example and exports the trained actor unchanged into `export_to_ncnn.py` → ncnn, with a green (auto-skipping) end-to-end smoke in `run_tests.sh`.

**Architecture:** Isolate SampleFactory in a dedicated `.venv-sf` (it pins `gymnasium<1.0`, which would otherwise downgrade the SB3/CleanRL stack). A thin trainer (`scripts/train_sf.py`) drives godot_rl's supported `sample_factory_training` entry point with macOS-safe, parity-safe overrides (serial/sync, `normalize_input=False`). A separate exporter (`scripts/export_sf_to_onnx.py`) loads the SF checkpoint, rebuilds the `ActorCritic`, and wraps its actor path to emit raw action logits with godot_rl's `obs`/`output` ONNX naming. An orchestrator (`scripts/train_sf.sh`) chains train → export → ncnn convert+parity, mirroring `train_cleanrl.sh`.

**Tech Stack:** Python 3.13, `sample_factory==2.1.1`, `godot_rl==0.8.2`, torch 2.12, pnnx/ncnn, Godot 4.5+, stdlib `unittest`, bash.

**Spec:** `docs/superpowers/specs/2026-06-05-sample-factory-backend-design.md`

---

## File Structure

- **Create** `requirements-sf.txt` — SF venv deps (`godot-rl-agents`, `sample_factory`).
- **Modify** `scripts/setup_training.sh` — add the third `.venv-sf` create + `--check` listing.
- **Modify** `test/python/test_setup_training.py` — assert `requirements-sf.txt` is named by `--check`.
- **Create** `scripts/train_sf.py` — pure helpers (`parse_args`, `build_sf_argv`, `client_port`) + `main()` driving `sample_factory_training`.
- **Create** `test/python/test_train_sf.py` — unit tests for the pure helpers (no SF import).
- **Create** `scripts/export_sf_to_onnx.py` — pure helper (`actor_logit_layout`) + `main()` (SF checkpoint → ONNX).
- **Create** `test/python/test_export_sf_to_onnx.py` — unit tests for `actor_logit_layout` (no SF import).
- **Create** `scripts/train_sf.sh` — orchestrator (train → export → ncnn).
- **Modify** `test/run_tests.sh` — add the guarded end-to-end SF smoke step.
- **Modify (docs)** `README.md`, `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`, `docs/BACKLOG.md`.

**Convention reminders (from CLAUDE.md):** Python 4-space indent; heavy imports (`torch`/`sample_factory`/`godot_rl`/`numpy`) **lazy inside `main()`** so pure helpers stay testable; tests are stdlib `unittest` auto-discovered by `run_tests.sh`. Do NOT push to `main`. Branch `feat/sample-factory-backend` already exists with the design commit.

---

## Task 1: `.venv-sf` dependency isolation

**Files:**
- Create: `requirements-sf.txt`
- Modify: `scripts/setup_training.sh`
- Test: `test/python/test_setup_training.py`

- [ ] **Step 1: Write the failing test**

Add this method to `class TestSetupTraining` in `test/python/test_setup_training.py` (after `test_check_mode_runs_and_names_next_step`):

```python
    def test_check_mode_names_sf_requirements(self):
        # The SF backend lives in a third venv (.venv-sf); --check must name its requirements file.
        result = subprocess.run(
            [SCRIPT, "--check"], cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout + result.stderr
        self.assertIn("requirements-sf.txt", out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_setup_training -v`
Expected: FAIL — `requirements-sf.txt` not in output (the file/wiring don't exist yet).

- [ ] **Step 3: Create `requirements-sf.txt`**

```
# SampleFactory training backend — isolated venv (.venv-sf).
# SF pins gymnasium<1.0, which would downgrade the SB3/CleanRL stack in .venv-train,
# so it lives in its own venv. See docs/superpowers/specs/2026-06-05-sample-factory-backend-design.md
godot-rl-agents==0.8.2
sample_factory==2.1.1
```

- [ ] **Step 4: Wire `.venv-sf` into `scripts/setup_training.sh`**

Add the interpreter + requirements vars next to the existing ones (after the `PYTHON_CONVERT=` line):

```bash
PYTHON_SF="${PYTHON_SF:-python3.13}"
REQ_SF="requirements-sf.txt"
```

Add the SF venv to the banner (after the `convert venv:` echo line):

```bash
echo "  sf venv:      .venv-sf     (interpreter: $PYTHON_SF, deps: $REQ_SF)"
```

Add `$REQ_SF` to the requirements-presence loop guard — change:

```bash
for f in "$REQ_TRAIN" "$REQ_CONVERT"; do
```

to:

```bash
for f in "$REQ_TRAIN" "$REQ_CONVERT" "$REQ_SF"; do
```

In the `--check` block, add the SF note (after the `PYTHON_CONVERT` note line) and mention the file (the test only needs the filename, which the banner already prints, but add an explicit line for clarity):

```bash
	command -v "$PYTHON_SF" >/dev/null 2>&1 || echo "NOTE: $PYTHON_SF not on PATH (needed for .venv-sf; override with PYTHON_SF=)."
```

Add the third create call (after the existing `create_venv "$PYTHON_CONVERT" ".venv" "$REQ_CONVERT"` line):

```bash
create_venv "$PYTHON_SF" ".venv-sf" "$REQ_SF"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_setup_training -v`
Expected: PASS (all methods, including the new one).

- [ ] **Step 6: Create the real `.venv-sf` (one-time, heavy)**

Run: `./scripts/setup_training.sh`
Expected: `.venv-train` and `.venv` reported "already exists — reusing"; `.venv-sf` created and `sample_factory==2.1.1` + `gymnasium-0.29.1` installed. Then sanity-check the import:

Run: `.venv-sf/bin/python -c "from godot_rl.wrappers.sample_factory_wrapper import sample_factory_training; import sample_factory; print('SF', sample_factory.__version__)"`
Expected: prints `SF 2.1.1` with no ImportError.

- [ ] **Step 7: Commit**

```bash
git add requirements-sf.txt scripts/setup_training.sh test/python/test_setup_training.py
git commit -m "feat: add .venv-sf isolated venv for the SampleFactory backend (#24)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `train_sf.py` pure helpers

**Files:**
- Create: `scripts/train_sf.py`
- Test: `test/python/test_train_sf.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_train_sf.py`:

```python
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_sf as ts  # noqa: E402


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        cfg = ts.parse_args([])
        self.assertEqual(cfg.timesteps, 1_000_000)
        self.assertEqual(cfg.base_port, 11008)
        self.assertEqual(cfg.env_agents, 1)
        self.assertEqual(cfg.experiment, "chase_sf")
        self.assertEqual(cfg.train_dir, "logs/sf")

    def test_overrides(self):
        cfg = ts.parse_args(["--timesteps", "2000", "--base_port", "12000", "--env_agents", "4"])
        self.assertEqual(cfg.timesteps, 2000)
        self.assertEqual(cfg.base_port, 12000)
        self.assertEqual(cfg.env_agents, 4)


class TestClientPort(unittest.TestCase):
    def test_single_worker_offset(self):
        # godot_rl's make_godot_env_func uses base_port + 1 + env_id; the single serial
        # worker is env_id=0, so the Godot client connects on base_port + 1.
        self.assertEqual(ts.client_port(11008), 11009)


class TestBuildSfArgv(unittest.TestCase):
    def test_contains_macos_and_parity_safe_overrides(self):
        cfg = ts.parse_args(["--timesteps", "2000", "--base_port", "11008", "--env_agents", "1"])
        argv = ts.build_sf_argv(cfg)
        self.assertIn("--serial_mode=True", argv)
        self.assertIn("--async_rl=False", argv)
        self.assertIn("--num_workers=1", argv)
        self.assertIn("--num_envs_per_worker=1", argv)
        self.assertIn("--normalize_input=False", argv)
        self.assertIn("--normalize_returns=False", argv)
        self.assertIn("--use_rnn=False", argv)
        self.assertIn("--device=cpu", argv)
        self.assertIn("--train_for_env_steps=2000", argv)
        self.assertIn("--base_port=11008", argv)
        self.assertIn("--env_agents=1", argv)
        self.assertIn("--experiment=chase_sf", argv)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_train_sf -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_sf'`.

- [ ] **Step 3: Write the pure helpers in `scripts/train_sf.py`**

Create `scripts/train_sf.py` with ONLY the module-top pure helpers for now (no `main()` yet):

```python
#!/usr/bin/env python3
"""Train the Chase The Target agent with SampleFactory (async PPO) over the godot-rl bridge.

Third training backend, alongside scripts/train_chase.py (SB3) and scripts/train_cleanrl.py
(CleanRL). It drives godot_rl's supported SampleFactory entry point
(`godot_rl.wrappers.sample_factory_wrapper.sample_factory_training`) with macOS-safe and
parity-safe overrides, then scripts/export_sf_to_onnx.py turns the SF checkpoint into ONNX that
flows unchanged into scripts/export_to_ncnn.py -> native ncnn deploy.

Runs in the isolated .venv-sf (SF pins gymnasium<1.0). Heavy imports (sample_factory / godot_rl /
torch) are LAZY inside main() so the pure helpers below stay unit-testable without those deps.

Design: docs/superpowers/specs/2026-06-05-sample-factory-backend-design.md
"""
from __future__ import annotations

import argparse
from typing import NamedTuple, Sequence


class SFConfig(NamedTuple):
    """Immutable SampleFactory run config (built from argv by parse_args)."""

    timesteps: int
    base_port: int
    env_agents: int
    speedup: int
    seed: int
    experiment: str
    train_dir: str


def parse_args(argv: Sequence[str] | None = None) -> SFConfig:
    """Parse argv into an immutable SFConfig. Raises SystemExit on unknown args (argparse)."""
    p = argparse.ArgumentParser(allow_abbrev=False, description="SampleFactory PPO for chase.")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--base_port", type=int, default=11008)
    p.add_argument("--env_agents", type=int, default=1)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment", type=str, default="chase_sf")
    p.add_argument("--train_dir", type=str, default="logs/sf")
    a = p.parse_args(argv)
    return SFConfig(
        timesteps=a.timesteps,
        base_port=a.base_port,
        env_agents=a.env_agents,
        speedup=a.speedup,
        seed=a.seed,
        experiment=a.experiment,
        train_dir=a.train_dir,
    )


def client_port(base_port: int) -> int:
    """Port the Godot client must connect on.

    godot_rl's make_godot_env_func computes `port = base_port; if env_config: port += 1 + env_id`.
    Our single serial worker is env_id=0, so the client listens on base_port + 1.
    """
    return base_port + 1


def build_sf_argv(cfg: SFConfig) -> list[str]:
    """Translate an SFConfig into the SampleFactory CLI argv list (the `extras` for parse_gdrl_args).

    Bakes in the overrides that make the run macOS-safe (serial/sync, single worker) and
    ncnn-parity-safe (input/return normalization OFF, so the exported actor is a plain MLP).
    """
    return [
        f"--experiment={cfg.experiment}",
        f"--train_for_env_steps={cfg.timesteps}",
        f"--base_port={cfg.base_port}",
        f"--env_agents={cfg.env_agents}",
        f"--seed={cfg.seed}",
        "--serial_mode=True",
        "--async_rl=False",
        "--num_workers=1",
        "--num_envs_per_worker=1",
        "--worker_num_splits=1",
        "--normalize_input=False",
        "--normalize_returns=False",
        "--use_rnn=False",
        "--device=cpu",
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_train_sf -v`
Expected: PASS (all three test classes).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_sf.py test/python/test_train_sf.py
git commit -m "feat: train_sf.py pure helpers (config, port offset, SF argv) (#24)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `train_sf.py` main() (SampleFactory driver)

**Files:**
- Modify: `scripts/train_sf.py`

This driver needs a live Godot client + SF, so it is not unit-tested here — it is exercised by the end-to-end smoke (Task 7). This task implements `main()` and verifies it parses/imports.

- [ ] **Step 1: Implement `main()` in `scripts/train_sf.py`**

Append to `scripts/train_sf.py`:

```python
def _build_args_namespace(cfg: SFConfig) -> argparse.Namespace:
    """Build the `args` namespace godot_rl's sample_factory_training expects.

    register_gdrl_env reads args.env_path / args.speedup / args.seed / args.viz;
    parse_gdrl_args reads args.experiment_dir / args.experiment_name / args.eval.
    env_path=None => in-editor training: SF opens the server and waits for the Godot client.
    """
    return argparse.Namespace(
        env_path=None,
        experiment_dir=cfg.train_dir,
        experiment_name=cfg.experiment,
        speedup=cfg.speedup,
        seed=cfg.seed,
        viz=False,
        eval=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    # Heavy import is lazy: only when actually training (keeps the pure helpers import-light).
    from godot_rl.wrappers.sample_factory_wrapper import sample_factory_training

    cfg = parse_args(argv)
    args = _build_args_namespace(cfg)
    extras = build_sf_argv(cfg)
    print(f"SampleFactory training: experiment={cfg.experiment} timesteps={cfg.timesteps} "
          f"base_port={cfg.base_port} client_port={client_port(cfg.base_port)} env_agents={cfg.env_agents}")
    status = sample_factory_training(args, extras)
    print(f"SampleFactory finished with status={status}")
    # SF status is an enum; treat the normal-termination value as success (0).
    return int(getattr(status, "value", status) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it imports and parses (no training)**

Run: `.venv-sf/bin/python scripts/train_sf.py --help`
Expected: argparse help text listing `--timesteps`, `--base_port`, `--env_agents`, etc.; exit 0. (This imports `train_sf` and runs argparse without launching SF.)

- [ ] **Step 3: Verify pure-helper tests still pass under the SF venv too**

Run: `.venv-sf/bin/python -m unittest test.python.test_train_sf -v`
Expected: PASS (confirms the lazy import didn't leak into the helpers).

- [ ] **Step 4: Commit**

```bash
git add scripts/train_sf.py
git commit -m "feat: train_sf.py main() driving sample_factory_training (#24)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `export_sf_to_onnx.py` pure helper

**Files:**
- Create: `scripts/export_sf_to_onnx.py`
- Test: `test/python/test_export_sf_to_onnx.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_export_sf_to_onnx.py`:

```python
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_sf_to_onnx as ex  # noqa: E402


class TestActorLogitLayout(unittest.TestCase):
    def test_single_discrete(self):
        total, nvec = ex.actor_logit_layout([5])
        self.assertEqual(total, 5)
        self.assertEqual(nvec, [5])

    def test_multi_discrete(self):
        total, nvec = ex.actor_logit_layout([3, 2, 4])
        self.assertEqual(total, 9)
        self.assertEqual(nvec, [3, 2, 4])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ex.actor_logit_layout([])

    def test_non_positive_raises(self):
        with self.assertRaises(ValueError):
            ex.actor_logit_layout([5, 0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_export_sf_to_onnx -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'export_sf_to_onnx'`.

- [ ] **Step 3: Write the pure helper in `scripts/export_sf_to_onnx.py`**

Create `scripts/export_sf_to_onnx.py` with the module docstring + pure helper only:

```python
#!/usr/bin/env python3
"""Export a trained SampleFactory checkpoint to ONNX for the ncnn deploy pipeline.

Loads the latest SF checkpoint, rebuilds the ActorCritic, and wraps its actor path so forward()
returns the RAW action logits (length sum(nvec)) with godot_rl's ONNX IO naming
(input "obs"/"state_ins", output "output"/"state_outs"). scripts/export_to_ncnn.py then consumes
the ONNX unchanged; the deploy-side ActionDecode argmaxes per logit segment.

Scoped to the chase example (single Discrete(5) -> MultiDiscrete([5])); obs/action shape is
overridable on the CLI. normalize_input/normalize_returns must have been OFF at train time so the
actor is a plain MLP (see the design's parity note). Runs in .venv-sf.

Design: docs/superpowers/specs/2026-06-05-sample-factory-backend-design.md
"""
from __future__ import annotations

import argparse
from typing import Sequence


def actor_logit_layout(nvec: Sequence[int]) -> tuple[int, list[int]]:
    """Map a MultiDiscrete nvec to (total_logits, [n0, n1, ...]).

    The actor head emits total_logits = sum(nvec): one contiguous logit segment per discrete
    sub-action. Raises ValueError on an empty nvec or any non-positive entry.
    """
    dims = [int(n) for n in nvec]
    if len(dims) == 0:
        raise ValueError("actor_logit_layout: empty nvec (no discrete actions)")
    if any(n <= 0 for n in dims):
        raise ValueError(f"actor_logit_layout: non-positive entry in nvec {dims}")
    return sum(dims), dims
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_export_sf_to_onnx -v`
Expected: PASS (all four cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_sf_to_onnx.py test/python/test_export_sf_to_onnx.py
git commit -m "feat: export_sf_to_onnx.py actor_logit_layout helper (#24)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `export_sf_to_onnx.py` main() (checkpoint → ONNX)

**Files:**
- Modify: `scripts/export_sf_to_onnx.py`

This is the API-sensitive task — it depends on SampleFactory 2.1.1 internals. Verify each SF symbol against the installed package before trusting the recipe; adjust attribute names inline if introspection differs, keeping the ONNX IO contract (input `obs`, output `output`, raw logits) fixed.

- [ ] **Step 1: Introspect the installed SF 2.1.1 API**

Run:

```bash
.venv-sf/bin/python - <<'PY'
import inspect
from sample_factory.model.actor_critic import create_actor_critic, ActorCritic
from sample_factory.algo.learning.learner import Learner
from sample_factory.cfg.arguments import load_from_checkpoint
print("create_actor_critic:", inspect.signature(create_actor_critic))
print("Learner.load_checkpoint:", inspect.signature(Learner.load_checkpoint))
print("ActorCritic methods:", [m for m in dir(ActorCritic) if m.startswith("forward") or "normaliz" in m])
PY
```

Expected: prints `create_actor_critic(cfg, obs_space, action_space)`, a `Learner.load_checkpoint(checkpoints, device)` static/classmethod, and `ActorCritic` exposing `forward_head`, `forward_core`, `forward_tail`, and a `normalize_obs`-style method. **If the names differ, note them and use the actual names in Step 2.**

- [ ] **Step 2: Implement `main()`**

Append to `scripts/export_sf_to_onnx.py`:

```python
def _latest_checkpoint(train_dir: str, experiment: str) -> str:
    """Newest SF checkpoint .pth under <train_dir>/<experiment>/checkpoint_p0/."""
    import glob
    import os

    ckpt_dir = os.path.join(train_dir, experiment, "checkpoint_p0")
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")), key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError(f"no SF checkpoint .pth under {ckpt_dir}")
    return cands[-1]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False, description="SF checkpoint -> ONNX (chase).")
    p.add_argument("--train_dir", type=str, default="logs/sf")
    p.add_argument("--experiment", type=str, default="chase_sf")
    p.add_argument("--obs_dim", type=int, default=5)
    p.add_argument("--nvec", type=int, nargs="+", default=[5])
    p.add_argument("--out", type=str, default="models/chase_sf_policy.onnx")
    return p.parse_args(argv)


def main(argv=None) -> int:
    # Lazy heavy imports (only when exporting).
    import pathlib

    import numpy as np
    import torch
    import torch.nn as nn
    from gymnasium import spaces

    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.algo.learning.learner import Learner
    from sample_factory.cfg.arguments import load_from_checkpoint
    from sample_factory.algo.utils.context import sf_global_context  # noqa: F401  (ensures registry init)

    args = parse_args(argv)
    total_logits, nvec = actor_logit_layout(args.nvec)

    # GodotEnv exposes obs as a Dict({"obs": Box}); action is MultiDiscrete(nvec).
    obs_space = spaces.Dict({"obs": spaces.Box(low=-np.inf, high=np.inf, shape=(args.obs_dim,), dtype=np.float32)})
    action_space = spaces.MultiDiscrete(np.array(nvec, dtype=np.int64))

    # Minimal cfg: load_from_checkpoint reconstructs the run's cfg from the saved config.json.
    base_cfg = argparse.Namespace(train_dir=args.train_dir, experiment=args.experiment)
    cfg = load_from_checkpoint(base_cfg)

    device = torch.device("cpu")
    actor_critic = create_actor_critic(cfg, obs_space, action_space).to(device).eval()

    ckpt_path = _latest_checkpoint(args.train_dir, args.experiment)
    print("loading SF checkpoint:", ckpt_path)
    ckpt = Learner.load_checkpoint([ckpt_path], device)
    actor_critic.load_state_dict(ckpt["model"])

    class OnnxableActor(nn.Module):
        """obs -> (head -> core(identity, no RNN) -> tail) -> raw action logits, with vestigial state."""

        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, obs, state_ins):
            normalized = self.inner.normalize_obs({"obs": obs})
            head = self.inner.forward_head(normalized)
            core_out, _ = self.inner.forward_core(head, state_ins)
            tail = self.inner.forward_tail(core_out, values_only=False, sample_actions=False)
            return tail["action_logits"], state_ins

    onnxable = OnnxableActor(actor_critic).to(device).eval()
    dummy_obs = torch.zeros(1, args.obs_dim)
    out_path = pathlib.Path(args.out).with_suffix(".onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        onnxable,
        args=(dummy_obs, torch.zeros(1).float()),
        f=str(out_path),
        opset_version=17,
        input_names=["obs", "state_ins"],
        output_names=["output", "state_outs"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "state_ins": {0: "batch_size"},
            "output": {0: "batch_size"},
            "state_outs": {0: "batch_size"},
        },
    )
    # Sanity: forward shape must be (1, total_logits).
    with torch.no_grad():
        logits, _ = onnxable(dummy_obs, torch.zeros(1).float())
    assert logits.shape == (1, total_logits), f"expected (1,{total_logits}) got {tuple(logits.shape)}"
    print("exported ONNX to:", out_path, "logits:", total_logits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> **Implementer note:** `normalize_obs`, `forward_head/core/tail`, and the `"action_logits"` key are the SF 2.1.1 names verified in Step 1. If Step 1 showed different names (e.g. a different forward_tail signature or a `model_outputs` key), substitute them here — the only fixed contract is: ONNX input `obs` (shape `[1, obs_dim]`), output `output` = raw logits (length `sum(nvec)`), plus the vestigial `state_ins`/`state_outs`.

- [ ] **Step 3: Re-run the pure-helper test (must still pass)**

Run: `.venv-train/bin/python -m unittest test.python.test_export_sf_to_onnx -v`
Expected: PASS (the new lazy imports must not break the helper import).

- [ ] **Step 4: Verify it imports under the SF venv**

Run: `.venv-sf/bin/python scripts/export_sf_to_onnx.py --help`
Expected: argparse help; exit 0. (Full export is validated end-to-end in Task 7.)

- [ ] **Step 5: Commit**

```bash
git add scripts/export_sf_to_onnx.py
git commit -m "feat: export_sf_to_onnx.py main() — SF checkpoint to ONNX actor logits (#24)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `train_sf.sh` orchestrator

**Files:**
- Create: `scripts/train_sf.sh`

- [ ] **Step 1: Write `scripts/train_sf.sh`**

Create `scripts/train_sf.sh` (mirrors `train_cleanrl.sh`, adds the export + ncnn-convert steps):

```bash
#!/usr/bin/env bash
# Orchestrates SampleFactory (async PPO) training over the godot-rl bridge, then exports the
# trained actor to ncnn:
#   1. start the SF trainer in .venv-sf (opens server on base_port+1, blocks until Godot connects)
#   2. launch the headless Godot chase training scene (connects as client on base_port+1)
#   3. wait for the trainer; kill Godot
#   4. export the SF checkpoint -> ONNX (.venv-sf)
#   5. convert ONNX -> ncnn + parity check (.venv)
# Third backend alongside train_chase.sh (SB3) and train_cleanrl.sh (CleanRL).
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

GODOT="${GODOT:-godot}"
PY_SF="${PY_SF:-.venv-sf/bin/python}"
PY_CONVERT="${PY_CONVERT:-.venv/bin/python}"
TIMESTEPS="${TIMESTEPS:-1000000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
BASE_PORT="${BASE_PORT:-11008}"
EXPERIMENT="${EXPERIMENT:-chase_sf}"
TRAIN_DIR="${TRAIN_DIR:-logs/sf}"
OUTDIR="${OUTDIR:-models}"
SCENE="res://examples/chase_the_target/chase_the_target_train.tscn"

CLIENT_PORT=$((BASE_PORT + 1))   # godot_rl single-worker offset (base_port + 1 + env_id, env_id=0)
ONNX_PATH="$OUTDIR/chase_sf_policy.onnx"

echo "Starting SampleFactory trainer (timesteps=$TIMESTEPS, base_port=$BASE_PORT)..."
"$PY_SF" scripts/train_sf.py --timesteps "$TIMESTEPS" --base_port "$BASE_PORT" \
	--speedup "$SPEEDUP" --experiment "$EXPERIMENT" --train_dir "$TRAIN_DIR" &
TRAINER_PID=$!

# Give the trainer a moment to bind the server socket before Godot connects.
sleep 5

echo "Launching headless Godot training scene on port $CLIENT_PORT..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" "port=$CLIENT_PORT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
set -e
echo "Trainer exited with code $TRAINER_RC"
[ "$TRAINER_RC" -eq 0 ] || exit "$TRAINER_RC"

echo "Exporting SF checkpoint -> ONNX..."
mkdir -p "$OUTDIR"
"$PY_SF" scripts/export_sf_to_onnx.py --train_dir "$TRAIN_DIR" --experiment "$EXPERIMENT" --out "$ONNX_PATH"

echo "Converting ONNX -> ncnn (+ parity)..."
"$PY_CONVERT" scripts/export_to_ncnn.py "$ONNX_PATH" --outdir "$OUTDIR"

echo "Done. ncnn model in $OUTDIR/ (chase_sf_policy.ncnn.param/.bin)"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/train_sf.sh`
Expected: no output; `test -x scripts/train_sf.sh && echo ok` prints `ok`.

- [ ] **Step 3: Verify the chase training scene accepts a `port=` cmdline arg**

Run: `grep -rn "port" addons/godot_native_rl/sync.gd | head` and confirm `NcnnSync` reads a `port=` override (the other backends rely on the default 11008; SF needs base_port+1).
Expected: a `port=` parse exists. **If it does NOT**, add it minimally to the scene's cmdline parsing the same way `speedup=`/`action_repeat=` are parsed, OR set `BASE_PORT=11007` so `CLIENT_PORT=11008` matches the default — prefer the latter (zero code change) and update Task 7 + the orchestrator default accordingly.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_sf.sh
git commit -m "feat: train_sf.sh orchestrator (train -> export -> ncnn) (#24)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: End-to-end SF smoke in `run_tests.sh`

**Files:**
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Add the guarded smoke step**

In `test/run_tests.sh`, insert this block immediately **before** the final `echo "All tests passed."` line:

```bash
echo "== SampleFactory backend smoke (skipped if .venv-sf absent) =="
if [ -x .venv-sf/bin/python ]; then
	SF_TMP="$(mktemp -d)"
	# Tiny run: enough env steps to write one checkpoint; serial/sync mode keeps it deterministic.
	TIMESTEPS="${SF_SMOKE_TIMESTEPS:-3000}" \
	TRAIN_DIR="$SF_TMP/logs" OUTDIR="$SF_TMP/models" EXPERIMENT="chase_sf_smoke" \
		./scripts/train_sf.sh
	test -f "$SF_TMP/models/chase_sf_policy.ncnn.param" || { echo "FAIL: SF ncnn .param not produced" >&2; rm -rf "$SF_TMP"; exit 1; }
	test -f "$SF_TMP/models/chase_sf_policy.ncnn.bin"   || { echo "FAIL: SF ncnn .bin not produced" >&2; rm -rf "$SF_TMP"; exit 1; }
	rm -rf "$SF_TMP"
	echo "SampleFactory smoke OK."
else
	echo "SKIP: .venv-sf not present (run scripts/setup_training.sh to enable the SF smoke)."
fi
```

> Note: `export_to_ncnn.py` already runs a parity check internally (and fails non-zero on mismatch), so asserting the `.ncnn.{param,bin}` files exist is sufficient — a parity failure aborts the orchestrator before this point.

- [ ] **Step 2: Run the SF smoke in isolation first (faster feedback than the full suite)**

Run: `SF_SMOKE_TIMESTEPS=3000 TRAIN_DIR=$(mktemp -d)/logs OUTDIR=$(mktemp -d)/models EXPERIMENT=chase_sf_smoke ./scripts/train_sf.sh`
Expected: SF trains briefly, exports ONNX, `export_to_ncnn.py` prints a passing parity check, and `chase_sf_policy.ncnn.param`/`.bin` are written. **If the SF↔Godot handshake hangs, recheck the port wiring (Task 6 Step 3); if export fails, recheck the SF API names (Task 5 Step 1).** Bump `SF_SMOKE_TIMESTEPS` if a checkpoint isn't written in time.

- [ ] **Step 3: Run the full suite**

Run: `./test/run_tests.sh`
Expected: all existing steps green, plus `SampleFactory smoke OK.` then `All tests passed.`

- [ ] **Step 4: Commit**

```bash
git add test/run_tests.sh
git commit -m "test: end-to-end SampleFactory smoke in run_tests.sh (auto-skips w/o .venv-sf) (#24)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Docs (same-change, per repo convention)

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`, `docs/BACKLOG.md`

- [ ] **Step 1: `CLAUDE.md` — add the key command**

Under "## Key commands", after the CleanRL backend bullet (`**Train (chase, CleanRL backend):**`), add:

```markdown
- **Train (chase, SampleFactory backend):** `./scripts/train_sf.sh` — SampleFactory async PPO over
  godot_rl's `sample_factory_training` (same chase scene; serial/sync + `normalize_input=False` so
  the actor is a plain MLP). Runs in the isolated **`.venv-sf`**; exports the SF checkpoint via
  `export_sf_to_onnx.py` → ncnn. `TIMESTEPS`/`BASE_PORT`/`EXPERIMENT`/`OUTDIR` overrides.
```

- [ ] **Step 2: `CLAUDE.md` — update the venv gotcha (two → three)**

In "## Operational gotchas", change the `**Two venvs**` bullet to:

```markdown
- **Three venvs** — `.venv` (3.14, pnnx+torch) convert; `.venv-train` (3.13, godot-rl+SB3) train;
  `.venv-sf` (3.13, SampleFactory — pins gymnasium<1.0, so isolated) for the SF backend only.
  Create all with `./scripts/setup_training.sh`.
```

- [ ] **Step 3: `CLAUDE.md` — mark item 18 done**

In "## Roadmap & backlog" → "Done:" list, append `, 18 (SampleFactory training backend)` to the running list.

- [ ] **Step 4: `README.md` — list the third backend**

Find the section listing training backends (search: `grep -n "CleanRL" README.md`). Add a SampleFactory entry alongside SB3 and CleanRL, e.g.:

```markdown
- **SampleFactory** (`scripts/train_sf.sh`) — async PPO; trains the chase example in the isolated
  `.venv-sf`, exports to ncnn unchanged. Demonstrates a third, architecturally distinct backend.
```

(Match the surrounding list's exact formatting.)

- [ ] **Step 5: `docs/godot-rl-gap-analysis-2026-06-02.md` — flip the SampleFactory row**

Run: `grep -ni "sample" docs/godot-rl-gap-analysis-2026-06-02.md`
Mark the SampleFactory-backend item as done/available (match the file's checkbox or status convention; e.g. `[ ]` → `[x]`).

- [ ] **Step 6: `docs/BACKLOG.md` — tick item 18**

Run: `grep -n "18" docs/BACKLOG.md` to find the item-18 line; flip its checkbox `- [ ]` → `- [x]`.

- [ ] **Step 7: Verify docs reference real paths**

Run: `for f in scripts/train_sf.sh scripts/train_sf.py scripts/export_sf_to_onnx.py requirements-sf.txt; do test -e "$f" && echo "ok $f" || echo "MISSING $f"; done`
Expected: four `ok` lines.

- [ ] **Step 8: Commit**

```bash
git add README.md CLAUDE.md docs/godot-rl-gap-analysis-2026-06-02.md docs/BACKLOG.md
git commit -m "docs: SampleFactory backend — README/CLAUDE/gap-analysis/backlog (#24)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Run the full suite once more**

Run: `./test/run_tests.sh`
Expected: all green, including `SampleFactory smoke OK.`

- [ ] **Confirm the SB3/CleanRL backends are untouched** (gymnasium 1.0.0 in `.venv-train`)

Run: `.venv-train/bin/pip show gymnasium | grep Version`
Expected: `Version: 1.0.0` (the SF venv's 0.29.1 is isolated in `.venv-sf`).

- [ ] **Open the PR**

```bash
git push -u origin feat/sample-factory-backend
gh pr create --title "feat: SampleFactory training backend (#24)" --body "$(cat <<'EOF'
Third training backend (after SB3 + CleanRL): SampleFactory async PPO over the godot_rl bridge,
trained on the chase example and exported to ncnn unchanged.

- Isolated in a new `.venv-sf` (SF pins gymnasium<1.0; keeps the SB3/CleanRL stack on 1.0.0).
- `train_sf.py` drives godot_rl's `sample_factory_training` with macOS-safe / parity-safe overrides.
- `export_sf_to_onnx.py` turns the SF checkpoint into ncnn-ready ONNX (raw actor logits, obs->output).
- `train_sf.sh` chains train -> export -> ncnn; end-to-end smoke in run_tests.sh (auto-skips w/o .venv-sf).

Closes #24.

Design: docs/superpowers/specs/2026-06-05-sample-factory-backend-design.md
Plan: docs/superpowers/plans/2026-06-05-sample-factory-backend.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Notes / known risks (carried from the spec)

- **SF API drift (Task 5):** the exporter is pinned to SampleFactory 2.1.1 internals. Step 1 of Task 5 introspects them first — trust the introspection over the recipe if they diverge.
- **macOS multiprocessing:** the serial/sync overrides in `build_sf_argv` are what make SF reliable here; do not "optimize" them away in the smoke.
- **Port wiring:** the SF↔Godot handshake hinges on the `base_port + 1` offset (Task 6 Step 3). If `NcnnSync` can't take a `port=` arg, use the `BASE_PORT=11007` workaround so the client lands on the default 11008.
- **Throughput is out of scope** — this backend is dialed down for correctness; the parallel-arena throughput showcase is a separate follow-up.

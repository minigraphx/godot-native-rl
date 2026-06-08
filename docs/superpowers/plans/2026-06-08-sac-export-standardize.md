# SAC Export Standardize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the SAC actor→TorchScript export out of `train_ball_chase.py` into a reusable, standalone-runnable module; document the torch-2.x dynamo breakage plus the verified `dynamo=False` fallback; guard both paths with a regression test.

**Architecture:** A new focused module `scripts/export_sac_torchscript.py` owns `export_sac_actor_as_torchscript` (moved verbatim) plus a thin `--checkpoint` CLI that loads a SAC `.zip` and exports without a training run. `train_ball_chase.py` imports the helper instead of embedding it. A new test file covers pure helpers, the TorchScript round-trip, and the `dynamo=False` ONNX fallback. Docs record the rationale.

**Tech Stack:** Python 3.13 (`.venv-train`: torch 2.12, stable-baselines3 2.4), pnnx (`.venv`), `unittest` (stdlib), Godot 4.5/4.6 headless harness (unaffected).

**Branch:** `feat/sac-export-standardize` (already created off `origin/main`).

**Spec:** `docs/superpowers/specs/2026-06-08-sac-export-standardize-design.md`

---

## File Structure

- **Create** `scripts/export_sac_torchscript.py` — SAC actor→TorchScript export module + CLI. Pure helpers (`latest_checkpoint`, `parse_args`) at top; torch-lazy `export_sac_actor_as_torchscript` + `main()`.
- **Create** `test/python/test_export_sac_torchscript.py` — pure tests + torch-gated round-trip + `dynamo=False` finding guard.
- **Modify** `scripts/train_ball_chase.py` — delete the embedded helper, import it from the new module.
- **Modify** `docs/ncnn_vs_onnx.md` — add a SAC/TorchScript fidelity note.
- **Modify** `docs/dev/DEVELOPMENT.md` — one-line pointer to the new script.
- **Modify** `CLAUDE.md` — update the BallChase/SAC bullet; add #81 to Done list.

---

## Task 1: New module — pure helpers (TDD)

**Files:**
- Create: `scripts/export_sac_torchscript.py`
- Test: `test/python/test_export_sac_torchscript.py`

- [ ] **Step 1: Write the failing pure tests**

Create `test/python/test_export_sac_torchscript.py`:

```python
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_sac_torchscript as m  # noqa: E402


class TestLatestCheckpoint(unittest.TestCase):
    def test_missing_dir_returns_empty(self):
        self.assertEqual(m.latest_checkpoint("/nonexistent/dir/xyz"), "")

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(m.latest_checkpoint(d), "")

    def test_picks_newest_by_mtime(self):
        import os
        import time
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "ball_chase_ckpt_5000_steps.zip"
            new = Path(d) / "ball_chase_ckpt_25000_steps.zip"
            old.touch()
            time.sleep(0.01)
            new.touch()
            # Force distinct mtimes regardless of fs granularity.
            os.utime(old, (1, 1))
            os.utime(new, (2, 2))
            self.assertEqual(m.latest_checkpoint(d), str(new))


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        a = m.parse_args([])
        self.assertEqual(a.checkpoint, "")
        self.assertEqual(a.checkpoint_dir, "models/ball_chase_checkpoints")
        self.assertEqual(a.pt_export_path, "models/ball_chase_sac.pt")

    def test_overrides(self):
        a = m.parse_args(["--checkpoint", "x.zip", "--pt_export_path", "out.pt"])
        self.assertEqual(a.checkpoint, "x.zip")
        self.assertEqual(a.pt_export_path, "out.pt")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv-train/bin/python -m pytest test/python/test_export_sac_torchscript.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'export_sac_torchscript'`.

- [ ] **Step 3: Create the module with pure helpers + the moved export helper**

Create `scripts/export_sac_torchscript.py`:

```python
#!/usr/bin/env python3
"""Export a saved SB3 **SAC** checkpoint's deterministic actor to TorchScript (.pt) + shape sidecar.

The SAC counterpart of `export_torchscript.py` (PPO). godot_rl's `export_model_as_onnx` cannot
export SAC under torch>=2.x: `torch.onnx.export` routes the actor through the dynamo/torch.export
path, which fails constructing the action `Normal(mean, std)` (GuardOnDataDependentSymNode). We
instead torch.jit.trace the deterministic actor `tanh(mu(latent_pi(extract_features(obs))))`
directly -- no distribution is built, so no guard fires -- and feed the `.pt` (+ shape sidecar) to
`export_to_ncnn.py`'s `--via torchscript` pnnx path. The exported actor is tanh(mean); the deploy
side must NOT squash again (see ball_chase_agent.gd). The legacy `dynamo=False` ONNX exporter also
works (parity ~2e-8) but is deprecated in torch>=2.9, so TorchScript is the recommended route.

Run under .venv-train (SB3 + torch).

Usage:
    .venv-train/bin/python scripts/export_sac_torchscript.py --checkpoint models/ball_chase_sac.zip
    .venv-train/bin/python scripts/export_sac_torchscript.py   # latest in models/ball_chase_checkpoints
then:
    .venv-train/bin/python scripts/export_to_ncnn.py models/ball_chase_sac.pt --via torchscript
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Reuse the sidecar writer from the converter (import-light: no torch at module load).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_to_ncnn import write_shape_sidecar  # noqa: E402


def latest_checkpoint(checkpoint_dir: str) -> str:
    """Newest `*.zip` (by mtime) in `checkpoint_dir`, or "" if none. Pure (no torch)."""
    zips = sorted(pathlib.Path(checkpoint_dir).glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return str(zips[-1]) if zips else ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args (argv defaults to sys.argv); pure + testable."""
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument("--checkpoint", type=str, default="",
                   help="path to a SAC checkpoint .zip; defaults to the latest in --checkpoint_dir")
    p.add_argument("--checkpoint_dir", type=str, default="models/ball_chase_checkpoints")
    p.add_argument("--pt_export_path", type=str, default="models/ball_chase_sac.pt")
    return p.parse_args(argv)


def export_sac_actor_as_torchscript(model, pt_path: pathlib.Path):
    """Trace SAC's deterministic actor `tanh(mu(latent_pi(features)))` to `pt_path` + sidecar.

    Returns (pt_path, sidecar_path). Equivalent to actor(obs, deterministic=True) but built
    without the action distribution, so torch.jit.trace stays on the legacy path (avoids the
    dynamo GuardOnDataDependentSymNode that breaks torch.onnx.export for SAC).
    """
    import torch

    actor = model.policy.actor.to("cpu")
    actor.eval()

    class DeterministicSacActor(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, obs):
            features = self.actor.extract_features(obs, self.actor.features_extractor)
            return torch.tanh(self.actor.mu(self.actor.latent_pi(features)))

    shape = (1, *model.observation_space.shape)
    dummy = torch.zeros(*shape, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(DeterministicSacActor(actor).eval(), dummy)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(pt_path))
    sidecar = write_shape_sidecar(pt_path, list(shape))
    return pt_path, sidecar


def main() -> None:
    from stable_baselines3 import SAC

    args = parse_args()
    ckpt = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    if not ckpt or not pathlib.Path(ckpt).is_file():
        raise SystemExit("No checkpoint found (looked for %s)" % (args.checkpoint or args.checkpoint_dir))

    model = SAC.load(ckpt)  # no env needed: export only touches the policy network
    print("Loaded checkpoint:", ckpt, "(num_timesteps=%d)" % model.num_timesteps)

    pt_path = pathlib.Path(args.pt_export_path).with_suffix(".pt")
    pt_path, sidecar = export_sac_actor_as_torchscript(model, pt_path)
    print("Exported TorchScript (deterministic actor = tanh(mean)) to:", pt_path)
    print("Wrote shape sidecar:   ", sidecar)
    print("Next: .venv-train/bin/python scripts/export_to_ncnn.py %s --via torchscript" % pt_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the pure tests to verify they pass**

Run: `.venv-train/bin/python -m pytest test/python/test_export_sac_torchscript.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_sac_torchscript.py test/python/test_export_sac_torchscript.py
git commit -m "feat: standalone SAC actor->TorchScript export module (#81)"
```

---

## Task 2: torch-gated round-trip + dynamo=False finding guard (TDD)

**Files:**
- Test: `test/python/test_export_sac_torchscript.py` (append)

- [ ] **Step 1: Append the gated tests**

Add to `test/python/test_export_sac_torchscript.py` (above the `if __name__` block):

```python
def _sac_stack_available() -> bool:
    try:
        import torch  # noqa: F401
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
        return True
    except Exception:
        return False


def _build_tiny_sac():
    """A tiny real SB3 SAC over a dummy Box(5)->Box(2) env (no env interaction needed)."""
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC

    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    return SAC("MlpPolicy", DummyEnv(), learning_starts=0, buffer_size=100, verbose=0)


@unittest.skipUnless(_sac_stack_available(), "torch/gymnasium/sb3 missing")
class TestSacTorchscriptRoundTrip(unittest.TestCase):
    def test_traced_actor_matches_eager_tanh_mean(self):
        import torch
        model = _build_tiny_sac()
        actor = model.policy.actor.to("cpu")
        actor.eval()
        obs = torch.zeros(1, 5, dtype=torch.float32)
        with torch.no_grad():
            feats = actor.extract_features(obs, actor.features_extractor)
            eager = torch.tanh(actor.mu(actor.latent_pi(feats))).numpy().reshape(-1)

        with tempfile.TemporaryDirectory() as d:
            pt = Path(d) / "sac.pt"
            out_pt, sidecar = m.export_sac_actor_as_torchscript(model, pt)
            self.assertTrue(out_pt.is_file())
            self.assertTrue(Path(sidecar).is_file())
            loaded = torch.jit.load(str(out_pt))
            with torch.no_grad():
                got = loaded(obs).numpy().reshape(-1)

        import numpy as np
        self.assertTrue(np.allclose(got, eager, atol=1e-6), f"{got} vs {eager}")

    def test_sidecar_records_input_shape(self):
        import json
        model = _build_tiny_sac()
        with tempfile.TemporaryDirectory() as d:
            pt = Path(d) / "sac.pt"
            _, sidecar = m.export_sac_actor_as_torchscript(model, pt)
            data = json.loads(Path(sidecar).read_text())
        # write_shape_sidecar records the traced input shape; assert the obs width.
        flat = json.dumps(data)
        self.assertIn("5", flat)


@unittest.skipUnless(_sac_stack_available(), "torch/gymnasium/sb3 missing")
class TestDynamoFalseFallbackGuard(unittest.TestCase):
    """Guards the documented finding: legacy `dynamo=False` ONNX export still works for SAC.

    If a future torch removes the legacy exporter this fails, flagging the doc claim as stale.
    We deliberately do NOT assert the default (dynamo) path raises -- that error is version-brittle.
    """
    def test_legacy_onnx_export_works_and_matches_eager(self):
        try:
            import onnxruntime as ort
        except Exception:
            self.skipTest("onnxruntime missing")
        import numpy as np
        import torch
        model = _build_tiny_sac()
        actor = model.policy.actor.to("cpu")
        actor.eval()

        class ActorWrapper(torch.nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, obs):
                return self.actor(obs, deterministic=True)

        wrapper = ActorWrapper(actor).eval()
        obs = torch.zeros(1, 5, dtype=torch.float32)
        with torch.no_grad():
            ref = wrapper(obs).numpy().reshape(-1)

        with tempfile.TemporaryDirectory() as d:
            onnx_path = Path(d) / "sac.onnx"
            torch.onnx.export(
                wrapper, (obs,), str(onnx_path),
                input_names=["input"], output_names=["output"], opset_version=17,
                dynamo=False,
            )
            sess = ort.InferenceSession(str(onnx_path))
            out = np.array(sess.run(None, {sess.get_inputs()[0].name: obs.numpy()})[0]).reshape(-1)
        self.assertTrue(np.allclose(out, ref, atol=1e-5), f"{out} vs {ref}")
```

- [ ] **Step 2: Run the gated tests to verify they pass**

Run: `.venv-train/bin/python -m pytest test/python/test_export_sac_torchscript.py -q`
Expected: PASS (all tests; gated ones run because `.venv-train` has torch/sb3/gymnasium/onnxruntime). If `onnxruntime` is absent the fallback test self-skips — that is acceptable.

- [ ] **Step 3: Commit**

```bash
git add test/python/test_export_sac_torchscript.py
git commit -m "test: SAC TorchScript round-trip + dynamo=False fallback guard (#81)"
```

---

## Task 3: Refactor `train_ball_chase.py` to import the shared helper

**Files:**
- Modify: `scripts/train_ball_chase.py`

- [ ] **Step 1: Delete the embedded helper and import it instead**

In `scripts/train_ball_chase.py`, remove the entire `def export_sac_actor_as_torchscript(model, pt_path):` function body (the whole def, including its docstring and the nested `DeterministicSacActor` class).

Then replace the existing import line (the moved helper was the only user of `write_shape_sidecar` in this file, so the old import is now dead — swap it, don't keep both):

```python
from export_to_ncnn import write_shape_sidecar  # noqa: E402
```

with:

```python
from export_sac_torchscript import export_sac_actor_as_torchscript  # noqa: E402
```

(The `sys.path.insert(0, ...)` above this import already puts `scripts/` on the path, so `export_sac_torchscript` resolves. If anything else in `train_ball_chase.py` still references `write_shape_sidecar`, keep the original import too — but with the helper moved it should not.)

Leave the call site in `main()` unchanged:

```python
    pt_path = pathlib.Path(args.pt_export_path).with_suffix(".pt")
    export_sac_actor_as_torchscript(model, pt_path)
    print("Exported TorchScript (deterministic actor = tanh(mean)) to:", pt_path)
```

- [ ] **Step 2: Verify the existing ball_chase tests still pass**

Run: `.venv-train/bin/python -m pytest test/python/test_train_ball_chase.py test/python/test_export_sac_torchscript.py -q`
Expected: PASS. `test_train_ball_chase.py` tests only the pure helpers (`latest_checkpoint`, `remaining_timesteps`, `parse_args`) that remain in `train_ball_chase.py`, so the move doesn't break it.

- [ ] **Step 3: Verify the module still imports cleanly (no leftover references)**

Run: `.venv-train/bin/python -c "import sys; sys.path.insert(0,'scripts'); import train_ball_chase; print(train_ball_chase.export_sac_actor_as_torchscript.__module__)"`
Expected: prints `export_sac_torchscript` (confirms the name now resolves via the import, not a local def).

- [ ] **Step 4: Commit**

```bash
git add scripts/train_ball_chase.py
git commit -m "refactor: train_ball_chase imports shared SAC export helper (#81)"
```

---

## Task 4: Documentation

**Files:**
- Modify: `docs/ncnn_vs_onnx.md`
- Modify: `docs/dev/DEVELOPMENT.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the fidelity note to `docs/ncnn_vs_onnx.md`**

In `docs/ncnn_vs_onnx.md`, in the "Current limitations of this project (truth in advertising)" list (around line 226), add a new bullet after the existing "All godot_rl action types deploy" bullet:

```markdown
- **SAC (and distribution-based continuous actors) export via TorchScript, not ONNX.** Under torch
  ≥2.x, `torch.onnx.export` routes the SAC actor through the dynamo / `torch.export` path, which
  cannot guard the action-distribution construction `Normal(mean, std)`
  (`GuardOnDataDependentSymNode`). We instead `torch.jit.trace` the deterministic actor
  `tanh(mean)` (`scripts/export_sac_torchscript.py`) → pnnx → ncnn. The legacy `dynamo=False` ONNX
  exporter still works (verified parity ~2e-8) but is deprecated in torch ≥2.9, so TorchScript is
  the recommended route. (PPO/A2C discrete and continuous still export cleanly via ONNX.)
```

- [ ] **Step 2: Add the pointer to `docs/dev/DEVELOPMENT.md`**

In `docs/dev/DEVELOPMENT.md`, in the export-contract section, immediately after the `continuous → … SAC's squashed-Gaussian deploys as tanh(mean) via the squash flag.` bullet (around line 105), add a sub-note:

```markdown
    - SAC's actor is exported with `scripts/export_sac_torchscript.py` (traces `tanh(mean)` to
      TorchScript → pnnx); `torch.onnx.export` can't export SAC under torch ≥2.x (dynamo guard on
      `Normal(mean, std)`). See `docs/ncnn_vs_onnx.md` and issue #81.
```

- [ ] **Step 3: Update the BallChase/SAC bullet in `CLAUDE.md`**

In `CLAUDE.md`, find the "Train (BallChase, SAC)" key-command bullet. Append a sentence pointing at the standalone exporter:

Locate:

```
  SAC ONNX export breaks under torch 2.x dynamo), then `scripts/export_to_ncnn.py models/ball_chase_sac.pt
  --via torchscript`.
```

Replace with:

```
  SAC ONNX export breaks under torch 2.x dynamo), then `scripts/export_to_ncnn.py models/ball_chase_sac.pt
  --via torchscript`. Re-export a saved SAC checkpoint without retraining via
  `scripts/export_sac_torchscript.py --checkpoint models/ball_chase_sac.zip` (see issue #81 / `docs/ncnn_vs_onnx.md`).
```

- [ ] **Step 4: Add #81 to the CLAUDE.md Done list**

In `CLAUDE.md`, in the "Roadmap & backlog" → Done list, add an entry consistent with the existing GitHub-issue-style notes (e.g. near the #74 line):

```
    GitHub #81 (SAC ONNX export broken under torch 2.x — standardized on TorchScript: promoted
    `export_sac_actor_as_torchscript` into `scripts/export_sac_torchscript.py` with a standalone
    `--checkpoint` CLI; documented + test-guarded the `dynamo=False` legacy-ONNX fallback. Note:
    GitHub issue #81.)
```

- [ ] **Step 5: Commit**

```bash
git add docs/ncnn_vs_onnx.md docs/dev/DEVELOPMENT.md CLAUDE.md
git commit -m "docs: SAC exports via TorchScript (torch-2.x dynamo); point at export_sac_torchscript (#81)"
```

---

## Task 5: Full suite + PR

**Files:** none (verification + git)

- [ ] **Step 1: Run the full test suite**

Run: `./test/run_tests.sh`
Expected: ends with `All tests passed.` and exit code 0. (Gate on that line / exit code — do NOT grep for `failed`/`ERROR`, both appear in passing runs.)

- [ ] **Step 2: Rebase onto latest origin/main**

```bash
git fetch origin
git rebase origin/main
```
Expected: clean rebase (this branch only touches SAC export + docs; no overlap with the open web-wasm PR #91). If `CLAUDE.md` conflicts, keep both sides' content and re-run Step 1.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin feat/sac-export-standardize
gh pr create --title "feat: standardize SAC ncnn export on TorchScript (#81)" --body "$(cat <<'EOF'
## Summary
- Promote `export_sac_actor_as_torchscript` out of `train_ball_chase.py` into a standalone, reusable `scripts/export_sac_torchscript.py` with a `--checkpoint` CLI (re-export a SAC checkpoint without retraining).
- `train_ball_chase.py` now imports the shared helper (behavior identical).
- Document that SAC / distribution-based continuous actors export via TorchScript because `torch.onnx.export` fails under torch ≥2.x dynamo (`Normal(mean, std)` guard), with the verified `dynamo=False` legacy-ONNX fallback (parity ~2e-8, deprecated in torch ≥2.9).
- Regression test guards both the TorchScript round-trip and the `dynamo=False` fallback.

## Test plan
- [ ] `.venv-train/bin/python -m pytest test/python/test_export_sac_torchscript.py test/python/test_train_ball_chase.py -q`
- [ ] `./test/run_tests.sh` green

Closes #81
EOF
)"
```

- [ ] **Step 4: Confirm CI is green**

Run: `gh pr checks --watch` (or `gh pr view --json statusCheckRollup`)
Expected: all checks pass.

---

## Self-Review Notes

- **Spec coverage:** new module (Task 1) ✓; standalone CLI (Task 1) ✓; helper moved + imported (Task 3) ✓; pure + round-trip + dynamo-guard tests (Tasks 1–2) ✓; docs in ncnn_vs_onnx/DEVELOPMENT/CLAUDE (Task 4) ✓; `Closes #81` (Task 5) ✓; no BACKLOG.md entry (verified #81 is GitHub-only) ✓.
- **Out of scope honored:** no `--via onnx-legacy` CLI; no live training run; `export_torchscript.py` (PPO) untouched.
- **Type/name consistency:** `latest_checkpoint`, `parse_args`, `export_sac_actor_as_torchscript` names identical across module, tests, and the `train_ball_chase.py` import.

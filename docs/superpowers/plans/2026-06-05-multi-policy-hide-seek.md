# Multi-policy trained example (Hide & Seek) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a trained Hide & Seek example where the seeker and hider learn two *distinct* policies (via a custom multi-policy PPO trainer), exported to two native ncnn models with golden-inference + behavioral regressions.

**Architecture:** A single-file CleanRL-style PPO (`scripts/train_hide_seek_multipolicy.py`) drives `CleanRLGodotEnv` (which vectorizes over the N Godot agents as N parallel envs). It reads `agent_policy_names` from the underlying env, builds a name→agent-index routing table, and maintains one PPO learner per distinct policy name. Each step it slices the batched obs per policy, runs each network, stitches actions back in agent order, and stores transitions in per-policy buffers. Distinct `policy_name`s are produced by a `--multi-policy` cmdline gate in `HideSeekAgent` so a single world scene serves both the existing shared-policy run and the new multi-policy run.

**Tech Stack:** Python 3.13 + torch + numpy + godot_rl (`.venv-train`); Godot 4.5 GDScript; ncnn via `NcnnRunner`; pnnx via `export_to_ncnn.py`.

**Spec:** `docs/superpowers/specs/2026-06-05-multi-policy-trained-example-design.md`. **Issue:** #26. **Follow-up:** #73.

---

## File Structure

**Create:**
- `scripts/train_hide_seek_multipolicy.py` — multi-policy PPO trainer (imports pure helpers + `export_actor_as_onnx` from `train_cleanrl`).
- `scripts/train_hide_seek_multipolicy.sh` — orchestrator; launches Godot with `--multi-policy`.
- `examples/hide_and_seek/hide_and_seek_multipolicy_train.tscn` — single-world train scene.
- `examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn` — `ParallelArena2D` train scene.
- `examples/hide_and_seek/hide_and_seek_multipolicy_eval.tscn` — both agents `NCNN_INFERENCE`, distinct models (behavioral check).
- `test/python/test_train_hide_seek_multipolicy.py` — pure-helper unit tests.
- `test/integration/run_hide_seek_multipolicy_smoke_test.py` — wire smoke (asserts `agent_policy_names`).
- `test/integration/trained_hide_seek_multipolicy_checker.gd` — behavioral checker node.
- `test/unit/test_hide_seek_multipolicy_golden_inference.gd` — golden-inference regression (two models).
- `examples/hide_and_seek/models/hide_seek_seeker.ncnn.{param,bin}` + `hide_seek_hider.ncnn.{param,bin}` — trained artifacts (committed after the training run).

**Modify:**
- `examples/hide_and_seek/hide_seek_math.gd` — add pure `policy_name_for(is_seeker, cmdline_args)`.
- `examples/hide_and_seek/hide_seek_agent.gd` — call it in `_ready()`.
- `test/unit/test_hide_seek_math.gd` — test the new pure helper.
- `test/run_tests.sh` — wire the smoke test + behavioral scene.
- `examples/hide_and_seek/README.md`, `CLAUDE.md`, `docs/BACKLOG.md` — docs + closeout.

---

## Phase 0: Prerequisites & API verification

### Task 0: Set up training env and confirm the `CleanRLGodotEnv` policy-name access path

**Files:** none (verification only).

- [ ] **Step 1: Create the two venvs**

Run: `./scripts/setup_training.sh`
Expected: creates `.venv` (convert) and `.venv-train` (train); exits 0.

- [ ] **Step 2: Confirm the agent-policy-names attribute path on `CleanRLGodotEnv`**

The trainer needs the per-agent policy names. `CleanRLGodotEnv` wraps the raw `GodotEnv`, which stores the names parsed from `env_info`. Confirm the exact attribute by inspecting the installed package:

Run: `.venv-train/bin/python -c "import inspect, godot_rl.wrappers.clean_rl_wrapper as m; print(inspect.getsourcefile(m))"`
Then `grep -rn "agent_policy_names\|policy_name" "$(.venv-train/bin/python -c 'import godot_rl,os;print(os.path.dirname(godot_rl.__file__))')"`

Expected: find where `agent_policy_names` is stored (e.g. `GodotEnv.agent_policy_names`). Record the access path; the trainer in Task 9 uses a small `_policy_names_from_env(env)` accessor — adjust that single function to the confirmed attribute. **If the names are not exposed by the wrapper**, fall back to reading them from the underlying env (`env._env` / `env.envs[0]`) — confirm which here.

- [ ] **Step 3: Confirm `reset`/`step` arity matches `train_cleanrl.py`**

`scripts/train_cleanrl.py:287,310` already calls `env.reset(seed)` → `(obs, info)` and `env.step(action_np)` → `(obs, reward, term, trunc, info)` with obs as a stacked `(num_envs, obs_dim)` array. Confirm the hide & seek scene behaves the same (it uses the same `NcnnSync`/protocol):

Run: `grep -n "def reset\|def step" "$(.venv-train/bin/python -c 'import godot_rl,os;print(os.path.dirname(godot_rl.__file__))')/wrappers/clean_rl_wrapper.py"`
Expected: signatures consistent with the calls in `train_cleanrl.py`. No code change — this de-risks Task 9.

- [ ] **Step 4: No commit** (verification only).

---

## Phase 1: Cmdline-gated policy identity

### Task 1: Pure helper `policy_name_for`

**Files:**
- Modify: `examples/hide_and_seek/hide_seek_math.gd`
- Test: `test/unit/test_hide_seek_math.gd`

- [ ] **Step 1: Write the failing test** — append to `test/unit/test_hide_seek_math.gd` inside its `_initialize()` (follow the file's existing `h.assert_*` style; `HideSeekMath` is already preloaded there):

```gdscript
	# policy_name_for: cmdline gate for multi-policy identity
	h.assert_eq(HideSeekMath.policy_name_for(true, ["res://x.tscn"]), "shared_policy", "no flag -> shared (seeker)")
	h.assert_eq(HideSeekMath.policy_name_for(false, ["res://x.tscn"]), "shared_policy", "no flag -> shared (hider)")
	h.assert_eq(HideSeekMath.policy_name_for(true, ["--multi-policy"]), "seeker", "flag + seeker -> seeker")
	h.assert_eq(HideSeekMath.policy_name_for(false, ["--multi-policy"]), "hider", "flag + hider -> hider")
	h.assert_eq(HideSeekMath.policy_name_for(true, ["speedup=8", "--multi-policy", "action_repeat=8"]), "seeker", "flag among other args")
```

- [ ] **Step 2: Run to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_math.gd`
Expected: FAIL — `Invalid call ... 'policy_name_for'` (method not defined).

- [ ] **Step 3: Implement the helper** — add to `examples/hide_and_seek/hide_seek_math.gd`:

```gdscript
# Multi-policy identity gate (see docs spec + issue #73). With "--multi-policy" on the cmdline the
# seeker/hider get DISTINCT policy names (two networks); otherwise both keep godot_rl's "shared_policy"
# default so the existing shared-policy example's wire handshake is unchanged.
static func policy_name_for(is_seeker: bool, cmdline_args: Array) -> String:
	if "--multi-policy" in cmdline_args:
		return "seeker" if is_seeker else "hider"
	return "shared_policy"
```

- [ ] **Step 4: Run to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_math.gd`
Expected: PASS (all assertions OK).

- [ ] **Step 5: Commit**

```bash
git add examples/hide_and_seek/hide_seek_math.gd test/unit/test_hide_seek_math.gd
git commit -m "feat: HideSeekMath.policy_name_for cmdline gate for multi-policy identity (#26)"
```

### Task 2: Wire the gate into `HideSeekAgent._ready()`

**Files:**
- Modify: `examples/hide_and_seek/hide_seek_agent.gd:31-35` (the `_ready()` body)

- [ ] **Step 1: Add the assignment** — in `HideSeekAgent._ready()`, after `super._ready()` and the `_game` lookup, add:

```gdscript
	# Distinct policy names only when launched with --multi-policy (see hide_seek_math.gd).
	policy_name = HideSeekMath.policy_name_for(is_seeker, OS.get_cmdline_args())
```

- [ ] **Step 2: Verify the shared-policy smoke test still passes (no flag → shared)**

Run: `PY=.venv/bin/python ./test/integration/run_hide_seek_smoke_test.py` (or via `GODOT=… .venv/bin/python test/integration/run_hide_seek_smoke_test.py`)
Expected: `HIDE&SEEK SMOKE TEST PASSED` — proves the existing scene (no `--multi-policy`) is unaffected.

- [ ] **Step 3: Commit**

```bash
git add examples/hide_and_seek/hide_seek_agent.gd
git commit -m "feat: HideSeekAgent derives policy_name from --multi-policy gate (#26)"
```

---

## Phase 2: Training scenes

### Task 3: Single-world multi-policy train scene

**Files:**
- Create: `examples/hide_and_seek/hide_and_seek_multipolicy_train.tscn`

- [ ] **Step 1: Create the scene** (identical wiring to `hide_and_seek_train.tscn`; the policy split comes from the cmdline flag, not the scene):

```
[gd_scene load_steps=3 format=3]

[ext_resource type="PackedScene" path="res://examples/hide_and_seek/hide_seek_world.tscn" id="1"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="2"]

[node name="HideSeekMultiPolicyTrain" type="Node2D"]

[node name="HideSeekWorld" parent="." instance=ExtResource("1")]

[node name="Sync" type="Node" parent="."]
script = ExtResource("2")
control_mode = 1
```

- [ ] **Step 2: Verify it loads + emits distinct policy names under the flag**

Run: `godot --headless --path . res://examples/hide_and_seek/hide_and_seek_multipolicy_train.tscn --multi-policy --quit-after 2`
Expected: no parse/load errors (it will warn about no trainer socket and time out — that's fine here; we only check it loads). Full wire assertion is Task 5.

- [ ] **Step 3: Commit**

```bash
git add examples/hide_and_seek/hide_and_seek_multipolicy_train.tscn
git commit -m "feat: single-world multi-policy hide&seek train scene (#26)"
```

### Task 4: Parallel multi-policy train scene

**Files:**
- Create: `examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn`

- [ ] **Step 1: Create the scene** (mirrors `hide_and_seek_train_parallel.tscn`; tiles the same world — distinct names still come from the cmdline flag, applied per tiled agent):

```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://addons/godot_native_rl/training/parallel_arena_2d.gd" id="1"]
[ext_resource type="PackedScene" path="res://examples/hide_and_seek/hide_seek_world.tscn" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="3"]

[node name="HideSeekMultiPolicyTrainParallel" type="Node2D"]

[node name="ParallelArena2D" type="Node2D" parent="."]
script = ExtResource("1")
world_scene = ExtResource("2")
count = 8
spacing = 1400.0

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 1
```

- [ ] **Step 2: Verify it loads**

Run: `godot --headless --path . res://examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn --multi-policy --quit-after 2`
Expected: loads without parse errors (socket timeout is fine).

- [ ] **Step 3: Commit**

```bash
git add examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn
git commit -m "feat: parallel multi-policy hide&seek train scene (#26)"
```

---

## Phase 3: Wire smoke test (proves the gate end-to-end pre-training)

### Task 5: Multi-policy wire smoke test

**Files:**
- Create: `test/integration/run_hide_seek_multipolicy_smoke_test.py`
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Write the test** (adapted from `run_hide_seek_smoke_test.py`; launches with `--multi-policy` and asserts the two distinct names arrive over the wire):

```python
#!/usr/bin/env python3
"""Multi-policy wire smoke test: launches the multi-policy train scene with --multi-policy and asserts
the env_info handshake carries agent_policy_names == ["seeker", "hider"] (two distinct policies),
n_agents == 2, and the step loop runs. Raw sockets (no trainer); modeled on run_hide_seek_smoke_test.py."""
import json
import os
import socket
import subprocess
import sys

HOST, PORT = "127.0.0.1", 11008
SCENE = "res://examples/hide_and_seek/hide_and_seek_multipolicy_train.tscn"
GODOT = os.environ.get("GODOT", "godot")
OBS_SIZE = 15


def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("socket closed early")
        buf += chunk
    return buf


def send(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(len(data).to_bytes(4, "little") + data)


def recv(sock):
    n = int.from_bytes(recvall(sock, 4), "little")
    return json.loads(recvall(sock, n).decode("utf-8"))


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(30)

    proc = None
    conn = None
    failures = []
    try:
        proc = subprocess.Popen(
            [GODOT, "--headless", "--path", ".", SCENE, "--multi-policy", "action_repeat=1", "speedup=1"]
        )
        conn, _ = server.accept()
        conn.settimeout(30)

        send(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})
        send(conn, {"type": "env_info"})
        info = recv(conn)
        if info.get("n_agents") != 2:
            failures.append("n_agents != 2 (got %r)" % info.get("n_agents"))
        names = info.get("agent_policy_names")
        if names != ["seeker", "hider"]:
            failures.append("agent_policy_names != ['seeker','hider'] (got %r)" % names)

        send(conn, {"type": "reset"})
        msg = recv(conn)
        if len(msg.get("obs") or []) != 2:
            failures.append("reset obs count != 2 (got %r)" % msg.get("obs"))

        for _ in range(3):
            send(conn, {"type": "action", "action": [{"move": 4}, {"move": 3}]})
            step = recv(conn)
            if step.get("type") != "step":
                failures.append("step type (got %r)" % step.get("type"))
                break

        send(conn, {"type": "close"})
    finally:
        if proc is not None:
            try:
                rc = proc.wait(timeout=15)
            except Exception:
                proc.kill()
                rc = -1
            if rc != 0:
                failures.append("godot exited with code %d" % rc)
        if conn is not None:
            conn.close()
        server.close()

    if failures:
        print("MULTI-POLICY SMOKE TEST FAILED:", failures)
        sys.exit(1)
    print("MULTI-POLICY SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run to verify it passes**

Run: `PY=.venv/bin/python .venv/bin/python test/integration/run_hide_seek_multipolicy_smoke_test.py` (with `GODOT` set if needed)
Expected: `MULTI-POLICY SMOKE TEST PASSED`. If `agent_policy_names` is `["shared_policy","shared_policy"]`, the cmdline gate (Task 2) or the `--multi-policy` launch arg is wrong — fix before continuing.

- [ ] **Step 3: Wire into `test/run_tests.sh`** — after the existing "Hide & seek self-play smoke test" block (line ~64), add:

```bash
echo "== Hide & seek MULTI-POLICY wire smoke test =="
PY="${PY:-.venv/bin/python}"
"$PY" test/integration/run_hide_seek_multipolicy_smoke_test.py
```

- [ ] **Step 4: Commit**

```bash
git add test/integration/run_hide_seek_multipolicy_smoke_test.py test/run_tests.sh
git commit -m "test: multi-policy wire smoke (agent_policy_names over the wire) (#26)"
```

---

## Phase 4: Trainer pure helpers (TDD)

### Task 6: `policy_index_map`

**Files:**
- Create: `scripts/train_hide_seek_multipolicy.py` (module skeleton + this helper)
- Test: `test/python/test_train_hide_seek_multipolicy.py`

- [ ] **Step 1: Write the failing test**

```python
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_hide_seek_multipolicy as mp  # noqa: E402


class TestPolicyIndexMap(unittest.TestCase):
    def test_single_world(self):
        self.assertEqual(mp.policy_index_map(["seeker", "hider"]),
                         {"seeker": [0], "hider": [1]})

    def test_parallel_interleaved(self):
        names = ["seeker", "hider", "seeker", "hider"]
        self.assertEqual(mp.policy_index_map(names),
                         {"seeker": [0, 2], "hider": [1, 3]})

    def test_first_seen_key_order(self):
        self.assertEqual(list(mp.policy_index_map(["hider", "seeker"]).keys()),
                         ["hider", "seeker"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_train_hide_seek_multipolicy -v` (or `cd` to repo root and `.venv-train/bin/python -m unittest discover -s test/python -p 'test_train_hide_seek_multipolicy.py'`)
Expected: `ModuleNotFoundError: No module named 'train_hide_seek_multipolicy'`.

- [ ] **Step 3: Create the module skeleton + helper** — `scripts/train_hide_seek_multipolicy.py`:

```python
#!/usr/bin/env python3
"""Train Hide & Seek with TWO distinct policies (seeker + hider) over the godot-rl bridge.

A custom single-file multi-policy PPO (sibling of scripts/train_cleanrl.py). CleanRLGodotEnv
vectorizes over the N Godot agents as N parallel envs; this trainer reads agent_policy_names, routes
each agent index to its policy, maintains one PPO learner per distinct name, and exports each actor
to ONNX (obs/state_ins -> output/state_outs) for scripts/export_to_ncnn.py -> native ncnn.

Run this FIRST (opens the server on 11008, waits), THEN launch the Godot scene with --multi-policy.
See scripts/train_hide_seek_multipolicy.sh. Design:
docs/superpowers/specs/2026-06-05-multi-policy-trained-example-design.md

Heavy imports (torch/numpy/godot_rl) are LAZY so the pure helpers stay unit-testable. Pure PPO
helpers (compute_gae, num_updates, layer_init) + the ONNX exporter are reused from train_cleanrl.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, NamedTuple, Sequence


def policy_index_map(agent_policy_names: Sequence[str]) -> Dict[str, List[int]]:
    """Map each distinct policy name to the agent indices using it (first-seen key order,
    ascending indices). e.g. ["seeker","hider","seeker"] -> {"seeker":[0,2],"hider":[1]}."""
    out: Dict[str, List[int]] = {}
    for i, name in enumerate(agent_policy_names):
        out.setdefault(name, []).append(i)
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv-train/bin/python -m unittest discover -s test/python -p 'test_train_hide_seek_multipolicy.py' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_hide_seek_multipolicy.py test/python/test_train_hide_seek_multipolicy.py
git commit -m "feat: policy_index_map routing helper for multi-policy trainer (#26)"
```

### Task 7: `split_by_policy` + `stitch_actions` (round-trip)

**Files:**
- Modify: `scripts/train_hide_seek_multipolicy.py`
- Test: `test/python/test_train_hide_seek_multipolicy.py`

- [ ] **Step 1: Write the failing test** — append:

```python
class TestSplitStitch(unittest.TestCase):
    def test_split_by_policy(self):
        index_map = {"seeker": [0, 2], "hider": [1, 3]}
        batched = np.array([[10.0], [11.0], [12.0], [13.0]])
        out = mp.split_by_policy(batched, index_map)
        np.testing.assert_array_equal(out["seeker"], np.array([[10.0], [12.0]]))
        np.testing.assert_array_equal(out["hider"], np.array([[11.0], [13.0]]))

    def test_stitch_is_inverse_of_split(self):
        index_map = {"seeker": [0, 2], "hider": [1, 3]}
        actions = np.array([[1], [2], [3], [4]], dtype=np.int64)  # (n_agents, action_dim)
        per_policy = mp.split_by_policy(actions, index_map)
        stitched = mp.stitch_actions(per_policy, index_map, n_agents=4)
        np.testing.assert_array_equal(stitched, actions)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv-train/bin/python -m unittest discover -s test/python -p 'test_train_hide_seek_multipolicy.py' -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'split_by_policy'`.

- [ ] **Step 3: Implement** — add to `scripts/train_hide_seek_multipolicy.py`:

```python
def split_by_policy(batched, index_map: Dict[str, List[int]]):
    """Slice a (n_agents, ...) array into {name: array[indices]} per policy. Lazy numpy."""
    import numpy as np

    arr = np.asarray(batched)
    return {name: arr[idx] for name, idx in index_map.items()}


def stitch_actions(per_policy_actions, index_map: Dict[str, List[int]], n_agents: int):
    """Inverse of split_by_policy for actions: scatter each policy's (n_p, action_dim) actions back
    into a single (n_agents, action_dim) int64 array in agent order. Lazy numpy."""
    import numpy as np

    first = next(iter(per_policy_actions.values()))
    action_dim = int(np.asarray(first).shape[1])
    out = np.zeros((n_agents, action_dim), dtype=np.int64)
    for name, idx in index_map.items():
        out[idx] = np.asarray(per_policy_actions[name])
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv-train/bin/python -m unittest discover -s test/python -p 'test_train_hide_seek_multipolicy.py' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_hide_seek_multipolicy.py test/python/test_train_hide_seek_multipolicy.py
git commit -m "feat: split_by_policy/stitch_actions round-trip for multi-policy routing (#26)"
```

### Task 8: `parse_args` → `MultiPolicyConfig`

**Files:**
- Modify: `scripts/train_hide_seek_multipolicy.py`
- Test: `test/python/test_train_hide_seek_multipolicy.py`

- [ ] **Step 1: Write the failing test** — append:

```python
class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        cfg = mp.parse_args([])
        self.assertEqual(cfg.timesteps, 800_000)
        self.assertEqual(cfg.onnx_export_dir, "models")
        self.assertEqual(cfg.policy_names, ("seeker", "hider"))

    def test_overrides(self):
        cfg = mp.parse_args(["--timesteps", "1234", "--speedup", "4"])
        self.assertEqual(cfg.timesteps, 1234)
        self.assertEqual(cfg.speedup, 4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv-train/bin/python -m unittest discover -s test/python -p 'test_train_hide_seek_multipolicy.py' -v`
Expected: FAIL — no `parse_args`.

- [ ] **Step 3: Implement** — add to `scripts/train_hide_seek_multipolicy.py`:

```python
class MultiPolicyConfig(NamedTuple):
    timesteps: int
    speedup: int
    action_repeat: int
    seed: int
    num_steps: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    update_epochs: int
    num_minibatches: int
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    onnx_export_dir: str
    policy_names: tuple  # expected names, for a fail-fast sanity check against the wire


def parse_args(argv: Sequence[str] | None = None) -> "MultiPolicyConfig":
    p = argparse.ArgumentParser(allow_abbrev=False, description="Multi-policy PPO for hide & seek.")
    p.add_argument("--timesteps", type=int, default=800_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--onnx_export_dir", type=str, default="models")
    a = p.parse_args(argv)
    return MultiPolicyConfig(
        timesteps=a.timesteps, speedup=a.speedup, action_repeat=a.action_repeat, seed=a.seed,
        num_steps=a.num_steps, learning_rate=a.learning_rate, gamma=a.gamma,
        gae_lambda=a.gae_lambda, update_epochs=a.update_epochs, num_minibatches=a.num_minibatches,
        clip_coef=a.clip_coef, ent_coef=a.ent_coef, vf_coef=a.vf_coef,
        max_grad_norm=a.max_grad_norm, onnx_export_dir=a.onnx_export_dir,
        policy_names=("seeker", "hider"),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv-train/bin/python -m unittest discover -s test/python -p 'test_train_hide_seek_multipolicy.py' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_hide_seek_multipolicy.py test/python/test_train_hide_seek_multipolicy.py
git commit -m "feat: MultiPolicyConfig + parse_args for multi-policy trainer (#26)"
```

---

## Phase 5: Trainer main loop + orchestrator

### Task 9: `main()` — per-policy PPO loop + dual ONNX export

**Files:**
- Modify: `scripts/train_hide_seek_multipolicy.py`

> This is the integration glue (exercised by the real training run in Task 11, not a unit test — it needs a live Godot env). It mirrors `scripts/train_cleanrl.py:main()` closely; the difference is per-policy state in dicts + the routing helpers. Reuse the pure helpers and the ONNX exporter from `train_cleanrl`.

- [ ] **Step 1: Implement `main()` and `_policy_names_from_env`** — add to `scripts/train_hide_seek_multipolicy.py`:

```python
def _policy_names_from_env(env) -> list:
    """Per-agent policy names from the underlying GodotEnv. Attribute path confirmed in Task 0;
    adjust here if it differs in the installed godot_rl."""
    for obj in (env, getattr(env, "_env", None), *(getattr(env, "envs", []) or [])):
        names = getattr(obj, "agent_policy_names", None)
        if names:
            return list(names)
    raise RuntimeError("could not read agent_policy_names from CleanRLGodotEnv (see Task 0)")


def main(argv: Sequence[str] | None = None) -> None:
    import pathlib

    import numpy as np
    import torch
    import torch.nn as nn

    from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv

    import train_cleanrl as tc  # reuse compute_gae, num_updates, layer_init, _build_agent, etc.

    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cpu")

    env = CleanRLGodotEnv(
        env_path=None, show_window=False, seed=cfg.seed, n_parallel=1,
        speedup=cfg.speedup, action_repeat=cfg.action_repeat,
    )
    n_agents = env.num_envs
    observation_dim = tc.obs_dim(env.single_observation_space)
    total_logits, nvec = tc.act_layout(env.single_action_space)

    names = _policy_names_from_env(env)
    index_map = policy_index_map(names)
    print(f"obs_dim={observation_dim} logits={total_logits} nvec={nvec} "
          f"n_agents={n_agents} policies={ {k: len(v) for k, v in index_map.items()} }")
    for expected in cfg.policy_names:
        if expected not in index_map:
            raise RuntimeError(f"expected policy '{expected}' not on the wire (got {list(index_map)})")

    # Per-policy learners + rollout storage (one set per distinct policy name).
    agents, opts, bufs = {}, {}, {}
    num_steps = cfg.num_steps
    for name, idx in index_map.items():
        np_ = len(idx)
        ag = tc._build_agent(observation_dim, total_logits).to(device)
        agents[name] = ag
        opts[name] = torch.optim.Adam(ag.parameters(), lr=cfg.learning_rate, eps=1e-5)
        bufs[name] = dict(
            obs=torch.zeros((num_steps, np_, observation_dim), device=device),
            actions=torch.zeros((num_steps, np_, len(nvec)), dtype=torch.long, device=device),
            logprobs=torch.zeros((num_steps, np_), device=device),
            rewards=torch.zeros((num_steps, np_), device=device),
            dones=torch.zeros((num_steps, np_), device=device),
            values=torch.zeros((num_steps, np_), device=device),
        )

    updates = tc.num_updates(cfg.timesteps, num_steps, n_agents)
    print(f"running {updates} updates over {n_agents} agents")

    next_obs_np, _ = env.reset(cfg.seed)
    next_obs = torch.tensor(np.asarray(next_obs_np, dtype=np.float32), device=device)
    next_done = torch.zeros(n_agents, device=device)

    def split_t(t):  # split a (n_agents, ...) torch tensor per policy
        return {name: t[idx] for name, idx in index_map.items()}

    for update in range(updates):
        for step in range(num_steps):
            no_split = split_t(next_obs)
            nd_split = split_t(next_done)
            full_action = np.zeros((n_agents, len(nvec)), dtype=np.int64)
            per_policy_action = {}
            for name, idx in index_map.items():
                ag = agents[name]
                b = bufs[name]
                ob = no_split[name]
                b["obs"][step] = ob
                b["dones"][step] = nd_split[name]
                with torch.no_grad():
                    logits = ag.logits(ob)
                    value = ag.value(ob)
                dists = tc._split_categoricals(logits, nvec)
                sampled = [d.sample() for d in dists]
                action = torch.stack(sampled, dim=1)
                b["actions"][step] = action
                b["logprobs"][step] = sum(d.log_prob(a) for d, a in zip(dists, sampled))
                b["values"][step] = value
                per_policy_action[name] = action.cpu().numpy().astype(np.int64)
            full_action = stitch_actions(per_policy_action, index_map, n_agents)

            next_obs_np, reward, terminations, truncations, _ = env.step(full_action)
            done = np.logical_or(np.asarray(terminations), np.asarray(truncations)).astype(np.float32)
            reward_t = torch.tensor(np.asarray(reward, dtype=np.float32), device=device)
            for name, idx in index_map.items():
                bufs[name]["rewards"][step] = reward_t[idx]
            next_obs = torch.tensor(np.asarray(next_obs_np, dtype=np.float32), device=device)
            next_done = torch.tensor(done, device=device)

        # Per-policy PPO update (mirrors train_cleanrl, independently per learner).
        for name, idx in index_map.items():
            ag, opt, b = agents[name], opts[name], bufs[name]
            np_ = len(idx)
            with torch.no_grad():
                next_value = ag.value(next_obs[idx])
            adv_np, ret_np = tc.compute_gae(
                b["rewards"].cpu().numpy(), b["values"].cpu().numpy(), b["dones"].cpu().numpy(),
                next_value.cpu().numpy(), next_done[idx].cpu().numpy(), cfg.gamma, cfg.gae_lambda)
            advantages = torch.tensor(adv_np, device=device)
            returns = torch.tensor(ret_np, device=device)

            b_obs = b["obs"].reshape(-1, observation_dim)
            b_actions = b["actions"].reshape(-1, len(nvec))
            b_logprobs = b["logprobs"].reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            batch_size = num_steps * np_
            minibatch_size = max(1, batch_size // cfg.num_minibatches)
            b_inds = np.arange(batch_size)
            for _ in range(cfg.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    mb = b_inds[start:start + minibatch_size]
                    logits = ag.logits(b_obs[mb])
                    dists = tc._split_categoricals(logits, nvec)
                    mb_actions = b_actions[mb]
                    new_logprob = sum(d.log_prob(mb_actions[:, i]) for i, d in enumerate(dists))
                    entropy = sum(d.entropy() for d in dists)
                    new_value = ag.value(b_obs[mb])
                    logratio = new_logprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    mb_adv = b_advantages[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                    v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()
                    loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ag.parameters(), cfg.max_grad_norm)
                    opt.step()

        msg = " ".join(f"{name}_rew={float(bufs[name]['rewards'].mean()):.3f}" for name in index_map)
        print(f"update {update + 1}/{updates} {msg}")

    # Export each policy's actor to ONNX for the ncnn pipeline.
    outdir = pathlib.Path(cfg.onnx_export_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in index_map:
        onnx_path = outdir / f"hide_seek_{name}.onnx"
        tc.export_actor_as_onnx(agents[name], observation_dim, str(onnx_path))
        torch.save(agents[name].state_dict(), outdir / f"hide_seek_{name}.pt")
        print("Exported ONNX to:", onnx_path)

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the module imports cleanly** (no env needed — just import + pure path)

Run: `.venv-train/bin/python -c "import sys; sys.path.insert(0,'scripts'); import train_hide_seek_multipolicy as m; print(m.parse_args([]).timesteps)"`
Expected: prints `800000` (confirms the module imports with torch/godot_rl lazy).

- [ ] **Step 3: Commit**

```bash
git add scripts/train_hide_seek_multipolicy.py
git commit -m "feat: multi-policy PPO main loop + dual ONNX export (#26)"
```

### Task 10: Orchestrator script

**Files:**
- Create: `scripts/train_hide_seek_multipolicy.sh`

- [ ] **Step 1: Create the script** (mirrors `train_hide_seek.sh`; passes `--multi-policy`, defaults to the parallel scene):

```bash
#!/usr/bin/env bash
# Orchestrates MULTI-POLICY self-play training (two distinct policies: seeker + hider):
#   1. start the Python multi-policy trainer (opens server on 11008, waits)
#   2. launch the headless Godot scene WITH --multi-policy (so the agents emit distinct policy_names)
#   3. wait for the trainer, then ensure Godot is gone
# SCENE override selects single vs parallel; defaults to the parallel (fast) scene.
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-800000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="${SCENE:-res://examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn}"

echo "Starting multi-policy trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_hide_seek_multipolicy.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" &
TRAINER_PID=$!

sleep 5

echo "Launching headless Godot scene ($SCENE) with --multi-policy..."
"$GODOT" --headless --path . "$SCENE" --multi-policy "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x scripts/train_hide_seek_multipolicy.sh
git add scripts/train_hide_seek_multipolicy.sh
git commit -m "feat: orchestrator for multi-policy hide&seek training (#26)"
```

---

## Phase 6: Training run + conversion

### Task 11: Run training and convert both models to ncnn

**Files:**
- Create (artifacts): `examples/hide_and_seek/models/hide_seek_seeker.ncnn.{param,bin}`, `hide_seek_hider.ncnn.{param,bin}`

> Compute-heavy + self-play; wrap in `caffeinate -is` on macOS (sleep gotcha). Use the parallel scene.

- [ ] **Step 1: Train** (start short to sanity-check convergence, then full)

Run: `caffeinate -is ./scripts/train_hide_seek_multipolicy.sh` (override `TIMESTEPS=` as needed)
Expected: per-update lines printing `seeker_rew=… hider_rew=…`; on finish, `models/hide_seek_seeker.onnx` and `models/hide_seek_hider.onnx` written. Watch the role-signed rewards diverge/co-adapt; pick a healthy stopping point (re-run with a higher/lower `TIMESTEPS`, manual best-checkpoint selection — rover precedent).

- [ ] **Step 2: Convert both ONNX → ncnn (with parity check)**

Run:
```bash
.venv-train/bin/python scripts/export_to_ncnn.py models/hide_seek_seeker.onnx --outdir examples/hide_and_seek/models
.venv-train/bin/python scripts/export_to_ncnn.py models/hide_seek_hider.onnx --outdir examples/hide_and_seek/models
```
Expected: each prints `PARITY OK: N/N` (ncnn↔ONNX, atol 1e-2) and writes `hide_seek_{seeker,hider}.ncnn.{param,bin}`.

- [ ] **Step 3: Commit the trained artifacts**

```bash
git add examples/hide_and_seek/models/hide_seek_seeker.ncnn.param examples/hide_and_seek/models/hide_seek_seeker.ncnn.bin \
        examples/hide_and_seek/models/hide_seek_hider.ncnn.param examples/hide_and_seek/models/hide_seek_hider.ncnn.bin
git commit -m "feat: trained seeker+hider ncnn models for multi-policy hide&seek (#26)"
```

---

## Phase 7: Regressions (golden-inference + behavioral)

### Task 12: Golden-inference regression (two models)

**Files:**
- Create: `test/unit/test_hide_seek_multipolicy_golden_inference.gd`

- [ ] **Step 1: Capture golden argmaxes from the real ncnn deploy path** — run a throwaway snippet to print `run_discrete_action` for 5 fixed 15-float obs against each model (model uses `in0`/`out0` blob names like the other golden tests). Use this scratch script and record its output:

```gdscript
# scratch: godot --headless --path . --script res://scratch_capture.gd  (delete after)
extends SceneTree
func _initialize() -> void:
	var obs_set := [
		[0.5,-0.1,0.7,0.4,-0.8,0.1,0.2,0.3,0.0,0.1,0.5,0.6,0.2,0.9,1.0],
		[0.9,0.5,0.5,-0.7,-0.1,0.3,0.1,0.2,0.4,0.0,0.7,0.1,0.3,0.2,1.0],
		[-0.2,0.8,0.2,0.6,-0.1,0.5,0.6,0.1,0.2,0.3,0.4,0.5,0.6,0.1,0.0],
		[-0.5,0.1,-0.8,0.6,0.2,0.1,0.0,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.0],
		[0.5,-0.2,0.9,0.7,0.5,0.2,0.1,0.3,0.4,0.5,0.6,0.1,0.2,0.3,1.0],
	]
	for tag in [["seeker", "res://examples/hide_and_seek/models/hide_seek_seeker.ncnn"], ["hider", "res://examples/hide_and_seek/models/hide_seek_hider.ncnn"]]:
		var r := NcnnRunner.new(); r.input_blob_name = "in0"; r.output_blob_name = "out0"
		r.load_model(ProjectSettings.globalize_path(tag[1] + ".param"), ProjectSettings.globalize_path(tag[1] + ".bin"))
		var line := "%s:" % tag[0]
		for o in obs_set:
			line += " %d" % r.run_discrete_action(PackedFloat32Array(o))
		print(line)
		r.free()
	quit()
```

Note: seeker obs use role flag 1.0 (index 14), hider obs use 0.0 — the snippet above varies it; for the committed golden, use role-appropriate flags per model. Record the printed argmaxes.

- [ ] **Step 2: Write the golden test** (modeled on `test_chase_cleanrl_golden_inference.gd`; fill `EXPECTED_SEEKER`/`EXPECTED_HIDER` with the captured values):

```gdscript
extends SceneTree
# Golden inference regression for the two multi-policy hide & seek ncnn models
# (scripts/train_hide_seek_multipolicy.py -> export_to_ncnn.py). Loads each model via NcnnRunner and
# asserts run_discrete_action() returns the captured argmax for 5 fixed observations. ncnn<->ONNX
# parity (atol 1e-2) was verified at conversion time by export_to_ncnn.py. If this fails after a
# retrain/model swap, recapture goldens (see the plan's scratch snippet) and update them here.

const Harness = preload("res://test/harness.gd")
# obs is 15 floats; index 14 is the role flag (seeker=1.0, hider=0.0).
const OBS: Array = [
	[0.5,-0.1,0.7,0.4,-0.8,0.1,0.2,0.3,0.0,0.1,0.5,0.6,0.2,0.9, 1.0],
	[0.9,0.5,0.5,-0.7,-0.1,0.3,0.1,0.2,0.4,0.0,0.7,0.1,0.3,0.2, 1.0],
	[-0.2,0.8,0.2,0.6,-0.1,0.5,0.6,0.1,0.2,0.3,0.4,0.5,0.6,0.1, 1.0],
	[-0.5,0.1,-0.8,0.6,0.2,0.1,0.0,0.2,0.3,0.4,0.1,0.2,0.3,0.4, 1.0],
	[0.5,-0.2,0.9,0.7,0.5,0.2,0.1,0.3,0.4,0.5,0.6,0.1,0.2,0.3, 1.0],
]
const EXPECTED_SEEKER: Array = [0, 0, 0, 0, 0]  # TODO: replace with captured argmaxes (role flag 1.0)
const EXPECTED_HIDER: Array  = [0, 0, 0, 0, 0]  # TODO: replace with captured argmaxes (role flag 0.0)

func _check(h, tag: String, base: String, expected: Array, role_flag: float) -> void:
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(base + ".param"),
		ProjectSettings.globalize_path(base + ".bin"))
	h.assert_true(ok, "%s model loads" % tag)
	if ok:
		for i in range(OBS.size()):
			var o: Array = OBS[i].duplicate()
			o[14] = role_flag
			var got := runner.run_discrete_action(PackedFloat32Array(o))
			h.assert_eq(got, int(expected[i]), "%s golden argmax #%d" % [tag, i])
	runner.free()

func _initialize() -> void:
	var h := Harness.new()
	_check(h, "seeker", "res://examples/hide_and_seek/models/hide_seek_seeker.ncnn", EXPECTED_SEEKER, 1.0)
	_check(h, "hider", "res://examples/hide_and_seek/models/hide_seek_hider.ncnn", EXPECTED_HIDER, 0.0)
	h.finish(self)
```

- [ ] **Step 3: Fill in the captured goldens, run, verify pass**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_multipolicy_golden_inference.gd`
Expected: PASS (all argmax assertions). Delete the scratch capture script.

- [ ] **Step 4: Commit**

```bash
git add test/unit/test_hide_seek_multipolicy_golden_inference.gd
git commit -m "test: golden-inference regression for multi-policy hide&seek models (#26)"
```

### Task 13: Behavioral checker + eval scene

**Files:**
- Create: `test/integration/trained_hide_seek_multipolicy_checker.gd`
- Create: `examples/hide_and_seek/hide_and_seek_multipolicy_eval.tscn`
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Write the behavioral checker** (counts catches over a fixed run; threshold generous — self-play floor, not a tight bar). Modeled on `trained_chase_checker.gd`; the game exposes `was_caught()`/`is_terminal()`:

```gdscript
extends Node
# Headless behavioral floor: runs BOTH trained policies (seeker via its model, hider via its) under
# ncnn inference and asserts the seeker catches the hider at least `min_catches` times within
# `frames_to_run`. Self-play co-adapts, so this is a generous sanity floor; the golden-inference test
# is the precise regression.

@export var game_path: NodePath
@export var seeker_path: NodePath
@export var hider_path: NodePath
@export var frames_to_run := 3000
@export var min_catches := 2

var _game
var _frames := 0
var _catches := 0
var _was_caught_last := false

func _ready() -> void:
	_game = get_node_or_null(game_path)
	var seeker = get_node_or_null(seeker_path)
	var hider = get_node_or_null(hider_path)
	if _game == null or seeker == null or hider == null:
		_fail("could not resolve game/seeker/hider nodes")
		return
	for a in [seeker, hider]:
		if a._ncnn_runner == null or not a._ncnn_runner.is_model_loaded():
			_fail("a trained ncnn model is not loaded")
			return

func _physics_process(_delta: float) -> void:
	if _game == null:
		return
	# Rising-edge count of catches (was_caught() is true the frame a catch lands).
	var caught: bool = _game.was_caught()
	if caught and not _was_caught_last:
		_catches += 1
	_was_caught_last = caught
	_frames += 1
	if _frames >= frames_to_run:
		if _catches >= min_catches:
			print("MULTI-POLICY HIDE&SEEK PASSED (%d catches in %d frames)" % [_catches, _frames])
			get_tree().quit(0)
		else:
			_fail("only %d catches in %d frames (need %d)" % [_catches, _frames, min_catches])

func _fail(reason: String) -> void:
	printerr("MULTI-POLICY HIDE&SEEK FAILED: %s" % reason)
	get_tree().quit(1)
```

- [ ] **Step 2: Create the eval scene** — instance the world, set the Sync + both agents to `NCNN_INFERENCE` with distinct model paths, add the checker. (Single world, not tiled, so per-node overrides on the instanced world are fine.):

```
[gd_scene load_steps=4 format=3]

[ext_resource type="PackedScene" path="res://examples/hide_and_seek/hide_seek_world.tscn" id="1"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="2"]
[ext_resource type="Script" path="res://test/integration/trained_hide_seek_multipolicy_checker.gd" id="3"]

[node name="HideSeekMultiPolicyEval" type="Node2D"]

[node name="HideSeekWorld" parent="." instance=ExtResource("1")]

[node name="Seeker" parent="HideSeekWorld" index="2"]
control_mode = 3
model_param_path = "res://examples/hide_and_seek/models/hide_seek_seeker.ncnn.param"
model_bin_path = "res://examples/hide_and_seek/models/hide_seek_seeker.ncnn.bin"

[node name="Hider" parent="HideSeekWorld" index="3"]
control_mode = 3
model_param_path = "res://examples/hide_and_seek/models/hide_seek_hider.ncnn.param"
model_bin_path = "res://examples/hide_and_seek/models/hide_seek_hider.ncnn.bin"

[node name="Sync" type="Node" parent="."]
script = ExtResource("2")
control_mode = 2

[node name="Checker" type="Node" parent="."]
script = ExtResource("3")
game_path = NodePath("../HideSeekWorld")
seeker_path = NodePath("../HideSeekWorld/Seeker")
hider_path = NodePath("../HideSeekWorld/Hider")
```

Note: confirm the `index=` values for the overridden `Seeker`/`Hider` child nodes match their order in `hide_seek_world.tscn` (Seeker is the 3rd child, Hider the 4th — indices 2 and 3). Adjust if the world scene changes.

- [ ] **Step 3: Run the eval scene, verify pass**

Run: `godot --headless --path . res://examples/hide_and_seek/hide_and_seek_multipolicy_eval.tscn`
Expected: `MULTI-POLICY HIDE&SEEK PASSED (…)`. If catches are below the floor, lower `min_catches` toward a value the trained pair reliably clears, or train longer (Task 11). Keep the floor conservative.

- [ ] **Step 4: Wire into `test/run_tests.sh`** — after the multi-policy wire smoke block (Task 5), add:

```bash
echo "== Trained multi-policy hide&seek behavioral check (headless) =="
"$GODOT" --headless --path . res://examples/hide_and_seek/hide_and_seek_multipolicy_eval.tscn
```

- [ ] **Step 5: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.` (golden + smoke + behavioral all green from a clean cache).

- [ ] **Step 6: Commit**

```bash
git add test/integration/trained_hide_seek_multipolicy_checker.gd examples/hide_and_seek/hide_and_seek_multipolicy_eval.tscn test/run_tests.sh
git commit -m "test: behavioral floor + eval scene for multi-policy hide&seek (#26)"
```

---

## Phase 8: Docs & closeout

### Task 14: Update docs and close the issue

**Files:**
- Modify: `examples/hide_and_seek/README.md`, `CLAUDE.md`, `docs/BACKLOG.md`

- [ ] **Step 1: README** — add a "Multi-policy (two distinct policies)" section to `examples/hide_and_seek/README.md` contrasting it with the shared-policy run, and document the run command + the `--multi-policy` flag:

```markdown
## Multi-policy variant (two distinct policies)

The same arena, but the seeker and hider learn **two separate networks** instead of one shared
policy. Run:

\`\`\`bash
caffeinate -is ./scripts/train_hide_seek_multipolicy.sh   # parallel scene by default
\`\`\`

The agents get distinct `policy_name`s (`seeker`/`hider`) only when Godot is launched with
`--multi-policy` (a process-global gate read by `HideSeekAgent`; the shared-policy run above keeps
`shared_policy`). The trainer (`scripts/train_hide_seek_multipolicy.py`) routes each agent to its
policy via the `agent_policy_names` wire field, then exports two ncnn models
(`models/hide_seek_seeker.ncnn.*`, `hide_seek_hider.ncnn.*`). Deploy both in
`hide_and_seek_multipolicy_eval.tscn`. A cleaner identity mechanism is tracked in issue #73.
```

- [ ] **Step 2: CLAUDE.md** — add a key-command bullet under the training commands:

```markdown
- **Train (hide & seek, two distinct policies):** `./scripts/train_hide_seek_multipolicy.sh` — custom
  multi-policy PPO; seeker + hider learn separate networks (distinct `policy_name`s via the
  `--multi-policy` cmdline gate), exported to two ncnn models. `SCENE=` for single vs parallel.
```

Also update the BACKLOG status line for item 45 in CLAUDE.md's "Done" list if present.

- [ ] **Step 3: BACKLOG** — flip item 45 (`docs/BACKLOG.md`) to ✅ done with a one-line summary + date, matching the format of other shipped items.

- [ ] **Step 4: Verify docs reference real paths**

Run: `ls scripts/train_hide_seek_multipolicy.sh examples/hide_and_seek/hide_and_seek_multipolicy_eval.tscn examples/hide_and_seek/models/hide_seek_seeker.ncnn.param`
Expected: all exist (no stale paths in docs).

- [ ] **Step 5: Commit**

```bash
git add examples/hide_and_seek/README.md CLAUDE.md docs/BACKLOG.md
git commit -m "docs: multi-policy hide&seek example; close backlog item 45 (#26)"
```

- [ ] **Step 6: Open the PR**

```bash
git push -u origin feat/multi-policy-hide-seek
gh pr create --title "Multi-policy trained example (Hide & Seek, two distinct policies)" \
  --body "Closes #26. Custom multi-policy PPO trains seeker+hider as two distinct policies routed by the agent_policy_names wire field; two ncnn models + golden-inference + behavioral regressions. Identity gated by --multi-policy (follow-up: #73)."
```

---

## Notes for the implementer

- **Run from the repo root.** Tests/scripts assume CWD = repo root.
- **Rebuild the extension on a fresh clone** before any `NcnnRunner` test (`bin/` is gitignored) — `scons platform=macos arch=arm64 target=template_debug`.
- **`class_name` is unreliable headless** — the scenes/tests here use path-based `extends`/preloads, keep it that way.
- **GDScript uses TABS.**
- **The order of Phases 6→7 matters:** the golden + behavioral tests need the trained models to exist. Don't wire them into `run_tests.sh` (Task 13 step 4) until the models are committed (Task 11), or the suite goes red.
- **DRY:** the trainer reuses `train_cleanrl`'s `compute_gae`, `num_updates`, `layer_init`, `_build_agent`, `_split_categoricals`, `obs_dim`, `act_layout`, `export_actor_as_onnx`. Do not re-implement them.

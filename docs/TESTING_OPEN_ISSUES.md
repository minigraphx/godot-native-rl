# Testing Guide: Worked-On But Still Open Issues

> This guide answers: *I have a spec + plan for an issue — what tests do I need, and how do I verify the implementation is complete?*

---

## 1. What "worked-on but open" means

An issue enters this state when it has:

| Artifact | Location |
|----------|----------|
| Design spec | `docs/superpowers/specs/<date>-<issue-slug>-design.md` |
| Implementation plan | `docs/superpowers/plans/<date>-<issue-slug>.md` |
| Feature branch | `feat/<slug>` or `claude/*` |
| **No green tests yet** | The plan's checkboxes are unchecked |

The implementation plan lists exact files to create/modify. Every file under `test/` listed in that plan is a **mandatory test** that must pass before the issue can be closed.

---

## 2. Test layer map

| Layer | Location | Runner | When to write |
|-------|----------|--------|--------------|
| **Unit (GDScript)** | `test/unit/test_*.gd` | `godot --headless --path . --script res://test/unit/test_<foo>.gd` | Always — one test file per new file/class |
| **Unit (Python)** | `test/python/test_*.py` | `.venv-train/bin/python -m unittest test.python.test_<foo>` | Every new Python script helper |
| **Integration (smoke)** | `test/integration/*_smoke_scene.tscn` + `*_checker.gd` | `godot --headless --path . res://test/integration/<foo>_smoke_scene.tscn` | New C++ extension surface, new addon scene |
| **Integration (protocol)** | `test/integration/run_*_test.py` | `.venv/bin/python test/integration/run_<foo>_test.py` | New wire-protocol behavior |
| **Golden inference** | `test/unit/test_*_golden_inference.gd` | same as unit | New ncnn model path (real fixture) |
| **Full suite** | `test/run_tests.sh` | `./test/run_tests.sh` | Before every merge |

> **Worktree note:** `run_tests.sh` requires the compiled `.so`/`.dylib` in `bin/` and both venvs — these are gitignored. Run the full suite only in the main working tree, not inside a `.claude/worktrees/*` path.

---

## 3. TDD workflow per issue

Follow the mandatory red-green-refactor cycle from `testing.md`:

```
1. Read the implementation plan   →  docs/superpowers/plans/<date>-<issue>.md
2. For each planned test file:
      a. Create the test file       (RED — all assertions fail)
      b. Run it, confirm it fails
      c. Write minimal production code
      d. Run it, confirm it passes   (GREEN)
      e. Refactor
3. Run ./test/run_tests.sh         (no regressions)
4. Verify coverage gate: every logical branch in the new code has a test
```

**Shortcut for single test file during dev** (avoids running the full 3-minute suite):
```bash
godot --headless --path . --script res://test/unit/test_<foo>.gd
```

---

## 4. Per-issue test requirements (currently-open issues)

> Only issues that are **open** and have (or are about to get) a spec + plan live here. When an
> issue closes, delete its subsection — don't leave it as a tombstone. Items already shipped are
> tracked in `CLAUDE.md`'s Done list and `docs/BACKLOG.md`.
>
> Recently graduated off this list (now closed): #33 Recurrent/LSTM deploy (item 22), #18 Running
> Normalization Sensor (item 47), #17 Observation History Buffer (item 46), #13 Expert-demo
> recording (item 10).

### #45 — Algorithm-agnostic training/deploy contract `priority:2`

Plan: none yet — write the spec first (`docs/superpowers/specs/<date>-algorithm-agnostic-contract-design.md`).

`test/unit/test_algorithm_agnostic_decode.gd` already exists — start by auditing its current
coverage before adding new paths.

Expected tests (once specced):
- Extend `test/unit/test_algorithm_agnostic_decode.gd` with non-PPO decode paths
- Add DQN / SAC golden paths to that file
- Smoke test with a synthetic DQN fixture (argmax over Q-values)

---

## 5. Running individual vs full test suites

### Single GDScript test
```bash
godot --headless --path . --script res://test/unit/test_<foo>.gd
```
Pass = exits 0, last line is `X assertions, 0 failed`.

### Single Python test module
```bash
.venv-train/bin/python -m unittest test.python.test_<foo> -v
```

### Full suite (before merge only, in main working tree)
```bash
./test/run_tests.sh
```
Gate: exits 0 and final line is `All tests passed.`

> **Never pipe to `tail` or `grep`** — if a test hangs (e.g. missing script-class cache), the pipe will also hang silently. Let the suite stream naturally.

---

## 6. Pre-merge checklist for closing an issue

Before creating the PR that closes an issue:

- [ ] Every test file listed in the implementation plan exists
- [ ] `./test/run_tests.sh` exits 0 (`All tests passed.`)
- [ ] `git diff main...HEAD` touches no file that isn't covered by at least one test
- [ ] C++ changes → extension rebuilt: `scons platform=macos arch=arm64 target=template_debug`
- [ ] Docs updated: `README.md`, `CLAUDE.md` current-state bullet, `docs/BACKLOG.md` checkbox, `docs/godot-rl-gap-analysis-2026-06-02.md`
- [ ] PR description contains `Closes #NN`

---

## 7. Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Headless Godot hangs ~0% CPU | Missing/stale `global_script_class_cache.cfg` | `run_tests.sh` regenerates it automatically; manually: `godot --headless --editor --quit` then `git clean -f -- '*.gd.uid'` |
| `Could not find base class` | `extends SomeClassName` — bare class_name unreliable headless | Use path-based extends: `extends "res://addons/godot_native_rl/.../foo.gd"` |
| ncnn runner returns `-1` | Blob names not set before inference | Set `input_blob_name = "in0"` / `output_blob_name = "out0"` on the runner |
| `Array[int]` assignment hangs | Assigning untyped `[2,3]` literal to typed @export | Type the local first: `var x: Array[int] = [2,3]` |
| Python `NotImplementedError` on `env.seed()` | Passing `seed=` to `PPO()` | Seed via env constructor only |

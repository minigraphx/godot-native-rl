# Godot Native RL — Project Memory

## What this is

A GDExtension-based reinforcement-learning framework for Godot 4.6+ that uses Tencent's **ncnn**
for native inference (statically linked C++, **no C#/.NET, no external runtime**). It speaks the
`godot_rl_agents` wire protocol for training, so you train with the stock `godot-rl` Python package
and deploy with native ncnn — on mobile, web, console, desktop, and edge.

**Positioning (north star):** focused superiority first (be clearly better at deployment), then
godot_rl feature parity, then Unity ML-Agents parity (long-term stretch). Strategy: start as a
complement to godot_rl, grow toward full replacement.

## Current state (working, on `main`)

- Full **train → convert → deploy loop** works end-to-end and is in CI-style headless tests.
- Components: `sync.gd` (`NcnnSync`, the bridge), `ncnn_ai_controller_2d.gd` (`NcnnAIController2D`,
  the agent contract), `src/ncnn_runner.{h,cpp}` (`NcnnRunner` C++ GDExtension).
- Example: `examples/chase_the_target/` — a 2D agent that learns to chase a relocating target,
  ships with a pre-trained ncnn model.
- Wire protocol is **fully godot_rl v0.8.2-compatible** (proven by real SB3 PPO training).

## Key commands

- **Build the extension:** `scons platform=macos arch=arm64 target=template_debug` (see README for other platforms). `godot` binary: `/opt/homebrew/bin/godot` (4.6.2).
- **Run all tests:** `./test/run_tests.sh` — headless GDScript unit tests + Python protocol test +
  inference smoke + trained-chase check + golden inference regression. Must be green before merge.
- **Train:** `TIMESTEPS=120000 ./scripts/train_chase.sh` (starts SB3 trainer, launches headless
  Godot training scene which connects on port 11008). ~34 min at 120k steps.
- **Convert ONNX→ncnn:** `cd models && ../.venv/bin/pnnx model.onnx 'inputshape=[1,5],[1]'`
- **Verify conversion:** `.venv-train/bin/python scripts/verify_ncnn_parity.py <onnx> <param> <bin> in0 out0`

## Operational gotchas (learned the hard way)

- **Two venvs:** `.venv` (Python 3.14) has `pnnx`+torch for conversion. `.venv-train` (Python
  **3.13** — torch wheels don't exist for 3.14) has `godot-rl onnxruntime ncnn onnxscript`. Keep
  them separate. Both gitignored.
- **`onnxscript` is required** for ONNX export (torch 2.12 dynamo exporter needs it) — not pulled
  in by godot-rl automatically.
- **Do NOT pass `seed=` to `PPO()`** — godot-rl's env wrapper raises `NotImplementedError` on
  `env.seed()`. Seed via the env constructor only.
- **pnnx `inputshape` must be quoted** (`'inputshape=[1,5],[1]'`) or zsh globs the brackets. The
  second `[1]` is godot-rl's vestigial `state_ins` input — pnnx prunes it → clean `in0`/`out0`.
- **Parity tolerance is `atol=1e-2`** — torch dynamo exporter vs ncnn InnerProduct differ by
  ~1e-3 to 5e-3 in float32; argmax is stable.
- **The bridge sets `done` at `reset_after`** (godot_rl convention) so episodes terminate and
  `ep_rew_mean` appears. (A future chip splits this into `terminated`/`truncated`.)

## Conventions

- GDScript uses **TAB** indentation. Dependency-free headless test harness at `test/harness.gd`
  (tests `extends SceneTree`, run via `godot --headless --path . --script res://test/...`).
- Use the **superpowers workflow**: brainstorm → spec (`docs/superpowers/specs/`) → plan
  (`docs/superpowers/plans/`) → TDD implement on a feature branch. Don't push to `main` directly.

## Roadmap & backlog

- **Strategy + gap analysis:** `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md`
  (four tracks: Sensors, Multi-Agent, Training Algorithms, DX/Distribution).
- **Novel addons + protocol findings:**
  `docs/superpowers/specs/2026-05-30-novel-addons-and-protocol-design.md` (10 addons in neither
  godot_rl nor Unity; 4 protocol upgrades incl. the `terminated`/`truncated` correctness fix).
- **Active work is queued as spawn-task chips** (15 of them): export_to_ncnn helper, ncnn_vs_onnx
  doc, sensors (relative-position, camera, navmesh), 3D controller+example, training backends
  (CleanRL/SampleFactory/SKRL), expert-demo recording, Signal→Reward+RewardBuilder (top DX
  priority), protocol v0.8 upgrades, INT8 quantization, async inference, LOD policy switching.

## The moat (why this beats godot_rl + Unity)

ncnn statically linked via C++ enables: web/WASM deployment (godot_rl's ONNX/.NET can't),
console deployment (no .NET cert issues), INT8 quantization game-side, async inference threads,
LOD policy switching, and Godot-native ideas (Signal→Reward, NavMesh sensor) — none replicable by
a Python-server framework or a managed-runtime one. Lead with these in all docs.

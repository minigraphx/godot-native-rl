# Operational Gotchas (learned the hard way)

> Long-form companion to `CLAUDE.md`. CLAUDE.md keeps only the few daily-biting items and links
> here for the rest. Contributor-facing.

- **Three venvs:** `.venv` (Python 3.14) has `pnnx`+torch for conversion. `.venv-train` (Python
  **3.13** — torch wheels don't exist for 3.14) is the training + verification venv: SB3 2.8.0 +
  gymnasium 1.2.2 + numpy≥2, plus the optional `ray[rllib]` add-on (the RLlib backend shares this
  venv since #126 — `requirements-rllib.txt` installed on top), and it runs `export_to_ncnn.py`.
  `godot-rl` is installed `--no-deps` here (its `stable-baselines3<=2.4.0` / `gymnasium<=1.0.0` caps
  conflict with that stack; its runtime use is gymnasium-1.2-compatible). `.venv-sf` (Python 3.13)
  has SampleFactory for the SF backend only — still isolated because SF pins `gymnasium<1.0`, which
  would downgrade the rest of the stack. Keep them separate. All gitignored.
- **`onnxscript` is required** for ONNX export (torch 2.12 dynamo exporter needs it) — not pulled
  in by godot-rl automatically.
- **`.venv-train` install: `pip install -c .github/ci-constraints.txt -r requirements-train.txt`,
  then `pip install --no-deps godot-rl==0.8.2`** (and, locally, `-r requirements-rllib.txt` for the
  ray add-on). `scripts/setup_training.sh` does all of this; CI (`ci.yml`) mirrors it minus the ray
  add-on. The `-c` file now just freezes `numpy==2.4.6` + `onnx==1.21.0` for reproducibility.
  **Why `godot-rl --no-deps`:** its declared `stable-baselines3<=2.4.0` / `gymnasium<=1.0.0` caps
  conflict with the SB3 2.8.0 / gymnasium 1.2.2 / ray stack, but its actual runtime use is
  gymnasium-1.2-compatible (proven end-to-end in #126). Pulling godot-rl *with* deps would drag the
  stack back to `numpy<2`.
  **Historical (pre-#126):** the venv used to install godot-rl with deps → `stable-baselines3<=2.4.0`
  → `numpy<2.0`, which fought the onnx/ml_dtypes chain and forced an `onnx==1.17.0` / `ml-dtypes==0.4.1`
  pin hack (newer onnx referenced `ml_dtypes.float4_e2m1fn` at import time, crashing every
  `torch.onnx.export` under numpy<2). Moving godot-rl to `--no-deps` let the whole stack go `numpy≥2`
  + modern onnx, retiring that hack. **Verify an install change by import + a real export, not
  `pip --dry-run`:** after installing, run `python -c "import onnx"` (must not raise) and
  `python scripts/make_synthetic_dqn.py` (self-checks ONNX↔eager parity).
- **Do NOT pass `seed=` to `PPO()`** — godot-rl's env wrapper raises `NotImplementedError` on
  `env.seed()`. Seed via the env constructor only.
- **pnnx `inputshape` must be quoted** (`'inputshape=[1,5],[1]'`) or zsh globs the brackets. The
  second `[1]` is godot-rl's vestigial `state_ins` input — pnnx prunes it → clean `in0`/`out0`.
- **Parity tolerance is `atol=1e-2`** — torch dynamo exporter vs ncnn InnerProduct differ by
  ~1e-3 to 5e-3 in float32; argmax is stable.
- **`ncnn::Mat::total()` over-counts (SIMD padding).** `total()` is `cstep * c`, and `cstep` is aligned up
  to a 16-byte boundary, so a `w=3` output reports 4 — copying `total()` floats yields a garbage trailing
  value (and can skew argmax). `NcnnRunner` copies the logical `w*h*d` elements per channel via
  `channel(q)` instead. This was fixed for item 21 (continuous deploy made it visible); **rebuild the
  extension** (`scons ... target=template_debug` and `template_release`) on a fresh clone — `addons/godot_native_rl/bin/` is
  gitignored.
- **The bridge sets `done` at `reset_after`** (godot_rl convention) so episodes terminate and
  `ep_rew_mean` appears. (A future chip splits this into `terminated`/`truncated`.)
- **`class_name` is unreliable headless:** the global class registry comes from
  `.godot/global_script_class_cache.cfg`, which is gitignored and is **not** rebuilt by
  `--headless`/`--script` runs (only an editor/import pass writes it). So `extends SomeClassName`
  fails (`Could not find base class`) on a fresh clone or after moving a `class_name` file. **Use
  path-based `extends "res://addons/godot_native_rl/.../foo.gd"`** for in-repo subclasses (the reward
  terms + example agents do this); reference scripts via `preload` consts, not bare `class_name`.
  **Fresh-clone trap:** with an empty/missing `.godot/` (or right after `rm
  global_script_class_cache.cfg`), a `class_name` base still can't resolve and the parse error fires
  inside a test's `_initialize()` *before* the harness reaches `quit()`, so headless Godot **hangs
  forever** (~0% CPU; looks like a slow test). A **stale** cache (after a branch switch that moved/removed
  a `class_name` file) is just as bad — the registry points the class at its old path, so the now-current
  file reports `hides a global script class` and dependent tests fail to compile. `run_tests.sh` self-heals
  both: it **regenerates the cache fresh on every run** (`rm` + `godot --headless --editor --quit`) before
  the tests. To do it manually: run that import pass once (or open the project in the editor).
- **Godot-generated `*.gd.uid` files are gitignored** (#181) — an editor/import pass scatters one per
  script, but they're ignored, so they no longer show as untracked noise or risk an accidental commit; no
  cleanup needed. (Three `.uid` are intentionally tracked; git keeps those regardless of the ignore.)
- **macOS/Apple Silicon: never let the machine sleep during training.** Sleep suspends the headless
  Godot client → the SB3 trainer blocks forever on the dead socket (`total_timesteps` freezes, process
  ~0% CPU, no `godot` process). The Godot client now self-terminates on `read_timeout_sec` (default
  60s), but the **trainer** side still blocks — kill and re-run (resumes from checkpoint). Prevent with
  `caffeinate -is ./scripts/train_rover.sh`. The local trainer+Godot can't survive the host sleeping —
  for unattended runs use an always-on machine/CI.
- **Capture golden-inference values with blob names set** (`runner.input_blob_name = "in0"`,
  `output_blob_name = "out0"`) — a bare `NcnnRunner` binds the wrong blob and returns the `-1` error
  sentinel for every input.
- **`global_position`/`to_local()` are unreliable headless** — nodes added via `add_child` in a
  `--script` test's `_initialize()` aren't `is_inside_tree()`, so `global_position` errors
  (`Condition "!is_inside_tree()" is true`) and `to_local()` returns identity. For a child→parent-local
  conversion that doesn't need the tree use `parent.transform * child.position` (equals
  `to_local(child.global_position)` when `parent` is a direct child, and is offset-invariant — this is
  how `RoverGame.read_obstacles` stays tile-offset-safe for `ParallelArena`).
- **INT8 quantize tools are NOT in the pip `ncnn` wheel** and the static-lib build sets
  `NCNN_BUILD_TOOLS=OFF`. Build `ncnn2table`/`ncnn2int8`/`ncnnoptimize` once with
  `scripts/build_ncnn_tools.sh` (uses `NCNN_SIMPLEOCV=ON`, so no OpenCV; we use the `.npy`
  calibration path `type=1`). The static `libncnn.a` already has `NCNN_INT8=ON`, so
  `NcnnRunner` runs int8 models with no C++ changes.
- **INT8 calibration `.npy` is CHW, normalized /255** (matching `run_inference_image`); the
  `ncnn2table shape=` arg is WHC and is reversed internally. INT8 parity is an **argmax
  agreement rate** (default ≥ 0.9), NOT logit closeness — quantization drifts logits by design.
- **Launching a *training* scene headless without a trainer now times out instead of hanging** —
  `NcnnSync.connect_to_server()` gives up after `connect_timeout_sec` (default 10s, falls back to human
  controls) and `_get_dict_json_message()` quits cleanly after `read_timeout_sec` (default 60s, matches
  godot_rl) if the trainer goes silent. Override per-run with `connect_timeout=` / `read_timeout=`
  cmdline args (seconds; `<= 0` disables). To exercise a scene's spawning/obs without training, still
  prefer a smoke scene with **no `Sync` node** (e.g. `parallel_arena_smoke_scene.tscn`), or just
  `load()` the `.tscn` as a `PackedScene` without instancing it into a running tree.

## Deploy `action_repeat` must match the training cadence (locomotion)

The quadruped hurdles policy trained at `action_repeat=4` collapses under the Sync default of 8
at deploy (~4 m, zero hurdles vs ~31 m / 4 hurdles): motor-velocity targets held twice as long
overshoot the joint swings of a dynamic gait. Statically-stable gaits (M1 walk) happen to
tolerate it, which hides the bug. Pin `action_repeat` on the Sync node of every deploy/eval
scene to the value the policy was trained with (the M2 scenes do).

## Discrete CNN policy trajectories are not portable across CPU architectures

ncnn runs convolutions in fp16 on ARM (Apple Silicon) but fp32 on x86. For a CNN policy this drifts
the output logits by a meaningful amount (~3 on an |8| scale for the visual-chase net). A
*continuous*-control policy absorbs that (motor means shift slightly, the gait holds — see the
quadruped, which passes a behavioral check cross-platform with margin). A *discrete*-argmax policy
does not: the drift flips the argmax on a fraction of frames, and over a long rollout the categorical
trajectory diverges completely — visual-chase catches 9–11/3600 frames locally on ARM but 0 on x86 CI
with the identical net and seed. So do NOT gate CI on a discrete CNN policy's full-trajectory outcome.
Gate on per-frame correctness instead: a golden test over fixed frames asserting argmax, with only
probes whose top-2 logit gap exceeds the cross-platform drift (>= 3) baked. Cover the live wiring with
an integration smoke (model loads, runs through the image route, emits varied actions). The
trajectory/catch-count is a real result — report it in docs as a locally-measured number, not a gate.

## Multiple articulated ragdolls in one Jolt physics space interfere

A single code-built quadruped/hexapod walks ~21 m solo (even at a large lane X offset). But putting
three of them in ONE shared Jolt space (3 × ~14 bodies + joints) makes every gait collapse to ~5 m
or stall — the articulated bodies contend for the solver's iteration budget, not collision (it
happens at 20 m lane spacing too). So a live side-by-side "race" of several ragdrolls is unstable.
The #60 M4 race instead runs each creature SEQUENTIALLY in clean solo physics (model-swap between
phases) and presents a leaderboard — robust, and each distance is the creature's true reach. If you
ever need simultaneous articulated agents, raise the physics solver iterations/substeps first and
verify each agent still matches its solo behaviour.

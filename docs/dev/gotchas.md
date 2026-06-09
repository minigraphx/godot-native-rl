# Operational Gotchas (learned the hard way)

> Long-form companion to `CLAUDE.md`. CLAUDE.md keeps only the few daily-biting items and links
> here for the rest. Contributor-facing.

- **Three venvs:** `.venv` (Python 3.14) has `pnnx`+torch for conversion. `.venv-train` (Python
  **3.13** — torch wheels don't exist for 3.14) has `godot-rl onnxruntime ncnn onnxscript` for SB3 +
  CleanRL training (and runs `export_to_ncnn.py`). `.venv-sf` (Python 3.13) has SampleFactory for the
  SF backend only — isolated because SF pins `gymnasium<1.0`, which would downgrade the SB3/CleanRL
  stack. Keep them separate. All gitignored.
- **`onnxscript` is required** for ONNX export (torch 2.12 dynamo exporter needs it) — not pulled
  in by godot-rl automatically.
- **Pin the onnx chain in `requirements-train.txt` (don't loosen) — and pin onnx to `1.17.0`, not
  `1.19.0`.** `godot-rl 0.8.2` caps `stable-baselines3<=2.4.0`, which caps `numpy<2.0`. Two traps
  follow:
  1. **The sdist trap (install crash).** Leave `ml_dtypes` unpinned and pip backtracks to the old
     `ml_dtypes 0.4.0` **sdist**, whose build-time `numpy==2.0.0rc1` doesn't exist → the whole
     `setup_training.sh` install crashes (seen on a fresh macOS arm64 / Python 3.13 box). Pin
     `ml-dtypes==0.4.1` (the last wheel for numpy<2).
  2. **The import trap (resolves but `import onnx` crashes).** `onnx>=1.18` references
     `ml_dtypes.float4_e2m1fn` at **import** time, which only exists in `ml_dtypes>=0.5` (numpy>=2).
     So `onnx==1.19.0` *resolves* fine under `numpy<2`, but `import onnx` raises
     `AttributeError: module 'ml_dtypes' has no attribute 'float4_e2m1fn'` — killing **every**
     `torch.onnx.export` in `.venv-train` (chase/rover/cleanrl training + the `make_synthetic_*`
     fixture generators), not just SAC. Pin `onnx==1.17.0` — the last onnx that imports under
     `ml_dtypes 0.4.x`.
  Working set: `numpy<2.0`, `ml-dtypes==0.4.1`, `onnx==1.17.0`, `onnx-ir==0.1.8`, `onnxscript==0.5.0`.
  **Verify by import + a real export, not just `pip --dry-run`** (dry-run only proves *resolvability*,
  which is what hid this): in a fresh `python3.13 -m venv` run `pip install -r requirements-train.txt`,
  then `python -c "import onnx"` (must not raise) and `python scripts/make_synthetic_dqn.py` (self-checks
  ONNX↔eager parity). To move to `numpy>=2` / newer onnx you must first move off `godot-rl 0.8.2`
  (it pins `stable-baselines3<=2.4.0`).
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
  both: it **regenerates the cache fresh on every run** (`rm` + `godot --headless --editor --quit`, then
  `git clean -f -- '*.gd.uid'`) before the tests. To do it manually: run that import pass once (or open the
  project in the editor).
- **Don't commit Godot-generated `*.gd.uid` files** — an editor/import pass scatters them (and can
  re-materialize moved scripts at their old paths); `git clean -f -- '*.gd.uid'` and delete stray
  root duplicates before committing.
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

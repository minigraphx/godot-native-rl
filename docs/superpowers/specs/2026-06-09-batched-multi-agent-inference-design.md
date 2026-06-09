# Batched multi-agent inference (C++) — design

**Issue:** [#34](https://github.com/minigraphx/godot-native-rl/issues/34) (Backlog item 23).
**Date:** 2026-06-09.
**Track:** deploy-side inference helper (the moat — native C++/ncnn crowd inference no
Python-server framework can match).

## Problem

Every agent currently runs its own forward pass: `NcnnControllerCore.choose_and_apply_action`
calls `runner.run_inference(obs_vec)` once per agent, and each `NcnnAIController` owns its **own**
`NcnnRunner` (loads the model per agent). For a crowd of N agents that share one policy this is
N separate GDScript↔C++ Variant round-trips and N `Extractor` setups per decision, with N copies
of the same loaded `Net` in memory.

## What "batched inference" actually means here (and what it does not)

ncnn has **no semantic batch dimension on CPU**. Its `InnerProduct`/Linear layers flatten the
whole input `Mat`, so a model exported for `[1, obs_dim]` cannot be fed `[N, obs_dim]` to get N
independent forward passes — ncnn processes one sample per `Extractor`. (True batched matmul is a
Vulkan-only path, out of scope.)

So the win is **not fewer FLOPs**. It is:

1. **One C++ call** processes all N agents instead of N Variant round-trips (less marshalling and
   per-call dispatch overhead).
2. **Thread-level parallelism across agents** — the N independent forward passes are fanned out
   over CPU cores, so a crowd's inference wall-time scales with available cores.
3. **One shared `Net`** in memory for a shared-policy crowd instead of N.

This honest framing is stated in the docs and code comments so expectations are correct.

## Architecture

Three layers, each independently testable:

### 1. C++ primitive — `NcnnRunner::run_inference_batch`

```cpp
Array run_inference_batch(const Array &p_inputs, int p_num_threads = -1);
```

- **Input:** `p_inputs` — an `Array` of `PackedFloat32Array`, one obs vector per agent. Each vector
  is shaped exactly like the single-agent `run_inference` input: validated against `input_shape_`
  when set (via the existing `build_mat_from_shape`), else treated as a flat 1-D `Mat` (via
  `create_input_mat_from_array`). Uses the existing `input_blob_name_` / `output_blob_name_`.
- **Output:** an `Array` of `PackedFloat32Array`, same length and order as `p_inputs`. A per-agent
  failure (bad input size, extract error) yields an **empty** `PackedFloat32Array` in that slot and
  a single `push_error`; other agents still return. Whole-batch precondition failures (model not
  loaded, empty `p_inputs`) `push_error` and return an empty `Array` (matches `run_inference_multi`).
- **Threading (approach A — chunked fan-out):**
  - `W = clamp(p_num_threads if > 0 else hardware_concurrency, 1, N)` worker threads.
  - Partition `[0, N)` into `W` contiguous slices; each worker loops its slice **serially**, creating
    one `ncnn::Extractor` per agent (extractors are single-use) and calling
    `ex.set_num_threads(1)` so we do not nest with ncnn's intra-layer OpenMP.
  - `ncnn::Net` is safe for concurrent extractors, so no per-agent locking is needed; each worker
    writes only its own output slots (no shared mutation, no race).
  - `W <= 1` runs a plain serial loop on the calling thread (no thread spawn).
  - **WASM / single-thread builds:** guarded with `#ifdef __EMSCRIPTEN__` (and a
    `hardware_concurrency() <= 1` runtime check) → always the serial path. The web export is
    single-threaded (see `docs/dev/building.md`), so this keeps it correct and dependency-free.
  - Uses `std::thread` only — **no new OpenMP usage in our code**, so issue #103 (shed libgomp on
    Linux) stays unaffected by this change.
- **Reuse:** `build_mat_from_shape` / `create_input_mat_from_array` for inputs,
  `output_mat_to_packed_float_array` for outputs, `run_inference_internal`'s blob-binding logic
  refactored into a small `extract_one(ex, input_mat, r_output)` helper shared by the single and
  batch paths (single-output only — the multi-IO recurrent path keeps `run_inference_multi`).

Bound in `_bind_methods` as
`D_METHOD("run_inference_batch", "inputs", "num_threads"), DEFVAL(-1)`.

### 2. GDScript — `CrowdController` (`addons/godot_native_rl/controllers/crowd_controller.gd`)

A single node that owns **one** shared `NcnnRunner` and drives a registered set of lightweight
crowd agents that share one policy.

- Loads the model once (same `model_param_path` / `model_bin_path` / blob-name pattern as
  `NcnnAIController2D._setup_ncnn_runner`, factored so we don't duplicate the file-read/load logic).
- Holds an ordered list of **crowd agents** (duck-typed, must implement `get_obs()`,
  `get_action_space()`, `set_action()`), discovered by walking the controller's **child subtree**
  in stable `get_children()` (scene-tree) order — same idiom as `collect_sensors`. Agents are
  parented under the `CrowdController`. Stable order keeps the batch index ↔ agent mapping
  reproducible for the parity tests. (A named-group override is intentionally **not** included —
  YAGNI; add later only if dynamically-spawned, scattered crowds need it.)
- `decide()` (called at control cadence):
  1. gather `get_obs()["obs"]` from each agent into an `Array` of `PackedFloat32Array`
     (optional shared `ObsNormalize` applied per-vector, like the core does);
  2. `var outs := _runner.run_inference_batch(inputs, num_threads)`;
  3. for each agent, decode its `outs[i]` via the existing `ActionDecode.decode_actions` against
     that agent's `get_action_space()` and call `set_action()`; skip agents whose output slot is
     empty (with a `push_error`).
- `@export var num_threads := -1` surfaces the thread count; `@export` for the group name and a
  `deterministic_inference` flag mirrored to a shared RNG, reusing the core's existing decode path.
- Pure-helper split: a `crowd_math.gd` (or static methods) for the gather/scatter index logic so it
  is unit-testable without a live `Net`. The action space and decode are **per agent**, so a crowd
  may mix obs/action layouts as long as they share the policy net.

### 3. Example — `examples/chase_the_target/chase_crowd*`

Reuses the **already-committed** `examples/chase_the_target/models/chase_the_target.ncnn.{param,bin}`
(5-dim obs, 5 discrete actions) — no new training.

- `chase_crowd_game.gd`: spawns K independent chaser+target pairs in a tiled arena (reuse
  `ChaseGame`'s pure obs/step helpers; one `ChaseGame` per pair or a multi-pair variant).
- Lightweight `CrowdChaseAgent` (Node2D) implementing the duck-typed contract by delegating to the
  existing `ChaseAgent.compute_obs` / `action_index_to_velocity` pure helpers — **no per-agent
  runner**, no reward machinery (inference-only deploy).
- `chase_crowd.tscn`: a `CrowdController` + K agents, headless-compatible (same pattern as the
  existing standalone play scenes), so it satisfies the "examples ship runnable nets" rule.

## Testing

All headless, wired into `test/run_tests.sh`:

1. **Batch parity golden** (`test/test_batch_inference_golden.gd`): load
   `chase_the_target.ncnn`, build N distinct obs vectors, assert
   `run_inference_batch(inputs)[i] == run_inference(inputs[i])` element-wise (exact — same op path)
   for every i, including N=1 and N=0 (empty Array) edge cases.
2. **Serial == threaded**: same inputs through `run_inference_batch(inputs, 1)` and
   `run_inference_batch(inputs, 8)` produce identical outputs (determinism / no race).
3. **Malformed slot**: one wrong-sized input vector yields an empty slot while the others succeed.
4. **Crowd helper unit test** (`test/test_crowd_controller.gd`): a fake runner returning canned
   batch outputs proves gather → decode → scatter wiring (no native `Net`), covering the
   empty-slot skip.
5. **Crowd scene smoke**: load `chase_crowd.tscn` headless, step it a few frames, assert all agents
   received a valid action and moved (mirrors the existing play-scene smoke tests).

## Docs & housekeeping (same PR)

- **README**: a "Batched / crowd inference" subsection under deploy + the `CrowdController` usage.
- **CLAUDE.md**: add the crowd scene to the examples line and a one-line command note.
- **`docs/dev/DEVELOPMENT.md`**: note the no-batch-dim reality + the chunked fan-out contract.
- **`docs/godot-rl-gap-analysis-2026-06-02.md`**: mark batched multi-agent inference shipped.
- **`docs/BACKLOG.md`**: tick item 23.
- PR `Closes #34`.

## Out of scope (YAGNI / follow-ups)

- True Vulkan batch-dim inference.
- A persistent cross-frame thread pool on `NcnnRunner` (approach B) — only worth it if profiling
  shows per-decision thread-spawn cost matters; file as a follow-up if so.
- Batched **image** (CNN) and batched **recurrent** inference — single-output float path only here.
- Per-agent distinct policies in one batch call (each `run_inference_batch` is one `Net`).

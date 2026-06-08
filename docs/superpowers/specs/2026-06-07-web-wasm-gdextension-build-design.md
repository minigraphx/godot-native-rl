# Web/WASM GDExtension build + buffer-based model loading — design

**Date:** 2026-06-07
**Issue:** #32 (Backlog item 25) — partial. Scope here is the **web/WASM build only**; the
addon-relocation / `plugin.cfg` / Asset Library submission parts of item 25 stay out of scope as
separate sub-projects.

## Goal

Produce a working **single-threaded** WebAssembly build of the `ncnn_runner` GDExtension and
**prove a trained policy runs inference in a real browser**. This makes the headline moat claim
("web/WASM deployment — godot_rl's ONNX/.NET can't") true rather than aspirational.

## Decisions (resolved during brainstorming)

- **Threading:** single-threaded first. `NCNN_THREADS=OFF`, WASM SIMD on, godot-cpp `threads=no`.
  No `SharedArrayBuffer`, therefore **no COOP/COEP headers** required — deploys to itch.io /
  GitHub Pages / any static host with zero server config. Our policies are tiny MLPs, so
  single-threaded inference is plenty. No multi-threaded variant in this scope (YAGNI).
- **Model loading:** switch deploy-side loading from path-based to **buffer-based, uniformly on
  every platform** (not a web-only branch). Web is the forcing function; uniformity is the clean
  outcome.
- **Done bar:** binary **plus** a manual in-browser proof (screenshot + console log), documented.
  Binary-inspection alone is not sufficient for this item.

## Background / why a code change is unavoidable

Today the controllers load a model with:

```gdscript
var absolute_param := ProjectSettings.globalize_path(model_param_path)
var absolute_bin := ProjectSettings.globalize_path(model_bin_path)
_ncnn_runner.load_model(absolute_param, absolute_bin)   # ncnn fopen(path)
```

On web, Godot packs assets into a `.pck` served through its own VFS. There is **no real file** for
libc `fopen` to open, so the current path-based load fails in-browser. ncnn exposes memory loaders
(`load_param_mem()` for the text `.param`; a memory `DataReader` —
`ncnn::DataReaderFromMemory` — for the `.bin` weights), so the fix is to read the two files via
Godot `FileAccess` and hand ncnn the bytes.

### Multi-model is preserved

"More than one model" works by having **one `NcnnRunner` instance per model**, not one runner
holding several. Each `NcnnRunner` wraps its own `ncnn::Net` (`net_ = make_unique<ncnn::Net>()`),
which a load call resets; controllers each do `NcnnRunner.new()`. The multi-policy hide & seek eval
scene runs a separate seeker and hider net side by side this way.

The buffer-load change is **per-instance and one-for-one**: `load_model_from_buffers(param, bin)`
replaces `load_model(path, path)` on the *same* runner, changing only *where the bytes come from*,
not *how many nets exist*. N models = N runners, unchanged. `run_inference_multi` (multi-IO /
recurrent, a separate single-net capability) is unaffected. This is locked down by a test that
loads **two** runners simultaneously (the multi-policy pair) and checks buffer-load vs path-load
parity for both.

## Prong 1 — Buffer-based model loading (prerequisite code change)

**C++ (`src/ncnn_runner.{h,cpp}`):**
- Add `bool load_model_from_buffers(const PackedByteArray &param, const PackedByteArray &bin)`.
  - Param: copy bytes to a NUL-terminated buffer, `net_->load_param_mem(...)`.
  - Bin: wrap bytes in `ncnn::DataReaderFromMemory`, `net_->load_model(dr)`.
  - Same `net_` reset + `model_loaded_` bookkeeping as `load_model`.
  - Bind the method in `_bind_methods`.
- Keep `load_model(param_path, bin_path)` as a thin wrapper (read files → delegate) so existing
  callers and tests keep working unchanged.

**GDScript (`addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` + `_3d.gd`):**
- Replace `globalize_path` + `load_model` with
  `FileAccess.get_file_as_bytes("res://…")` for each of `.param`/`.bin` → `load_model_from_buffers`.
- Validate both reads (empty / missing file → existing `push_error` path, fail closed exactly as
  today). Drop the now-unused `globalize_path` calls.

**Tests:**
- Existing golden/inference GDScript tests already exercise these controllers → regress the change
  on desktop for free.
- New parity unit: on the committed chase model, assert `load_model_from_buffers` and
  path-based `load_model` produce **identical** inference output. Extend to load **two** runners
  simultaneously (the multi-policy seeker/hider pair) and check parity for both, locking in the
  multi-model guarantee.

## Prong 2 — The emscripten build

- New `scripts/cross/build_web.sh`, mirroring `scripts/cross/build_zig.sh`:
  - ncnn: `emcmake cmake -S thirdparty/ncnn -B build-web-wasm32 -DNCNN_THREADS=OFF
    -DNCNN_OPENMP=OFF -DNCNN_VULKAN=OFF -DNCNN_SIMD=ON -DNCNN_BUILD_TOOLS=OFF
    -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF
    -DBUILD_SHARED_LIBS=OFF -DNCNN_SHARED_LIB=OFF` (exact flags verified during the build);
    static `libncnn.a`, install into the per-target dir the existing `SConstruct` already probes.
  - extension: `scons platform=web target=template_{debug,release} threads=no ncnn_openmp=no`.
- Output: `bin/libncnn_runner.web.template_{debug,release}.wasm`.
- Add web keys to `ncnn_runner.gdextension`
  (`web.debug.wasm32` / `web.release.wasm32` — exact feature tag/suffix verified against
  godot-cpp's actual web output during the build).
- Pin the **emsdk version** in the script + `docs/dev/building.md`, matching what godot-cpp `4.5`
  expects.
- Add web as a **compile-only** job to `.github/workflows/cross-build.yml` (CI can't run a
  browser), consistent with the other cross-targets.

## Prong 3 — In-browser proof (the done bar)

- Export `examples/chase_the_target` (smallest 2D example, committed trained model) to web with the
  extension-capable template, single-threaded.
- Serve it as a **plain static directory — no COOP/COEP headers** (the whole point of
  single-threaded) and load it in a browser; confirm the agent chases the target driven by native
  ncnn inference.
- Capture a screenshot + the JS console line proving `NcnnRunner` loaded and inferred.
- Document the full recipe (emsdk setup, build, export, serve, success criteria) in
  `docs/dev/building.md`; note completion in the issue/backlog.

## Risks — de-risk first (spike at the top of the implementation plan)

1. **GATING:** Godot web GDExtension export needs an extension-capable web template with
   `threads=no`. If Godot 4.5's stock web template does **not** support single-threaded
   GDExtension, the single-threaded approach is blocked and the only path is `threads=yes` (which
   reopens COOP/COEP). **Verify this before building anything.** If it fails, **stop and bring
   options back to the user** — do not silently switch to a COOP/COEP-requiring multi-threaded
   build.
2. Exact web library **feature tag/suffix** in the `.gdextension` (`wasm32` vs thread variants).
3. **emsdk ↔ godot-cpp `4.5`** version compatibility.

## Scope boundaries (YAGNI)

- Single-threaded only — no multi-threaded web variant.
- Web build only — addon relocation, `plugin.cfg` metadata, and Asset Library submission (the rest
  of item 25) are out of scope, each its own sub-project.
- The buffer-load change is the one piece of "improve the code I'm touching" included, because web
  literally cannot work without it.

## Definition of done

- [ ] `load_model_from_buffers` lands; controllers use it uniformly; parity test (incl. two
      simultaneous runners) green; full desktop suite green.
- [ ] `scripts/cross/build_web.sh` produces `bin/libncnn_runner.web.template_{debug,release}.wasm`;
      manifest has web keys; web compile-only CI job green.
- [ ] One example exported to web, run in a real browser with **no special headers**, screenshot +
      console proof captured, recipe documented in `building.md`.
- [ ] Issue #32 comment + `docs/BACKLOG.md` updated to reflect the web sub-task as done.

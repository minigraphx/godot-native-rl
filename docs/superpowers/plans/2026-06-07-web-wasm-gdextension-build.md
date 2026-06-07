# Web/WASM GDExtension Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a single-threaded WebAssembly build of the `ncnn_runner` GDExtension and prove a trained policy runs ncnn inference in a real browser with no special server headers.

**Architecture:** Three prongs. (1) Switch deploy-side model loading from `fopen`-paths to Godot `FileAccess` bytes uniformly on every platform, because ncnn cannot `fopen` inside Godot's web `.pck`. (2) An emscripten build of static ncnn (`NCNN_THREADS=OFF`, SIMD on) + `scons platform=web threads=no`, emitting `.wasm` and a `web.*` manifest key. (3) A documented manual in-browser proof exporting `chase_the_target` and serving it as plain static files.

**Tech Stack:** GDExtension (godot-cpp `4.5`), ncnn (tag `20260526`), SCons, CMake, emsdk/emscripten, GDScript headless test harness.

**Source spec:** `docs/superpowers/specs/2026-06-07-web-wasm-gdextension-build-design.md`

**Branch:** `feat/web-wasm-build` (already created, design doc committed).

---

## File Structure

- `src/ncnn_runner.h` / `src/ncnn_runner.cpp` — add `load_model_from_buffers(PackedByteArray, PackedByteArray)`; keep `load_model(path, path)` as a wrapper.
- `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` / `_3d.gd` — read bytes via `FileAccess`, call `load_model_from_buffers`.
- `test/unit/test_buffer_load_parity.gd` — new GDScript regression: buffer-load == path-load, including **two runners loaded simultaneously**.
- `scripts/cross/build_web.sh` — new emscripten build script (mirrors `scripts/cross/build_zig.sh`).
- `ncnn_runner.gdextension` — add `web.debug.wasm32` / `web.release.wasm32` keys.
- `.github/workflows/cross-build.yml` — add a compile-only `web` matrix leg.
- `docs/dev/building.md` — web build + export + serve + in-browser proof recipe.
- `docs/BACKLOG.md` + issue #32 — mark the web sub-task done.

---

## Task 1: De-risk gate — confirm single-threaded web GDExtension is viable

**This task gates everything else. If it fails, STOP and report options to the user — do NOT silently switch to a multi-threaded (COOP/COEP-requiring) build.**

**Files:** none (investigation + notes).

- [ ] **Step 1: Install + activate emsdk pinned to a godot-cpp 4.5-compatible version**

```bash
cd ~ && git clone https://github.com/emscripten-core/emsdk.git 2>/dev/null || true
cd ~/emsdk && ./emsdk install 3.1.64 && ./emsdk activate 3.1.64
source ~/emsdk/emsdk_env.sh
emcc --version   # expect: emcc ... 3.1.64
```

Note: 3.1.64 is the starting candidate (matches the emscripten range godot-cpp `4.5` documents). If the godot-cpp web build rejects it, try the version named in `godot-cpp/tools/web.py` / its CI, and record the working version in notes.

- [ ] **Step 2: Confirm godot-cpp 4.5 compiles for `platform=web threads=no`**

```bash
cd "/Users/andreas/Documents/Godot Native RL"
test -d godot-cpp || git clone -b 4.5 --depth 1 https://github.com/godotengine/godot-cpp.git
source ~/emsdk/emsdk_env.sh
cd godot-cpp && scons platform=web target=template_debug threads=no -j4 ; cd ..
```

Expected: builds `godot-cpp/bin/libgodot-cpp.web.template_debug.wasm32.a` (or `.nothreads.wasm32.a`). Record the exact filename produced — it determines the manifest feature tag in Task 6.

- [ ] **Step 3: Confirm Godot 4.5 can export a GDExtension web build single-threaded**

Verify the editor in use is 4.5+ and that the **Web export template supports GDExtension single-threaded**. Do a documentation/changelog check plus confirm the export preset exposes "Thread Support" as toggleable:

```bash
"${GODOT:-$(command -v godot || command -v godot-mono)}" --version
```

Decision: Godot ≥ 4.4 ships web GDExtension support; 4.5 expected OK. Confirm "Thread Support" can be **disabled** in the Web preset while still loading the extension.

- [ ] **Step 4: Record go/no-go**

Write a 3–5 line note in the plan PR / task output: emsdk version that works, the godot-cpp web archive filename (→ feature tag), and GO or NO-GO. **If NO-GO (single-threaded web GDExtension unsupported), STOP and report to the user.**

- [ ] **Step 5: Commit notes (if any scratch files)**

No code commit expected here; proceed to Task 2 only on GO.

---

## Task 2: `load_model_from_buffers` (C++) + parity test

**Files:**
- Modify: `src/ncnn_runner.h:32`
- Modify: `src/ncnn_runner.cpp:39` (add new method after `load_model`)
- Test: `test/unit/test_buffer_load_parity.gd` (create)

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_buffer_load_parity.gd`:

```gdscript
extends SceneTree
# Regression: load_model_from_buffers (bytes via FileAccess) must produce identical inference to
# the path-based load_model, and must work for TWO runners loaded simultaneously (multi-model =
# multiple NcnnRunner instances, the multi-policy pattern). Web cannot fopen inside Godot's .pck,
# so the buffer path is the deploy path on every platform; this pins it against regressions.

const MODEL_PARAM := "res://models/chase_sf_policy.ncnn.param"
const MODEL_BIN   := "res://models/chase_sf_policy.ncnn.bin"
const Harness = preload("res://test/harness.gd")

const OBS: Array = [
	[ 0.5479, -0.1222,  0.7172,  0.3947, -0.8116],
	[ 0.9512,  0.5223,  0.5721, -0.7438, -0.0992],
	[-0.2584,  0.8535,  0.2877,  0.6455, -0.1132],
]

func _make_runner() -> NcnnRunner:
	var r := NcnnRunner.new()
	r.input_blob_name = "in0"
	r.output_blob_name = "out0"
	return r

func _initialize() -> void:
	var h := Harness.new()

	# Path-based reference runner.
	var ref := _make_runner()
	var ok_ref := ref.load_model(
		ProjectSettings.globalize_path(MODEL_PARAM),
		ProjectSettings.globalize_path(MODEL_BIN))
	h.assert_true(ok_ref, "reference path-load succeeds")

	# Two buffer-loaded runners loaded simultaneously (multi-model guarantee).
	var param_bytes := FileAccess.get_file_as_bytes(MODEL_PARAM)
	var bin_bytes := FileAccess.get_file_as_bytes(MODEL_BIN)
	h.assert_true(param_bytes.size() > 0, "param bytes read")
	h.assert_true(bin_bytes.size() > 0, "bin bytes read")

	var a := _make_runner()
	var b := _make_runner()
	var ok_a := a.load_model_from_buffers(param_bytes, bin_bytes)
	var ok_b := b.load_model_from_buffers(param_bytes, bin_bytes)
	h.assert_true(ok_a, "buffer-load runner A succeeds")
	h.assert_true(ok_b, "buffer-load runner B succeeds while A is loaded")

	if ok_ref and ok_a and ok_b:
		for obs_values in OBS:
			var obs := PackedFloat32Array(obs_values)
			var want := ref.run_discrete_action(obs)
			h.assert_eq(a.run_discrete_action(obs), want, "A parity for %s" % str(obs_values))
			h.assert_eq(b.run_discrete_action(obs), want, "B parity for %s" % str(obs_values))

	ref.free(); a.free(); b.free()
	h.finish(self)
```

- [ ] **Step 2: Run it and verify it fails (method missing)**

```bash
cd "/Users/andreas/Documents/Godot Native RL"
"${GODOT:-$(command -v godot || command -v godot-mono)}" --headless --path . --script res://test/unit/test_buffer_load_parity.gd
```

Expected: FAIL — `Invalid call. Nonexistent function 'load_model_from_buffers'`.

- [ ] **Step 3: Declare the method in the header**

In `src/ncnn_runner.h`, after line 32 (`bool load_model(...)`) add:

```cpp
    bool load_model_from_buffers(const PackedByteArray &p_param, const PackedByteArray &p_bin);
```

Add the include near the other variant includes (after line 11):

```cpp
#include <godot_cpp/variant/packed_byte_array.hpp>
```

- [ ] **Step 4: Implement the method + bind it**

In `src/ncnn_runner.cpp`, add the datareader include near the top (after `#include <net.h>`):

```cpp
#include <datareader.h>
```

Add this method immediately after `load_model`'s closing brace (after `src/ncnn_runner.cpp:67`):

```cpp
bool NcnnRunner::load_model_from_buffers(const PackedByteArray &p_param, const PackedByteArray &p_bin) {
    if (p_param.is_empty() || p_bin.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner.load_model_from_buffers: param and bin buffers must be non-empty.");
        return false;
    }

    net_ = std::make_unique<ncnn::Net>();
    model_loaded_ = false;

    // ncnn's load_param_mem() needs a NUL-terminated C string of the text .param.
    std::vector<char> param_text(p_param.size() + 1);
    std::memcpy(param_text.data(), p_param.ptr(), p_param.size());
    param_text[p_param.size()] = '\0';

    const int param_result = net_->load_param_mem(param_text.data());
    if (param_result != 0) {
        UtilityFunctions::push_error("NcnnRunner.load_model_from_buffers: failed to parse param buffer.");
        net_.reset();
        return false;
    }

    // The .bin weights load from an advancing memory cursor via DataReaderFromMemory.
    const unsigned char *bin_cursor = reinterpret_cast<const unsigned char *>(p_bin.ptr());
    ncnn::DataReaderFromMemory bin_reader(bin_cursor);
    const int bin_result = net_->load_model(bin_reader);
    if (bin_result != 0) {
        UtilityFunctions::push_error("NcnnRunner.load_model_from_buffers: failed to load bin buffer.");
        net_.reset();
        return false;
    }

    model_loaded_ = true;
    return true;
}
```

Add `#include <vector>` to the cpp's standard includes if not present (it already includes `<cstring>` for `memcpy`; check the top block and add `<vector>`).

Bind the method in `_bind_methods` (next to the existing `load_model` bind, ~`src/ncnn_runner.cpp:20`):

```cpp
    ClassDB::bind_method(D_METHOD("load_model_from_buffers", "param", "bin"), &NcnnRunner::load_model_from_buffers);
```

- [ ] **Step 5: Rebuild the macOS extension**

```bash
cd "/Users/andreas/Documents/Godot Native RL"
scons platform=macos arch=arm64 target=template_debug -j8
```

Expected: links `bin/libncnn_runner.macos.template_debug.arm64.dylib` with no errors.

- [ ] **Step 6: Run the parity test, verify it passes**

```bash
"${GODOT:-$(command -v godot || command -v godot-mono)}" --headless --path . --script res://test/unit/test_buffer_load_parity.gd
```

Expected: all assertions pass; harness prints `0 failed`.

- [ ] **Step 7: Commit**

```bash
git add src/ncnn_runner.h src/ncnn_runner.cpp test/unit/test_buffer_load_parity.gd
git commit -m "feat: NcnnRunner.load_model_from_buffers (bytes via FileAccess) + parity test"
```

---

## Task 3: Switch controllers to uniform buffer-loading

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd:76-81`
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd:76-81`

- [ ] **Step 1: Replace the load block in `ncnn_ai_controller_2d.gd`**

Replace lines 76–81 (the `globalize_path` + `load_model` block) with:

```gdscript
	var param_bytes := FileAccess.get_file_as_bytes(model_param_path)
	var bin_bytes := FileAccess.get_file_as_bytes(model_bin_path)
	if param_bytes.is_empty() or bin_bytes.is_empty():
		push_error("NcnnAIController2D: cannot read model files '%s' / '%s'." % [model_param_path, model_bin_path])
		_ncnn_runner.queue_free()
		_ncnn_runner = null
		return
	if not _ncnn_runner.load_model_from_buffers(param_bytes, bin_bytes):
		push_error("NcnnAIController2D: failed to load ncnn model.")
		_ncnn_runner.queue_free()
		_ncnn_runner = null
```

- [ ] **Step 2: Make the identical change in `ncnn_ai_controller_3d.gd`**

Replace the matching lines 76–81 in `ncnn_ai_controller_3d.gd` with the same block, but with the error string prefix `NcnnAIController3D:` instead of `NcnnAIController2D:`.

- [ ] **Step 3: Run the full test suite**

```bash
cd "/Users/andreas/Documents/Godot Native RL"
./test/run_tests.sh
```

Expected: ends with `All tests passed.` (golden/inference/trained-chase tests exercise both controllers, regressing the change on desktop).

- [ ] **Step 4: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd
git commit -m "refactor: controllers load models via FileAccess buffers (uniform, web-safe)"
```

---

## Task 4: `scripts/cross/build_web.sh`

**Files:**
- Create: `scripts/cross/build_web.sh`

- [ ] **Step 1: Write the build script**

Create `scripts/cross/build_web.sh` (mirrors `scripts/cross/build_zig.sh`):

```bash
#!/usr/bin/env bash
# Build ncnn (static, single-threaded) + the GDExtension for the web (WASM) target via emscripten.
# Single-threaded by design: NCNN_THREADS=OFF + scons threads=no, so the exported game needs NO
# COOP/COEP headers and deploys to any static host (itch.io, GitHub Pages). See docs/dev/building.md.
#
# Usage: source your emsdk env first, then: scripts/cross/build_web.sh
# Requires: emsdk activated (emcc on PATH), cmake, scons, python3; ./godot-cpp + ./thirdparty/ncnn.
# Env: NCNN_JOBS caps ncnn compile parallelism (default = CPU count).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

command -v emcc >/dev/null || { echo "emcc not found; source your emsdk_env.sh first" >&2; exit 2; }

CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
NCNN_JOBS="${NCNN_JOBS:-$CPUS}"

ncnn_build="thirdparty/ncnn/build-web-wasm32"
if [ ! -f "$ncnn_build/install/lib/libncnn.a" ]; then
  emcmake cmake -S thirdparty/ncnn -B "$ncnn_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DNCNN_SHARED_LIB=OFF \
    -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF -DNCNN_VULKAN=OFF -DNCNN_SIMD=ON
  cmake --build "$ncnn_build" --parallel "$NCNN_JOBS"
  cmake --install "$ncnn_build" --prefix "$repo/$ncnn_build/install"
fi

for cfg in template_debug template_release; do
  scons platform=web arch=wasm32 target="$cfg" threads=no ncnn_openmp=no -j"$CPUS"
done

echo "== built web wasm32 =="
ls -la bin/ | grep -i web || true
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/cross/build_web.sh
```

- [ ] **Step 3: Run it end-to-end (emsdk must be active)**

```bash
cd "/Users/andreas/Documents/Godot Native RL"
source ~/emsdk/emsdk_env.sh
scripts/cross/build_web.sh
```

Expected: produces `bin/libncnn_runner.web.template_debug.wasm32.wasm` and `...template_release.wasm32.wasm` (exact suffix per godot-cpp's web output recorded in Task 1; adjust the manifest in Task 6 to match).

- [ ] **Step 4: Sanity-check the wasm output**

```bash
ls -la bin/ | grep -i web
file bin/libncnn_runner.web.template_debug.wasm32.wasm   # expect: WebAssembly (wasm) binary module
```

- [ ] **Step 5: Commit**

```bash
git add scripts/cross/build_web.sh
git commit -m "build: emscripten web/WASM build script (single-threaded ncnn)"
```

---

## Task 5: SConstruct — verify the web suffix (only if needed)

**Files:**
- Possibly modify: `SConstruct:111-129` (suffix handling)

- [ ] **Step 1: Check whether the web library name is correct as-emitted**

If Task 4 already produced `bin/libncnn_runner.web.*.wasm` with a valid `.wasm` suffix, **no change is needed** — godot-cpp's web tool sets `SHLIBSUFFIX=.wasm` itself. Skip to Task 6.

- [ ] **Step 2: If the suffix is wrong (e.g. `.dylib`/`.so` on the web output), pin it**

Only if Step 1 shows a wrong suffix, add a web branch alongside the existing platform branches at `SConstruct:115`:

```python
elif env["platform"] == "web":
    env["SHLIBSUFFIX"] = ".wasm"
    env["SHOBJSUFFIX"] = ".o"
```

Then re-run `scripts/cross/build_web.sh` and confirm the `.wasm` name.

- [ ] **Step 3: Commit (only if changed)**

```bash
git add SConstruct
git commit -m "build: pin web SHLIBSUFFIX to .wasm"
```

---

## Task 6: Manifest web keys

**Files:**
- Modify: `ncnn_runner.gdextension`

- [ ] **Step 1: Add web library entries**

Append to the `[libraries]` section of `ncnn_runner.gdextension`, using the exact feature tag recorded in Task 1 / produced in Task 4 (single-threaded → `wasm32`, not a `.threads.` tag):

```ini
web.debug.wasm32 = "res://bin/libncnn_runner.web.template_debug.wasm32.wasm"
web.release.wasm32 = "res://bin/libncnn_runner.web.template_release.wasm32.wasm"
```

- [ ] **Step 2: Verify Godot parses the manifest with the new keys**

```bash
cd "/Users/andreas/Documents/Godot Native RL"
"${GODOT:-$(command -v godot || command -v godot-mono)}" --headless --path . --editor --quit 2>&1 | grep -i "gdextension\|error" | head
```

Expected: no GDExtension parse errors about the web keys (the desktop build still loads as before).

- [ ] **Step 3: Commit**

```bash
git add ncnn_runner.gdextension
git commit -m "build: add web.wasm32 library keys to the GDExtension manifest"
```

---

## Task 7: Compile-only web CI leg

**Files:**
- Modify: `.github/workflows/cross-build.yml`

- [ ] **Step 1: Add a `web` job after the `ios` job**

Append to `.github/workflows/cross-build.yml`:

```yaml
  # Web (WASM) — emscripten, single-threaded. Compile-only; the in-browser proof is manual
  # (documented in docs/dev/building.md), since CI has no browser.
  web:
    name: web
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4

      - name: Install build tools
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential scons cmake git python3

      - name: Setup emscripten
        uses: mymindstorm/setup-emsdk@v14
        with:
          version: 3.1.64

      - name: Clone godot-cpp + ncnn
        run: |
          git clone -b "${GODOT_CPP_BRANCH}" --depth 1 https://github.com/godotengine/godot-cpp.git
          mkdir -p thirdparty
          git clone -b "${NCNN_TAG}" --depth 1 https://github.com/Tencent/ncnn.git thirdparty/ncnn

      - name: Build web (wasm32)
        run: scripts/cross/build_web.sh

      - name: Upload binaries
        uses: actions/upload-artifact@v4
        with:
          name: cross-web
          path: bin/
          if-no-files-found: error
```

Use the emsdk version confirmed in Task 1 (3.1.64 or the corrected value).

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/cross-build.yml
git commit -m "ci: compile-only web/WASM cross-build leg"
```

---

## Task 8: In-browser proof + docs (the done bar)

**Files:**
- Modify: `docs/dev/building.md` (add a "Web (WASM)" section)
- Add: `docs/dev/img/web-chase-proof.png` (screenshot)

- [ ] **Step 1: Build both web configs (if not already present)**

```bash
cd "/Users/andreas/Documents/Godot Native RL"
source ~/emsdk/emsdk_env.sh
scripts/cross/build_web.sh
ls bin/ | grep -i web   # both template_debug + template_release .wasm present
```

- [ ] **Step 2: Export `chase_the_target` to web, single-threaded**

In the Godot editor (4.5): Project → Export → add a **Web** preset, **disable Thread Support**, set Main Scene to `res://examples/chase_the_target/chase_the_target.tscn` (the inference/deploy scene — it's the one wired to `NCNN_INFERENCE` + a trained model), export to `build/web/index.html`. Or headless:

```bash
"${GODOT:-$(command -v godot || command -v godot-mono)}" --headless --path . --export-debug "Web" build/web/index.html
```

Expected: `build/web/index.html`, `.wasm`, `.pck`, `.js` produced, with the extension `.wasm` bundled.

- [ ] **Step 3: Serve as plain static files — NO special headers**

```bash
cd "/Users/andreas/Documents/Godot Native RL/build/web"
python3 -m http.server 8060
```

Open `http://localhost:8060/` in a browser. (A plain `http.server` sends no COOP/COEP headers — that is the point: single-threaded must run without cross-origin isolation.)

- [ ] **Step 4: Confirm inference runs**

In the browser, confirm the chase agent moves toward the target (driven by native ncnn). Open DevTools console and confirm there is no `load_model` / extension error and that the agent acts. Capture:
- a screenshot of the running game → save to `docs/dev/img/web-chase-proof.png`
- the console showing a clean load (no errors)

Success criterion: the agent visibly chases the target in-browser, served with no COOP/COEP headers.

- [ ] **Step 5: Document the recipe in `docs/dev/building.md`**

Add a "Web (WASM)" section covering: emsdk version + `emsdk_env.sh`, `scripts/cross/build_web.sh`, the single-threaded export preset (Thread Support OFF), serving with a plain static server (no COOP/COEP), and embedding the proof screenshot. State explicitly that single-threaded means no special headers and therefore itch.io / GitHub Pages work unmodified.

- [ ] **Step 6: Commit**

```bash
git add docs/dev/building.md docs/dev/img/web-chase-proof.png
git commit -m "docs: web/WASM build + export + in-browser proof recipe"
```

---

## Task 9: Close out the issue + backlog

**Files:**
- Modify: `docs/BACKLOG.md` (item 25 — note the web sub-task done)
- Modify: `CLAUDE.md` (if a web build command belongs in Key commands)

- [ ] **Step 1: Update BACKLOG item 25**

In `docs/BACKLOG.md`, under item 25, note the web/WASM build + in-browser proof is complete (single-threaded, no COOP/COEP), referencing this plan and the spec. Leave the addon-relocation / `plugin.cfg` / Asset Library submission sub-tasks open.

- [ ] **Step 2: Add a web build command to CLAUDE.md Key commands (terse)**

Add one line near the other build commands:

```
- **Build the extension (web/WASM):** `source ~/emsdk/emsdk_env.sh && scripts/cross/build_web.sh` (single-threaded; no COOP/COEP needed at deploy — see docs/dev/building.md).
```

- [ ] **Step 3: Commit**

```bash
git add docs/BACKLOG.md CLAUDE.md
git commit -m "docs: record web/WASM build done under backlog item 25 (#32)"
```

- [ ] **Step 4: Post an issue #32 progress comment**

```bash
gh issue comment 32 --body "Web/WASM sub-task done: single-threaded WASM GDExtension builds via scripts/cross/build_web.sh, loads models from Godot FileAccess buffers (web .pck has no fopen-able files), and a trained chase policy runs native ncnn inference in-browser with NO COOP/COEP headers (itch.io / GitHub Pages work unmodified). Recipe + screenshot in docs/dev/building.md. Still open for this issue: addon relocation, plugin.cfg metadata, Asset Library submission."
```

---

## Final verification before PR

- [ ] `./test/run_tests.sh` ends with `All tests passed.` (desktop, buffer-load path).
- [ ] `scripts/cross/build_web.sh` produces both `.wasm` binaries; `file` reports valid WebAssembly.
- [ ] In-browser proof captured (screenshot + clean console), served with no special headers.
- [ ] Docs updated: `building.md`, `CLAUDE.md`, `BACKLOG.md`; issue #32 commented.
- [ ] Rebase onto latest `origin/main`, then open the PR.

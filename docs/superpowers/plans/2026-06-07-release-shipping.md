# Release Shipping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship tagged releases of the addon (prebuilt binaries, no source needed) via GitHub Releases, with a Godot Asset Library entry pointing at the release-asset zip.

**Architecture:** Move `.gdextension` + the gitignored `bin/` output into `addons/godot_native_rl/` so the repo tree is a canonical drop-in. A tag-triggered `release.yml` workflow builds all platforms (debug+release), assembles a lean addon zip + a lean examples zip, runs a drop-in smoke, and publishes a GitHub Release. AssetLib uses the `Custom` download provider pointed at the addon zip (binaries never enter git).

**Tech Stack:** SCons + godot-cpp + ncnn (C++ GDExtension), zig/NDK/Xcode cross-builds, GitHub Actions, Godot 4.5 headless.

**Spec:** `docs/superpowers/specs/2026-06-07-release-shipping-design.md`

**Branch:** `feat/release-shipping-32` (already created; the spec is committed there).

---

## File Structure

**Moved/modified (restructure):**
- `ncnn_runner.gdextension` → `addons/godot_native_rl/ncnn_runner.gdextension` (manifest paths repointed)
- `SConstruct:127` — output target → `addons/godot_native_rl/bin/`
- `scripts/cross/build_ios.sh:49,52,53` — xcframework assembly paths → addon bin
- `.gitignore:5` — `bin/` → `addons/godot_native_rl/bin/`
- `.github/workflows/ci.yml:108,135` — artifact upload/download path → addon bin
- `.github/workflows/cross-build.yml:74,100` — artifact upload path → addon bin
- Cosmetic `ls` log lines: `build_zig.sh:45`, `build_android.sh:43`, `build_ios.sh:58`
- Docs: `docs/dev/building.md`, `docs/dev/gotchas.md`, `docs/dev/DEVELOPMENT.md`, `addons/godot_native_rl/plugin_runtime_check.gd:13`, `CLAUDE.md`

**Created (release machinery):**
- `.github/workflows/release.yml` — tag-triggered build + package + smoke + publish
- `docs/dev/RELEASING.md` — the release runbook (tag → CI → AssetLib edit)

**Modified (docs/housekeeping):**
- `README.md` — new Installation section
- `docs/BACKLOG.md:426` — tick item 25
- `CLAUDE.md` — release commands

---

## Phase 1 — Restructure: move the extension into the addon

### Task 1: Move and repoint the `.gdextension` manifest

**Files:**
- Move: `ncnn_runner.gdextension` → `addons/godot_native_rl/ncnn_runner.gdextension`
- Modify: the moved manifest's `[libraries]` paths

- [ ] **Step 1: Move the manifest with git**

```bash
cd "$(git rev-parse --show-toplevel)"
git mv ncnn_runner.gdextension addons/godot_native_rl/ncnn_runner.gdextension
```

- [ ] **Step 2: Repoint all library paths to the addon bin/**

Edit `addons/godot_native_rl/ncnn_runner.gdextension`: replace every `res://bin/` with `res://addons/godot_native_rl/bin/`. The result must be exactly:

```ini
[configuration]
entry_symbol = "ncnn_runner_library_init"
compatibility_minimum = "4.5"

[libraries]
macos.debug.arm64 = "res://addons/godot_native_rl/bin/libncnn_runner.macos.template_debug.arm64.dylib"
macos.release.arm64 = "res://addons/godot_native_rl/bin/libncnn_runner.macos.template_release.arm64.dylib"
windows.debug.x86_64 = "res://addons/godot_native_rl/bin/libncnn_runner.windows.template_debug.x86_64.dll"
windows.release.x86_64 = "res://addons/godot_native_rl/bin/libncnn_runner.windows.template_release.x86_64.dll"
linux.debug.x86_64 = "res://addons/godot_native_rl/bin/libncnn_runner.linux.template_debug.x86_64.so"
linux.release.x86_64 = "res://addons/godot_native_rl/bin/libncnn_runner.linux.template_release.x86_64.so"
ios.debug = "res://addons/godot_native_rl/bin/libncnn_runner.ios.template_debug.xcframework"
ios.release = "res://addons/godot_native_rl/bin/libncnn_runner.ios.template_release.xcframework"
android.debug.arm64 = "res://addons/godot_native_rl/bin/libncnn_runner.android.template_debug.arm64.so"
android.release.arm64 = "res://addons/godot_native_rl/bin/libncnn_runner.android.template_release.arm64.so"
android.debug.x86_64 = "res://addons/godot_native_rl/bin/libncnn_runner.android.template_debug.x86_64.so"
android.release.x86_64 = "res://addons/godot_native_rl/bin/libncnn_runner.android.template_release.x86_64.so"
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: move ncnn_runner.gdextension into addon, repoint lib paths (#32)"
```

### Task 2: Repoint the SCons output target + iOS xcframework paths + .gitignore

**Files:**
- Modify: `SConstruct:127`
- Modify: `scripts/cross/build_ios.sh:49,52,53`
- Modify: `.gitignore:5`

- [ ] **Step 1: Repoint the SCons `SharedLibrary` target**

In `SConstruct`, replace the `target=` line (currently line 127):

```python
library = env.SharedLibrary(
    target=os.path.join("addons", "godot_native_rl", "bin", "libncnn_runner{}{}".format(env["suffix"], env["SHLIBSUFFIX"])),
    source=sources,
)
```

- [ ] **Step 2: Repoint the iOS xcframework assembly**

In `scripts/cross/build_ios.sh`, the `-create-xcframework` block (lines ~49-53) must read/write under the addon bin. Replace those three `bin/` paths:

```sh
  out="addons/godot_native_rl/bin/libncnn_runner.ios.$cfg.xcframework"
  rm -rf "$out"
  xcodebuild -create-xcframework \
    -library "addons/godot_native_rl/bin/libncnn_runner.ios.$cfg.arm64.dylib" \
    -library "addons/godot_native_rl/bin/libncnn_runner.ios.$cfg.universal.simulator.dylib" \
    -output "$out"
```

(Keep the surrounding loop/structure; only the three paths change. The per-arch dylibs are produced by `scons`, which now writes into the addon bin via Task 2 Step 1, so they already land there.)

- [ ] **Step 3: Repoint `.gitignore`**

In `.gitignore`, change the `bin/` line (line 5) to:

```gitignore
addons/godot_native_rl/bin/
```

- [ ] **Step 4: Rebuild locally and verify the binary lands in the new path**

Run:
```bash
scons platform=macos arch=arm64 target=template_debug
ls addons/godot_native_rl/bin/libncnn_runner.macos.template_debug.arm64.dylib
```
Expected: the `.dylib` exists at the new path; no `bin/` directory recreated at repo root.

- [ ] **Step 5: Run the full test suite (proves the manifest path repoint is correct)**

Run: `./test/run_tests.sh`
Expected: all green (the extension loads from `res://addons/godot_native_rl/bin/…`; if the path were wrong, `NcnnRunner` would fail to instantiate and inference tests would fail).

- [ ] **Step 6: Commit**

```bash
git add SConstruct scripts/cross/build_ios.sh .gitignore
git commit -m "build: output GDExtension into addon bin/, repoint iOS + gitignore (#32)"
```

### Task 3: Repoint CI workflow artifact paths + cosmetic log/doc references

**Files:**
- Modify: `.github/workflows/ci.yml:108,135`
- Modify: `.github/workflows/cross-build.yml:74,100`
- Modify: `scripts/cross/build_zig.sh:45`, `scripts/cross/build_android.sh:43`, `scripts/cross/build_ios.sh:58`
- Modify: `docs/dev/building.md`, `docs/dev/gotchas.md`, `docs/dev/DEVELOPMENT.md`, `addons/godot_native_rl/plugin_runtime_check.gd:13`

- [ ] **Step 1: Repoint `ci.yml` upload + download paths**

In `.github/workflows/ci.yml`, both `path: bin/` occurrences (the build job's "Upload extension binaries", line ~108, and the test job's "Download extension binaries", line ~135) become:

```yaml
          path: addons/godot_native_rl/bin/
```

- [ ] **Step 2: Repoint `cross-build.yml` upload paths**

In `.github/workflows/cross-build.yml`, both `path: bin/` occurrences (lines ~74 and ~100) become:

```yaml
          path: addons/godot_native_rl/bin/
```

- [ ] **Step 3: Fix cosmetic `ls` log lines in the cross scripts**

Replace `bin/` with `addons/godot_native_rl/bin/` in these log-only lines:
- `scripts/cross/build_zig.sh:45` → `ls -la addons/godot_native_rl/bin/ | grep "$plat" || true`
- `scripts/cross/build_android.sh:43` → `ls -la addons/godot_native_rl/bin/ | grep android || true`
- `scripts/cross/build_ios.sh:58` → `ls -d addons/godot_native_rl/bin/*ios* || true`

- [ ] **Step 4: Update doc/path mentions of `bin/`**

Update these human-facing references so they name the new path (search each file for `bin/` and fix the GDExtension-output mentions — leave `.venv/bin`, `tools-bin`, and `/usr/bin` style references untouched):
- `docs/dev/building.md` (the "Build output is written to `bin/`" line ~119 and the iOS example lines ~293-297 with `res://bin/…`)
- `docs/dev/gotchas.md:23` ("`bin/` is gitignored" → `addons/godot_native_rl/bin/`)
- `docs/dev/DEVELOPMENT.md:194` (same gitignored mention)
- `addons/godot_native_rl/plugin_runtime_check.gd:13` (error string "missing from bin/" → "missing from addons/godot_native_rl/bin/")

- [ ] **Step 5: Verify no stale GDExtension `bin/` references remain**

Run:
```bash
git grep -nE "res://bin/|[^./a-zA-Z-]bin/lib|path: bin/" -- . ':!godot-cpp' ':!thirdparty'
```
Expected: no output (all GDExtension `bin/` references now point at the addon path).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "ci+docs: repoint bin/ references to addon path (#32)"
```

---

## Phase 2 — Release workflow

### Task 4: Add the tag-triggered release workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Write the workflow file**

Create `.github/workflows/release.yml` with exactly this content:

```yaml
name: Release

# Tag-triggered. Push vX.Y.Z to build all platforms (debug+release), assemble the addon +
# examples zips, smoke-test the packaged addon, and publish a GitHub Release. The addon zip's
# sha256 is printed in the release notes for the Asset Library Custom-download entry.
on:
  push:
    tags: ["v*"]

permissions:
  contents: write   # create the GitHub Release

env:
  GODOT_CPP_BRANCH: "4.5"
  NCNN_TAG: "20260526"
  NCNN_JOBS: "2"
  ADDON_BIN: addons/godot_native_rl/bin

jobs:
  guard:
    name: Version guard
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Assert tag matches plugin.cfg version
        run: |
          tag="${GITHUB_REF_NAME#v}"
          cfg="$(sed -n 's/^version="\(.*\)"/\1/p' addons/godot_native_rl/plugin.cfg)"
          echo "tag=$tag  plugin.cfg=$cfg"
          if [ "$tag" != "$cfg" ]; then
            echo "::error::tag v$tag does not match plugin.cfg version $cfg"; exit 1
          fi

  # Linux/Windows (zig) + Android (NDK) — one target per leg, on Linux runners.
  build-unix:
    name: build ${{ matrix.target }}
    needs: guard
    runs-on: ubuntu-24.04
    strategy:
      fail-fast: false
      matrix:
        target: [linux, windows, android-arm64, android-x86_64]
    steps:
      - uses: actions/checkout@v4
      - name: Install build tools
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential scons cmake git python3
      - name: Install zig
        if: matrix.target == 'linux' || matrix.target == 'windows'
        uses: mlugg/setup-zig@v2
        with:
          version: 0.16.0
      - name: Clone godot-cpp + ncnn
        run: |
          git clone -b "${GODOT_CPP_BRANCH}" --depth 1 https://github.com/godotengine/godot-cpp.git
          mkdir -p thirdparty
          git clone -b "${NCNN_TAG}" --depth 1 https://github.com/Tencent/ncnn.git thirdparty/ncnn
      - name: Build
        run: |
          case "${{ matrix.target }}" in
            linux|windows) scripts/cross/build_zig.sh "${{ matrix.target }}" ;;
            android-arm64) ANDROID_NDK_ROOT="$ANDROID_NDK_LATEST_HOME" scripts/cross/build_android.sh arm64 ;;
            android-x86_64) ANDROID_NDK_ROOT="$ANDROID_NDK_LATEST_HOME" scripts/cross/build_android.sh x86_64 ;;
          esac
      - uses: actions/upload-artifact@v4
        with:
          name: bin-${{ matrix.target }}
          path: ${{ env.ADDON_BIN }}/
          if-no-files-found: error

  # macOS host (arm64) + iOS device/simulator — needs Xcode, so a macOS runner.
  build-macos:
    name: build macos+ios
    needs: guard
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - name: Install scons
        run: brew install scons
      - name: Clone godot-cpp + ncnn
        run: |
          git clone -b "${GODOT_CPP_BRANCH}" --depth 1 https://github.com/godotengine/godot-cpp.git
          mkdir -p thirdparty
          git clone -b "${NCNN_TAG}" --depth 1 https://github.com/Tencent/ncnn.git thirdparty/ncnn
      - name: Build macOS host (arm64, debug+release)
        run: |
          scons platform=macos arch=arm64 target=template_debug -j"$(sysctl -n hw.ncpu)"
          scons platform=macos arch=arm64 target=template_release -j"$(sysctl -n hw.ncpu)"
      - name: Build iOS (device + simulator)
        run: scripts/cross/build_ios.sh
      - uses: actions/upload-artifact@v4
        with:
          name: bin-macos-ios
          path: ${{ env.ADDON_BIN }}/
          if-no-files-found: error

  package:
    name: package + publish
    needs: [build-unix, build-macos]
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Collect all platform binaries
        uses: actions/download-artifact@v4
        with:
          pattern: bin-*
          merge-multiple: true
          path: ${{ env.ADDON_BIN }}
      - name: Sanity-check the binary set
        run: |
          ls -la "$ADDON_BIN"
          for f in \
            libncnn_runner.macos.template_release.arm64.dylib \
            libncnn_runner.windows.template_release.x86_64.dll \
            libncnn_runner.linux.template_release.x86_64.so \
            libncnn_runner.android.template_release.arm64.so \
            libncnn_runner.ios.template_release.xcframework ; do
            test -e "$ADDON_BIN/$f" || { echo "::error::missing $f"; exit 1; }
          done
      - name: Assemble zips
        run: |
          ver="${GITHUB_REF_NAME}"
          mkdir -p dist
          # Addon zip: the whole addon (GDScript + .gdextension + bin/), drop-in at a project root.
          zip -r -q "dist/godot-native-rl-addon-${ver}.zip" addons/godot_native_rl
          # Examples zip: lean — example scenes + a sample model + a short README. No binaries.
          cp .github/release/examples-README.md examples/README_RELEASE.md
          zip -r -q "dist/godot-native-rl-examples-${ver}.zip" examples models/chase_cleanrl_policy.ncnn.param models/chase_cleanrl_policy.ncnn.bin examples/README_RELEASE.md \
            -x 'examples/**/*.import'
          ( cd dist && sha256sum ./*.zip > SHA256SUMS.txt )
          cat dist/SHA256SUMS.txt
      - name: Drop-in smoke (extract addon zip into a clean project, load NcnnRunner)
        run: |
          GODOT_URL="https://github.com/godotengine/godot/releases/download/4.5-stable/Godot_v4.5-stable_linux.x86_64.zip"
          curl -fsSL "$GODOT_URL" -o godot.zip
          unzip -q godot.zip
          godot_bin="$(ls Godot_v4.5-stable_linux.x86_64)"
          chmod +x "$godot_bin"
          mkdir -p smoke/proj
          ( cd smoke/proj && unzip -q "$GITHUB_WORKSPACE/dist/godot-native-rl-addon-${GITHUB_REF_NAME}.zip" )
          printf '%s\n' 'config_version=5' '[application]' 'config/features=PackedStringArray("4.5")' > smoke/proj/project.godot
          cat > smoke/proj/smoke.gd <<'EOF'
extends SceneTree
func _init() -> void:
    var r := ClassDB.instantiate("NcnnRunner")
    if r == null:
        push_error("NcnnRunner not registered — extension failed to load")
        quit(1)
    else:
        print("OK: NcnnRunner loaded from packaged addon")
        quit(0)
EOF
          "$GITHUB_WORKSPACE/$godot_bin" --headless --path smoke/proj --script res://smoke.gd
      - name: Create GitHub Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          ver="${GITHUB_REF_NAME}"
          notes="$(mktemp)"
          {
            echo "## Install"
            echo "- **Asset Library:** search \"Godot Native RL\" in the editor's AssetLib tab."
            echo "- **Manual:** download \`godot-native-rl-addon-${ver}.zip\` and unzip at your project root."
            echo "- Examples: also grab \`godot-native-rl-examples-${ver}.zip\` (needs the addon zip too)."
            echo
            echo "### addon zip sha256 (for the Asset Library Custom download)"
            echo '```'
            grep 'addon' dist/SHA256SUMS.txt
            echo '```'
          } > "$notes"
          gh release create "$ver" \
            dist/godot-native-rl-addon-${ver}.zip \
            dist/godot-native-rl-examples-${ver}.zip \
            dist/SHA256SUMS.txt \
            --title "$ver" \
            --notes-file "$notes" \
            --generate-notes
```

- [ ] **Step 2: Create the examples-zip README that the workflow copies in**

Create `.github/release/examples-README.md`:

```markdown
# Godot Native RL — Examples

These are the example scenes (chase, rover, hide & seek, ball chase) plus a sample
trained model. They depend on the **Godot Native RL addon** — download
`godot-native-rl-addon-<version>.zip` from the same release and unzip it at your project
root alongside this `examples/` folder, then open the project in Godot 4.5+.
```

- [ ] **Step 3: Confirm the sample model paths referenced in the examples zip exist**

Run:
```bash
git ls-files models/chase_cleanrl_policy.ncnn.param models/chase_cleanrl_policy.ncnn.bin
```
Expected: both are listed (committed fixtures). If you'd rather ship a different committed model, run `git ls-files 'models/*.ncnn.param'` and update the two `models/…` paths in the `Assemble zips` step accordingly.

- [ ] **Step 4: Lint the workflow YAML**

Run:
```bash
python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/release.yml')); print('yaml ok')"
```
Expected: `yaml ok`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/release.yml .github/release/examples-README.md
git commit -m "ci: tag-triggered release workflow — build, package 2 zips, smoke, publish (#32)"
```

---

## Phase 3 — Docs, runbook, housekeeping

### Task 5: Add the RELEASING runbook

**Files:**
- Create: `docs/dev/RELEASING.md`

- [ ] **Step 1: Write the runbook**

Create `docs/dev/RELEASING.md`:

```markdown
# Releasing

Releases ship prebuilt binaries via **GitHub Releases**; the **Asset Library** entry points at
the release-asset addon zip (the `Custom` download provider — binaries never enter git).
Background + rationale: `docs/superpowers/specs/2026-06-07-release-shipping-design.md`.

## Cut a release

1. Decide the version (SemVer; pre-1.0 ⇒ API/wire protocol may still break).
2. Bump `addons/godot_native_rl/plugin.cfg` `version="X.Y.Z"` if it isn't already, commit.
3. Tag and push:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
4. `release.yml` runs: version guard → build all platforms (debug+release) → assemble
   `godot-native-rl-addon-vX.Y.Z.zip` + `godot-native-rl-examples-vX.Y.Z.zip` → drop-in smoke →
   create the GitHub Release. (A tag/`plugin.cfg` mismatch fails the guard job.)
5. Open the published release; copy the **addon zip sha256** from the notes (or `SHA256SUMS.txt`).

## Update the Asset Library entry

The Asset Library has no write API — this is a manual web edit, once per release.

1. Go to <https://godotengine.org/asset-library> → your asset → **Edit** (first time: **Submit**).
2. Set/confirm:
   - **Repository / Browse URL + Issues URL** → this (public) repo — lets moderators inspect source.
   - **Repository host** → `Custom`.
   - **Download URL** → the addon zip's release-asset link:
     `https://github.com/<owner>/<repo>/releases/download/vX.Y.Z/godot-native-rl-addon-vX.Y.Z.zip`
   - **Download hash** → the addon zip sha256 from step 5.
   - **Version** → `X.Y.Z`; **Godot version** → `4.5`.
3. Submit; wait for moderation approval (first submission only; edits are usually fast).

## Prerequisites (one-time)

- The repo must be **public** (Settings → General → Danger Zone → Change visibility) so the
  AssetLib browse/issues URLs resolve and moderators can read the source.
```

- [ ] **Step 2: Commit**

```bash
git add docs/dev/RELEASING.md
git commit -m "docs: add release runbook (#32)"
```

### Task 6: README Installation section + CLAUDE.md + BACKLOG tick

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/BACKLOG.md:426`

- [ ] **Step 1: Add an Installation section to the README**

Add this section to `README.md` (place it directly above the existing build-from-source / "Build the extension" content, so "use it" precedes "build it"):

```markdown
## Installation (use the addon — no build needed)

You don't need the C++/SCons/ncnn toolchain to *use* this framework — just the prebuilt addon.

- **Asset Library (in-editor):** open the **AssetLib** tab in Godot 4.5+, search
  "Godot Native RL", install. It drops `addons/godot_native_rl/` (with native binaries for
  macOS/Windows/Linux/Android/iOS) into your project.
- **Manual:** download `godot-native-rl-addon-<version>.zip` from
  [Releases](../../releases) and unzip at your project root. For the demo scenes, also grab
  `godot-native-rl-examples-<version>.zip` (drop it in alongside the addon).

Then enable the plugin in **Project → Project Settings → Plugins**.

Building from source (for contributors / new platforms) is covered below.
```

- [ ] **Step 2: Add release commands to CLAUDE.md**

In `CLAUDE.md`, under "## Key commands", add this bullet (group it near the build bullet):

```markdown
- **Cut a release:** bump `addons/godot_native_rl/plugin.cfg` `version=`, then `git tag vX.Y.Z &&
  git push origin vX.Y.Z` → `.github/workflows/release.yml` builds all platforms, assembles the
  addon + examples zips, smoke-tests the packaged addon, and publishes a GitHub Release. Then
  update the Asset Library entry by hand (`Custom` download → the addon-zip URL + sha256). Full
  runbook: [docs/dev/RELEASING.md](docs/dev/RELEASING.md).
```

- [ ] **Step 3: Tick BACKLOG item 25**

In `docs/BACKLOG.md`, change item 25's checkbox from `⬜` to `✅` (line ~426) and append a short
done-note in the same style as the other done items (mention: addon now holds `.gdextension` +
`bin/`; tag-triggered `release.yml`; two zips; AssetLib via `Custom` download URL; `Closes #32`).

- [ ] **Step 4: Verify docs are internally consistent**

Run:
```bash
git grep -n "res://bin/" -- README.md docs/ CLAUDE.md ':!docs/superpowers'
```
Expected: no output (no doc still tells users the old root `bin/` path).

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md docs/BACKLOG.md
git commit -m "docs: installation section, release command, tick backlog 25 (#32)"
```

---

## Phase 4 — Ship

### Task 7: Open the PR (and the manual go-live steps)

- [ ] **Step 1: Push the branch and open a PR**

```bash
git push -u origin feat/release-shipping-32
gh pr create --title "Release shipping: addon packaging + tag-triggered releases (#32)" \
  --body "$(cat <<'EOF'
Implements the release-shipping design (Closes #32).

- Moves `.gdextension` + `bin/` output into `addons/godot_native_rl/` (canonical drop-in).
- Tag-triggered `release.yml`: builds all platforms (debug+release), assembles a lean addon
  zip + lean examples zip, runs a drop-in smoke (loads `NcnnRunner` from the packaged zip in a
  clean project), publishes a GitHub Release with the addon-zip sha256 in the notes.
- AssetLib uses the `Custom` download provider → the release-asset zip; binaries never enter git.
- Adds `docs/dev/RELEASING.md`, README Installation section; ticks BACKLOG item 25.

Spec: `docs/superpowers/specs/2026-06-07-release-shipping-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Confirm CI is green on the PR**

Run: `gh pr checks --watch`
Expected: the existing `ci` + `cross-build` matrices pass with the relocated paths. (The `release` workflow does not run on a PR — it's tag-only — so it's exercised only at the first real tag.)

- [ ] **Step 3 (manual, after merge): make the repo public + first tag**

These are owner actions, not CI:
1. GitHub → Settings → General → Danger Zone → **Change visibility → Public**.
2. `git checkout main && git pull && git tag v0.1.0 && git push origin v0.1.0`.
3. Watch `release.yml` complete; open the release; copy the addon-zip sha256.
4. Submit to the Asset Library per `docs/dev/RELEASING.md` (Custom download → addon-zip URL + sha256).

---

## Self-Review notes

- **Spec coverage:** restructure (Task 1-3 ↔ spec §Components.1), release workflow + version guard
  + two zips + sha256 (Task 4 ↔ §2/§6), drop-in smoke (Task 4 Step 1 `package`→smoke ↔ §3),
  AssetLib runbook + public-repo prereq (Task 5 ↔ §4), README/CLAUDE/BACKLOG/`Closes #32`
  (Task 6 ↔ §5). Versioning v0.1.0 enforced by the guard + RELEASING (↔ §6). All spec sections
  map to a task.
- **Binaries-never-in-git** holds: nothing commits `addons/godot_native_rl/bin/` (still
  gitignored, Task 2 Step 3); CI builds fresh and ships via release assets only.
- **Examples zip is lean** (no binaries) per the locked decision; its README points at the addon zip.
```

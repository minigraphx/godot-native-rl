# Release Shipping — Design

**Date:** 2026-06-07
**Status:** Approved (brainstorm), pending implementation plan
**Issue:** Closes #32 ([Backlog 25] Asset Library release / extension packaging)
**Milestone:** v1.0 — Asset Library launch + flagship showcase

## Problem

A game developer who wants to *use* this framework should not need the C++/scons/ncnn
source, three Python venvs, or a build toolchain. They need a drop-in addon with prebuilt
binaries. Today there is no release mechanism at all: no git tags, no GitHub Releases, no
Asset Library entry. The `.gdextension` manifest and the `bin/` binaries live at the repo
root (gitignored), not inside the addon, so the repo is not yet a clean drop-in.

This design defines how we ship releases.

## Decisions (from brainstorm)

| Question | Decision |
|---|---|
| Distribution channels | **Both**; GitHub Releases is the single source of truth, AssetLib is a thin pointer at it |
| Release package shape | **Two zips**: lean addon-only + lean examples-only |
| Where committed binaries live for AssetLib | **Nowhere** — AssetLib points its *download* at the GitHub release-asset zip (custom-URL mode). No dist repo, no binaries in git history |
| Repo visibility | **Public** (prerequisite) — lets AssetLib's browse/issues URL show real source while the download is the prebuilt zip |
| Release trigger | **Push a git tag** `vX.Y.Z` |
| First version | **`v0.1.0`** (SemVer; pre-1.0 ⇒ API/protocol may still break) |
| Repo layout | **Move** `.gdextension` + binary output into `addons/godot_native_rl/` |
| Binaries shipped | **Debug + release, all platforms** (~170 MB) — editor loads the debug host lib; release exports need the release libs, so both are required for a true drop-in |
| Examples zip | **Lean** — `examples/**` + sample model + README; users who want to run it also download the addon zip |

## Why binaries never enter git

Full binary set is ~170 MB (debug+release, all platforms); even release-only is ~85 MB.
The dev repo's `.git` is ~2.7 MB of pure source. Committing ~170 MB per release into any
branch (they share one `.git`) bloats history permanently — ~1.7 GB after 10 releases, a
>600× blowup punishing every contributor and CI clone. Therefore:

- **GitHub release assets** carry the binaries. They live *outside* git objects; per-file
  limit is 2 GB (our zips are well under); storage/bandwidth is free and uncapped for public
  repos. Zero git bloat.
- **AssetLib** uses custom-download mode: `download_url` → the release-asset addon zip,
  `download_hash` → its sha256. Its browse/issues URLs point at the public repo so moderators
  can inspect source. No dist repo needed (and none kept as fallback — the public source
  removes the only reason moderation would have balked at a custom download URL).

## Verification (2026-06-07)

The load-bearing assumption — AssetLib can point its *download* at a GitHub release-asset zip
without committing binaries — was verified against current sources:

- The AssetLib backend's download-provider enum includes **`Custom`** (alongside GitHub /
  GitLab / Bitbucket / gogs / cgit). In `Custom` mode the "Download Commit" field holds the
  **full download URL** instead of a git commit hash.
- The canonical GDExtension publishing template
  ([`nathanfranke/gdextension`](https://github.com/nathanfranke/gdextension), current through
  Godot 4.3+) ships binary GDExtensions exactly this way: CI builds all platforms into a
  gitignored `bin/`, a **GitHub Release** carries the addon zip as an asset, and the AssetLib
  entry uses **Repository host = `Custom`, Download URL = the release-asset link**. It
  explicitly warns against Actions artifacts (they expire) — use release assets.
- [Issue #63](https://github.com/godotengine/godot-asset-library/issues/63), which *proposed
  banning* custom download URLs to enforce "1 repo == 1 addon", was **closed without
  enforcement** — `Custom` remains live and is the recommended path for binary addons.

⇒ The custom-URL approach is the ecosystem norm, not a workaround. No dist repo is needed,
including as a fallback. (Keeping `bin/` gitignored + CI-built also matches the template's
own convention.)

## Architecture

```
git tag vX.Y.Z  ──push──▶  .github/workflows/release.yml
                               │
            ┌──────────────────┼─────────────────────────┐
            ▼                  ▼                          ▼
   build all platforms   assemble 2 zips           drop-in smoke
   (zig / NDK / macOS    addon-only + examples-     (extract addon zip
    runner; debug +       only; sha256 of addon      into empty project,
    release)              zip                         load NcnnRunner headless)
                               │
                               ▼
                       GitHub Release  (both zips + changelog + sha256 in notes)
                               │
                               ▼ (manual, per release)
                       AssetLib entry: download_url = addon zip, download_hash = sha256
```

## Components

### 1. One-time repo restructure (#32)

- Move `ncnn_runner.gdextension` → `addons/godot_native_rl/ncnn_runner.gdextension`.
- Repoint manifest library paths `res://bin/…` → `res://addons/godot_native_rl/bin/…`
  (all platform lines).
- Repoint `SConstruct` output target:
  `target=os.path.join("bin", …)` → `target=os.path.join("addons", "godot_native_rl", "bin", …)`.
- Move the `bin/` `.gitignore` entry to `addons/godot_native_rl/bin/`.
- Update every other path reference: `.github/workflows/ci.yml` (build job output + artifact),
  `.github/workflows/cross-build.yml` (`path: bin/` upload), `scripts/cross/*.sh`,
  `test/run_tests.sh`, README build commands, and any doc that says `bin/…`.
- Examples reference *classes* (`NcnnRunner`), not binary paths, so scenes are unaffected.
- Verification: `scons …` puts the lib in the new path; `./test/run_tests.sh` stays green;
  Godot still discovers the relocated `.gdextension` (it scans the project tree, so location
  is free — only explicit path references break).

### 2. Release workflow — `.github/workflows/release.yml`

- **Trigger:** `push: { tags: ['v*'] }`.
- **Version guard:** parse `plugin.cfg` `version=`; assert it equals the tag minus `v`; fail
  the job loud on mismatch (prevents shipping a tag that disagrees with the addon metadata).
- **Build matrix:** reuse the existing cross-build strategy —
  - zig: `linux`, `windows`
  - NDK: `android-arm64`, `android-x86_64`
  - macOS runner: `ios` (device + simulator xcframework) **and** macOS host (arm64)
  - each leg builds **both** `template_debug` and `template_release`.
  - Collect all artifacts into a single `addons/godot_native_rl/bin/` tree.
- **Assemble zips** (in a packaging job that downloads all build artifacts):
  - `godot-native-rl-addon-vX.Y.Z.zip` → contains exactly `addons/godot_native_rl/**`
    (GDScript + `.gdextension` + `bin/` debug+release all platforms). Drop-in: unzip at a
    project root.
  - `godot-native-rl-examples-vX.Y.Z.zip` → `examples/**` + a sample model + a short README
    stating "also install the addon zip (drop both into your project)". **No binaries.**
- **sha256:** compute `sha256sum` of the addon zip; capture for the release notes.
- **Publish:** create the GitHub Release for the tag with both zips attached, an auto-generated
  changelog (commits since the previous tag), and the addon-zip sha256 printed in the notes
  (the value pasted into AssetLib).

### 3. Drop-in smoke (release-time verification)

Before publishing (or as a required job the publish depends on): on a host runner
(macOS or Linux), extract the assembled **addon zip** into a fresh empty Godot project and run
`godot --headless --quit` with a tiny script that instantiates `NcnnRunner`. This proves the
packaged zip actually loads the extension on at least the host platform — catching a broken
manifest path or missing host binary before users hit it. (Other platforms remain
compile-only, as today; running them needs real devices/emulators.)

### 4. AssetLib runbook (manual, documented)

- **One-time submission:** category Tools/Addon; Godot version 4.5; browse URL + issues URL →
  the public dev repo; **download mode = custom**, `download_url` → the GitHub release-asset
  addon zip, `download_hash` → its sha256.
- **Per release:** edit the AssetLib entry — new version string, new `download_url`, new
  `download_hash`. (AssetLib has no public write API; this is a short web-form edit and is the
  one unavoidable manual step.)
- Documented step-by-step in `docs/dev/RELEASING.md`.

### 5. Docs & housekeeping

- New **`docs/dev/RELEASING.md`** — the runbook: bump `plugin.cfg` version → tag `vX.Y.Z` →
  push → what CI does → grab the sha256 from the release notes → update the AssetLib entry.
- **README** — an **Installation** section distinct from build-from-source: (a) Asset Library
  (in-editor), (b) manual zip download from Releases. Note the examples zip needs the addon zip
  too.
- **CLAUDE.md** — add release commands to the key-commands list.
- **`docs/BACKLOG.md`** — tick item 25.
- The closing PR does `Closes #32`.
- **Make the repo public** — manual GitHub setting; a release prerequisite, noted in
  `RELEASING.md`.

### 6. Versioning

SemVer. The git tag is the source of truth; CI enforces `plugin.cfg` agreement. First release
**`v0.1.0`** (`plugin.cfg` already declares `0.1.0`); pre-1.0 signals the API/wire protocol may
still break. Bump to `v1.0.0` for the AssetLib-launch milestone when the surface is stable.

## Out of scope (YAGNI)

- A separate distribution repo (custom-URL + public source removes the need).
- A "downloader" plugin that fetches binaries on first run.
- Per-platform thinned zips (one fat addon zip is simpler; AssetLib installs all platforms
  regardless).
- Automating the AssetLib edit (no public write API; manual edit is acceptable per release).
- Signing/notarization of the native binaries (revisit if platform gatekeepers demand it).

## Testing strategy

- Restructure: `./test/run_tests.sh` green after the move (proves paths repointed correctly).
- Release workflow: the drop-in smoke job (§3) is the load-bearing test — the packaged zip
  loads `NcnnRunner` headless in a clean project.
- Cross-builds: existing compile-only guards continue to prove non-host targets link.
- Version guard: a tag/`plugin.cfg` mismatch fails the workflow (self-testing).
```

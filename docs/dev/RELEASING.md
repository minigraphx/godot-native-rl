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

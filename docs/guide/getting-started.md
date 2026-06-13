# Getting Started

For **game developers** who downloaded this package and want to run an example or train a policy.

## 1. Prerequisites
- **Godot 4.5+**.
- The **`NcnnRunner` GDExtension** for your platform.

## 2. Get the extension
**Prebuilt (recommended, coming):** download the extension for your platform from the project
**Releases** and drop `bin/` + `ncnn_runner.gdextension` into the project root. *(Prebuilt binaries
are not published yet — use the build path below until they are.)*

**Build from source (works today):** follow
[docs/dev/building.md](../dev/building.md). You'll clone `godot-cpp`, build ncnn as a static lib,
and run SCons. This produces `bin/<platform>/...`.

## 3. Enable the plugin

Open the project in the Godot editor, then go to **Project → Project Settings → Plugins** and enable **Godot Native RL**. This is a one-time step per clone. Headless training does not require this — the plugin only affects editor tooling.

> **macOS — clear the download quarantine first.** Prebuilt binaries (from a Release zip or AssetLib `Custom` download) aren't Apple-notarized, so Gatekeeper blocks the `.dylib` and the `NcnnRunner` extension fails to load (examples fall back to "no inference"). Run once after unzipping: `xattr -dr com.apple.quarantine addons/godot_native_rl/bin` (or right-click each `.dylib` → Open). Building from source avoids this.

## 4. Next
- Run a shipped example: [running-examples.md](running-examples.md)
- Train your own AI: [training.md](training.md)
- Build an agent in your own scene: [building-your-agent.md](building-your-agent.md)

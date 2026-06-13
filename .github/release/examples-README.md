# Godot Native RL — Examples

Runnable example scenes (chase, rover, locomotion, visual CNN, cooperative MARL, …) plus sample
trained models. This zip ships a ready-to-run **`project.godot`** — you just need the native addon.

## Run it (no manual file creation)

1. Download `godot-native-rl-addon-<version>.zip` from the **same release** and extract it into
   **this same folder**, so `addons/` sits next to this `project.godot` and `examples/`:
   ```
   <folder>/
     project.godot             ← shipped here
     addons/godot_native_rl/   ← from the addon zip
     examples/  models/
   ```
2. **macOS only** — the prebuilt native library isn't Apple-notarized, so Gatekeeper quarantines it
   on download. Clear the quarantine once, or the examples silently fall back to "no inference":
   ```
   xattr -dr com.apple.quarantine addons/godot_native_rl/bin
   ```
   (or right-click each `.dylib` → Open once.)
3. Open the folder as a project in **Godot 4.5+** and press **F5**. You land on a **demo launcher** —
   pick a demo and click it. Press **Esc** to return to the menu.

## Which scene do I run?

You don't need to know — the launcher lists the runnable play scenes for you. If you browse the files
directly, the convention is: a **bare-named** scene (e.g. `chase_the_target.tscn`) or a `*_track.tscn`
/ `*_race.tscn` is a play scene; a `*_train*.tscn` **hangs** waiting for a Python trainer; a
`*_world.tscn` is a sub-scene building block. When in doubt, run `examples/launcher.tscn`.

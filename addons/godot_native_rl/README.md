# Godot Native RL

Native [ncnn](https://github.com/Tencent/ncnn) inference for Godot 4.5+. Train with
`godot_rl_agents`, deploy a statically-linked C++ GDExtension (no .NET, no Python runtime) on
desktop, mobile, console, and **web**. This package bundles the GDScript library and prebuilt
`NcnnRunner` binaries for every platform.

## Quick start

1. **Enable** the plugin: Project → Project Settings → Plugins → *Godot Native RL*.
   (Also auto-packs your model files into exports — see below.)
2. Add a trained ncnn model (`*.ncnn.param` + `*.ncnn.bin`) to your project.
3. Drive an agent with `NcnnAIController2D` / `NcnnAIController3D` (`control_mode = NCNN_INFERENCE`,
   plus the `model_param_path` / `model_bin_path` exports), or use `NcnnRunner` directly:

```gdscript
var runner := NcnnRunner.new()
runner.input_blob_name = "in0"
runner.output_blob_name = "out0"
# Load from bytes so it works inside an exported .pck (incl. web), not just the editor.
runner.load_model_from_buffers(
	FileAccess.get_file_as_bytes(param_path),
	FileAccess.get_file_as_bytes(bin_path))
var action := runner.run_discrete_action(PackedFloat32Array(obs))
```

## Exporting your game

With the plugin **enabled**, your `*.ncnn.param` / `*.ncnn.bin` are packed into exports
automatically. (Godot otherwise skips these raw data files and the game fails at runtime with
`cannot read model files` — on every platform.)

**Web:** in the Web export preset set **Extension Support: ON** and **Thread Support: OFF**. The
bundled WASM binary is single-threaded, so the game needs **no COOP/COEP headers** — it runs on
itch.io / GitHub Pages with no server configuration.

## Full documentation

Converting trained models, training backends, INT8 quantization, sensors, and the API reference:
<https://github.com/minigraphx/godot-native-rl>

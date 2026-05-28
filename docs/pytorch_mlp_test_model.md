# PyTorch MLP Test Model

This guide creates a tiny MLP and exports it to ncnn (`.param` + `.bin`) so you can test `NcnnRunner` in Godot without downloading a large model.

## 1) Prepare Python environment

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch pnnx
```

## 2) Export a tiny model

Run:

```bash
python scripts/export_test_mlp.py
```

Default output files:

- `models/test_mlp.ncnn.param`
- `models/test_mlp.ncnn.bin`

The script also prints suggested blob names. Use those names in your Godot helper:

- `input_blob_name`
- `output_blob_name`

## 3) Configure in Godot

In your helper node/script:

- `model_param_path` -> `res://models/test_mlp.ncnn.param`
- `model_bin_path` -> `res://models/test_mlp.ncnn.bin`
- `input_blob_name`/`output_blob_name` -> values printed by the script

## 4) Run a sanity call

Call inference with 8 input floats (default `input_dim=8`):

```gdscript
var out = _native_runner.run_inference(PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
print("Inference output: ", out)
```

Expected behavior:

- `load_model(...)` returns `true`
- output has 2 floats by default (`output_dim=2`)

## Optional: customize dimensions

```bash
python scripts/export_test_mlp.py \
  --name policy_mlp \
  --input-dim 32 \
  --hidden-dim 64 \
  --output-dim 6
```

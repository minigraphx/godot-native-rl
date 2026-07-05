@tool
extends EditorExportPlugin
# Force-packs the project's ncnn model files into every game export.
#
# ncnn `.ncnn.param` / `.ncnn.bin` are raw data files referenced by string path (not as Resource
# dependencies), so Godot's exporter skips them and the controllers fail at runtime with "cannot
# read model files" in an exported game — on EVERY platform, web included. This plugin adds them
# back, so a game developer needs no manual `include_filter`. Registered by `plugin.gd` while the
# addon is enabled.

const ModelFileScan := preload("res://addons/godot_native_rl/export/model_file_scan.gd")

func _get_name() -> String:
	return "GodotNativeRLModelPacker"

func _export_begin(features: PackedStringArray, is_debug: bool, path: String, flags: int) -> void:
	# Skip roots come from the ProjectSetting (godot_native_rl/export/skip_roots) or the
	# addon-safe default (res://test only — see #337: res://models is real deploy content in
	# downstream projects). The prune must be VISIBLE: add_file bypasses preset filters, so a
	# silent skip looks like the "cannot read model files" bug this plugin exists to prevent.
	var skip_roots := ModelFileScan.effective_skip_roots()
	for root in skip_roots:
		for pruned in ModelFileScan.find_model_files(root, []):
			push_warning("Godot Native RL: NOT packing '%s' (under skip root '%s' — see the %s project setting)."
				% [pruned, root, ModelFileScan.SKIP_ROOTS_SETTING])
	for file_path in ModelFileScan.find_model_files("res://", skip_roots):
		var bytes := FileAccess.get_file_as_bytes(file_path)
		if bytes.is_empty():
			push_warning("Godot Native RL: skipping unreadable model file '%s' during export." % file_path)
			continue
		add_file(file_path, bytes, false)

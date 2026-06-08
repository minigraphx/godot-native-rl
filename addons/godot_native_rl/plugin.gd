@tool
extends EditorPlugin

# Toggleable addon for the Asset Library. The GDExtension (NcnnRunner) and all class_names
# auto-register independently of this plugin — but enabling it (a) surfaces a clear error if the
# native binary isn't loaded for this platform (a fresh download has no prebuilt binary), and
# (b) registers an EditorExportPlugin that auto-packs your ncnn model files into game exports
# (without it, exported games crash with "cannot read model files" — see export/).
const RuntimeCheck = preload("res://addons/godot_native_rl/plugin_runtime_check.gd")
const ModelExportPlugin = preload("res://addons/godot_native_rl/export/model_export_plugin.gd")

var _export_plugin: EditorExportPlugin = null

func _enter_tree() -> void:
	var msg := RuntimeCheck.extension_error_message(ClassDB.class_exists("NcnnRunner"))
	if msg != "":
		push_error(msg)
	_export_plugin = ModelExportPlugin.new()
	add_export_plugin(_export_plugin)

func _exit_tree() -> void:
	if _export_plugin != null:
		remove_export_plugin(_export_plugin)
		_export_plugin = null

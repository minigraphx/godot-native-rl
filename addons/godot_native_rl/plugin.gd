@tool
extends EditorPlugin

# Toggleable addon for the Asset Library. The GDExtension (NcnnRunner) and all class_names
# auto-register independently of this plugin — but enabling it (a) surfaces a clear error if the
# native binary isn't loaded for this platform (a fresh download has no prebuilt binary), and
# (b) registers an EditorExportPlugin that auto-packs your ncnn model files into game exports
# (without it, exported games crash with "cannot read model files" — see export/), and
# (c) installs the NcnnAIController script templates into res://script_templates/ (copy-if-
# missing — your edited copies are never touched; see script_template_installer.gd).
const RuntimeCheck = preload("res://addons/godot_native_rl/plugin_runtime_check.gd")
const ModelExportPlugin = preload("res://addons/godot_native_rl/export/model_export_plugin.gd")
const TemplateInstaller = preload("res://addons/godot_native_rl/script_template_installer.gd")

var _export_plugin: EditorExportPlugin = null

func _enter_tree() -> void:
	var msg := RuntimeCheck.extension_error_message(ClassDB.class_exists("NcnnRunner"))
	if msg != "":
		push_error(msg)
	_export_plugin = ModelExportPlugin.new()
	add_export_plugin(_export_plugin)
	var plan := TemplateInstaller.build_plan(
		TemplateInstaller.TEMPLATE_SOURCES,
		TemplateInstaller.DEST_ROOT,
		func(p: String) -> bool: return FileAccess.file_exists(p)
	)
	for err in TemplateInstaller.execute_plan(plan):
		push_error("Godot Native RL: script template install failed: %s" % err)

func _exit_tree() -> void:
	if _export_plugin != null:
		remove_export_plugin(_export_plugin)
		_export_plugin = null

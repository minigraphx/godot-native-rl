@tool
extends EditorPlugin

# Toggleable addon for the Asset Library. The GDExtension (NcnnRunner) and all class_names
# auto-register independently of this plugin, so enabling it is optional for using the library —
# but on enable we surface a clear error if the native binary isn't loaded for this platform,
# since the addon is useless without it (a fresh download has no prebuilt binary).
const RuntimeCheck = preload("res://addons/godot_native_rl/plugin_runtime_check.gd")

func _enter_tree() -> void:
	var msg := RuntimeCheck.extension_error_message(ClassDB.class_exists("NcnnRunner"))
	if msg != "":
		push_error(msg)

func _exit_tree() -> void:
	pass

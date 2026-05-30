@tool
extends EditorPlugin

# Marker EditorPlugin so this is a recognized, toggleable addon for the Asset Library.
# The GDExtension (NcnnRunner) and all class_names auto-register independently of this
# plugin being enabled, so enabling it is optional for using the library.
func _enter_tree() -> void:
	pass

func _exit_tree() -> void:
	pass

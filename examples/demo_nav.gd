extends Node
# Autoload (registered in the shipped examples project.godot, #226): pressing Escape from any demo
# scene returns to the launcher menu, so a user never gets stuck inside a demo with no way back.
# Inert unless ui_cancel fires; a no-op if the launcher scene is missing.

const LAUNCHER := "res://examples/launcher.tscn"

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		var tree := get_tree()
		# Don't reload if we're already on the launcher.
		var current := tree.current_scene
		if current != null and current.scene_file_path == LAUNCHER:
			return
		if ResourceLoader.exists(LAUNCHER):
			tree.change_scene_to_file(LAUNCHER)

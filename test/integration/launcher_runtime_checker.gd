extends Node
# Regression for #239: a scene loaded via change_scene_to_file (what the demo launcher does) must
# still initialize NcnnSync. Lives on the SceneTree root so it survives the scene swap. Each frame
# it inspects the current scene's Sync; PASS as soon as `initialized` is true, FAIL on timeout
# (the bug: NcnnSync awaited the one-shot root `ready` signal, which never re-fires post-boot, so
# inference never started and the scene stayed frozen).

@export var timeout_frames := 300

var _frames := 0

func _process(_delta: float) -> void:
	_frames += 1
	var cur := get_tree().current_scene
	if cur != null:
		var sync := _find_sync(cur)
		if sync != null and sync.initialized:
			print("LAUNCHER RUNTIME PASSED (Sync initialized %d frames after change_scene)" % _frames)
			get_tree().quit(0)
			return
	if _frames >= timeout_frames:
		printerr("LAUNCHER RUNTIME FAILED: Sync never initialized %d frames after change_scene_to_file" % _frames)
		get_tree().quit(1)

# The Sync node: exposes `initialized` and the `_initialize` method (path-based, no class_name).
func _find_sync(node: Node) -> Node:
	for child in node.get_children():
		if child.has_method("_initialize") and ("initialized" in child):
			return child
	return null

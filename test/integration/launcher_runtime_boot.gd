extends Node
# Boot scene for the #239 regression: reproduces the launcher's flow — after the tree has fully
# booted (root `ready` already fired), load a play scene via change_scene_to_file. A checker parented
# to the root survives the swap and asserts the loaded scene's Sync initializes. (See
# launcher_runtime_checker.gd.)

const CHECKER := "res://test/integration/launcher_runtime_checker.gd"
const DEMO := "res://examples/chase_the_target/chase_the_target.tscn"

func _ready() -> void:
	var checker: Node = load(CHECKER).new()
	checker.name = "LauncherRuntimeChecker"
	get_tree().root.add_child.call_deferred(checker)  # deferred: root is busy setting up at boot
	await get_tree().root.ready
	await get_tree().process_frame  # let the deferred add_child land before swapping scenes
	get_tree().change_scene_to_file(DEMO)

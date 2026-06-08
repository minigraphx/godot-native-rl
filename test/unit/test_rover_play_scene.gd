extends SceneTree

const Harness = preload("res://test/harness.gd")
const PLAY_SCENE := "res://examples/rover_3d/rover_3d.tscn"

func _initialize() -> void:
	var h := Harness.new()
	var packed := load(PLAY_SCENE) as PackedScene
	h.assert_true(packed != null, "rover play scene loads")
	if packed == null:
		h.finish(self)
		return

	var scene := packed.instantiate()
	root.add_child(scene)

	var agent := scene.get_node_or_null("RoverAgent")
	h.assert_true(agent != null, "play scene has rover agent")
	if agent != null:
		h.assert_eq(agent.control_mode, 3, "rover agent uses NCNN_INFERENCE")
		h.assert_eq(agent.model_param_path,
			"res://examples/rover_3d/models/rover_policy.ncnn.param",
			"rover param model configured")
		h.assert_eq(agent.model_bin_path,
			"res://examples/rover_3d/models/rover_policy.ncnn.bin",
			"rover bin model configured")

	var sync := scene.get_node_or_null("Sync")
	h.assert_true(sync != null, "play scene has inference sync")
	if sync != null:
		h.assert_eq(sync.control_mode, 2, "sync uses NCNN_INFERENCE")

	h.assert_true(scene.get_node_or_null("Camera3D") is Camera3D, "play scene has camera")
	h.assert_true(scene.get_node_or_null("DirectionalLight3D") is DirectionalLight3D,
		"play scene has light")
	h.assert_true(scene.get_node_or_null("Ground") is MeshInstance3D, "play scene has ground mesh")
	h.assert_true(scene.get_node_or_null("AgentBody/RoverMesh") is MeshInstance3D,
		"play scene has rover mesh")
	h.assert_true(scene.get_node_or_null("Goal/GoalMesh") is MeshInstance3D,
		"play scene has goal mesh")

	scene.free()
	h.finish(self)

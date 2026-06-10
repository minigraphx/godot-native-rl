extends SceneTree

# Smoke tests for the drop-in sensor scenes (#112): each .tscn must instantiate headlessly,
# carry the right script (referenced by res:// path), and be usable with its defaults.

const Harness = preload("res://test/harness.gd")

const SCENES_DIR := "res://addons/godot_native_rl/sensors/scenes"

func _initialize() -> void:
	var h := Harness.new()

	# --- RaycastSensor2D.tscn ---
	var packed_2d: PackedScene = load("%s/RaycastSensor2D.tscn" % SCENES_DIR)
	h.assert_true(packed_2d != null, "RaycastSensor2D.tscn loads")
	var ray2d = packed_2d.instantiate()
	h.assert_true(ray2d is Node2D, "RaycastSensor2D root is Node2D")
	h.assert_true(ray2d.has_method("get_observation"), "RaycastSensor2D has sensor script")
	h.assert_true(ray2d.obs_size() > 0, "RaycastSensor2D obs_size > 0 with defaults")
	ray2d.free()

	# --- RaycastSensor3D.tscn ---
	var packed_3d: PackedScene = load("%s/RaycastSensor3D.tscn" % SCENES_DIR)
	h.assert_true(packed_3d != null, "RaycastSensor3D.tscn loads")
	var ray3d = packed_3d.instantiate()
	h.assert_true(ray3d is Node3D, "RaycastSensor3D root is Node3D")
	h.assert_true(ray3d.has_method("get_observation"), "RaycastSensor3D has sensor script")
	h.assert_true(ray3d.obs_size() > 0, "RaycastSensor3D obs_size > 0 with defaults")
	ray3d.free()

	# --- CameraSensor2D.tscn: SubViewport + Camera2D pre-wired ---
	var packed_cam2d: PackedScene = load("%s/CameraSensor2D.tscn" % SCENES_DIR)
	h.assert_true(packed_cam2d != null, "CameraSensor2D.tscn loads")
	var cam2d = packed_cam2d.instantiate()
	h.assert_true(cam2d.viewport is SubViewport, "CameraSensor2D viewport export pre-wired")
	h.assert_eq(cam2d.viewport.size, Vector2i(36, 36), "CameraSensor2D SubViewport is 36x36")
	h.assert_eq(cam2d.viewport.render_target_update_mode, SubViewport.UPDATE_ALWAYS,
		"CameraSensor2D SubViewport renders every frame")
	h.assert_true(cam2d.viewport.get_node_or_null("Camera2D") is Camera2D,
		"CameraSensor2D has a Camera2D inside the SubViewport")
	h.assert_true(cam2d.is_key_valid(cam2d.observation_key),
		"CameraSensor2D observation_key valid (contains \"2d\")")
	cam2d.free()

	# --- CameraSensor3D.tscn: SubViewport + Camera3D pre-wired ---
	var packed_cam3d: PackedScene = load("%s/CameraSensor3D.tscn" % SCENES_DIR)
	h.assert_true(packed_cam3d != null, "CameraSensor3D.tscn loads")
	var cam3d = packed_cam3d.instantiate()
	h.assert_true(cam3d.viewport is SubViewport, "CameraSensor3D viewport export pre-wired")
	h.assert_true(cam3d.viewport.get_node_or_null("Camera3D") is Camera3D,
		"CameraSensor3D has a Camera3D inside the SubViewport")
	h.assert_true(cam3d.is_key_valid(cam3d.observation_key),
		"CameraSensor3D observation_key valid (godot_rl routes image obs on the \"2d\" substring)")
	cam3d.free()

	h.finish(self)

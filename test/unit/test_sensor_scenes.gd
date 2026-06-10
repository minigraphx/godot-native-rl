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

	h.finish(self)

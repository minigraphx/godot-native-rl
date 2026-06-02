extends SceneTree

const Harness = preload("res://test/harness.gd")
const GridSensor3D = preload("res://addons/godot_native_rl/sensors/grid_sensor_3d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = GridSensor3D.new()
	s.grid_size_x = 3
	s.grid_size_z = 3
	s.cell_width = 2.0
	s.cell_height = 2.0
	s.detection_mask = 1

	# obs_size reflects grid_x * grid_z * layers
	h.assert_eq(s.obs_size(), 9, "obs_size == 3*3*1")

	# Empty-overlap stub -> zeros, length == obs_size
	var empty_fn := func(_c: Vector3, _sz: Vector3) -> Array:
		return []
	s.set_overlap_fn_for_test(empty_fn)
	var obs_empty: Array = s.get_observation()
	h.assert_eq(obs_empty.size(), 9, "obs length == obs_size")
	var all_zero := true
	for v in obs_empty:
		if absf(v) > 1e-9:
			all_zero = false
	h.assert_true(all_zero, "empty overlap -> zeros")

	# Hit only at the cell nearest origin -> count 1 at center index 4
	var centers: Array = []
	var hit_center_fn := func(c: Vector3, _sz: Vector3) -> Array:
		centers.append(c)
		if c.length() < 1e-5:
			return [0b1]
		return []
	s.set_overlap_fn_for_test(hit_center_fn)
	var obs_hit: Array = s.get_observation()
	h.assert_eq(obs_hit[4], 1.0, "hit at center cell -> count 1 at index 4")
	h.assert_eq(centers.size(), 9, "queried 9 cells")

	# Cells live on the X/Z plane (y == 0 for all centers)
	var all_y_zero := true
	for c in centers:
		if absf(c.y) > 1e-5:
			all_y_zero = false
	h.assert_true(all_y_zero, "cells on X/Z plane (y==0)")

	# Degenerate grid -> empty obs + obs_size 0
	s.grid_size_x = 0
	h.assert_eq(s.get_observation().size(), 0, "grid 0 -> empty obs")
	h.assert_eq(s.obs_size(), 0, "grid 0 -> obs_size 0")

	s.free()
	h.finish(self)

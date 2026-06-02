extends SceneTree

const Harness = preload("res://test/harness.gd")
const GridSensor2D = preload("res://addons/godot_native_rl/sensors/grid_sensor_2d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = GridSensor2D.new()
	s.grid_size_x = 3
	s.grid_size_y = 3
	s.cell_width = 10.0
	s.cell_height = 10.0
	s.detection_mask = 1

	# obs_size reflects grid * layers
	h.assert_eq(s.obs_size(), 9, "obs_size == 3*3*1")

	# Empty-overlap stub -> all zeros, length == obs_size
	var empty_fn := func(_c: Vector2, _sz: Vector2) -> Array:
		return []
	s.set_overlap_fn_for_test(empty_fn)
	var obs_empty: Array = s.get_observation()
	h.assert_eq(obs_empty.size(), 9, "obs length == obs_size")
	var all_zero := true
	for v in obs_empty:
		if absf(v) > 1e-9:
			all_zero = false
	h.assert_true(all_zero, "empty overlap -> zeros")

	# Stub that reports a layer-1 hit only for the cell nearest origin -> count 1 there
	var centers: Array = []
	var hit_center_fn := func(c: Vector2, _sz: Vector2) -> Array:
		centers.append(c)
		if c.length() < 1e-5:
			return [0b1]
		return []
	s.set_overlap_fn_for_test(hit_center_fn)
	var obs_hit: Array = s.get_observation()
	# center cell is i1,j1 -> index 4
	h.assert_eq(obs_hit[4], 1.0, "hit at center cell -> count 1 at index 4")
	h.assert_eq(centers.size(), 9, "queried 9 cells")

	# Cell centers translate with node position
	centers.clear()
	s.position = Vector2(100.0, 0.0)
	s.set_overlap_fn_for_test(hit_center_fn)
	s.get_observation()
	var found_translated := false
	for c in centers:
		if (c - Vector2(100.0, 0.0)).length() < 1e-5:
			found_translated = true
	h.assert_true(found_translated, "center cell shifted to node position")

	# Multi-layer mask through the wrapper: 2 layers -> obs_size doubles, an object on
	# both mapped layers increments both slots of its cell (integration of n_layers > 1).
	s.position = Vector2.ZERO
	s.detection_mask = 0b101
	h.assert_eq(s.obs_size(), 18, "2 layers -> obs_size 3*3*2")
	var multi_fn := func(c: Vector2, _sz: Vector2) -> Array:
		if c.length() < 1e-5:
			return [0b101]
		return []
	s.set_overlap_fn_for_test(multi_fn)
	var obs_multi: Array = s.get_observation()
	h.assert_eq(obs_multi.size(), 18, "multi-layer obs length == obs_size")
	# center cell (i1,j1) base index = (1*3+1)*2 = 8; both layer slots set
	h.assert_eq(obs_multi[8], 1.0, "center cell layer slot 0 -> 1")
	h.assert_eq(obs_multi[9], 1.0, "center cell layer slot 1 -> 1")
	s.detection_mask = 1

	# Degenerate grid -> empty obs + obs_size 0
	s.grid_size_x = 0
	h.assert_eq(s.get_observation().size(), 0, "grid 0 -> empty obs")
	h.assert_eq(s.obs_size(), 0, "grid 0 -> obs_size 0")

	s.free()
	h.finish(self)

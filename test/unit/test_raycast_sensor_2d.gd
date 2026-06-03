extends SceneTree

const Harness = preload("res://test/harness.gd")
const RaycastSensor2D = preload("res://addons/godot_native_rl/sensors/raycast_sensor_2d.gd")
const RaycastMath = preload("res://addons/godot_native_rl/sensors/raycast_math.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RaycastSensor2D.new()
	s.n_rays = 4
	s.ray_length = 100.0
	s.cone_degrees = 90.0

	# obs_size reflects n_rays
	h.assert_eq(s.obs_size(), 4, "obs_size == n_rays")

	# All-miss stub -> all zeros, length == n_rays
	var miss_fn := func(_o: Vector2, _d: Vector2) -> float:
		return -1.0
	s.set_cast_fn_for_test(miss_fn)
	var obs_miss: Array = s.get_observation()
	h.assert_eq(obs_miss.size(), 4, "obs length == n_rays")
	var all_zero := true
	for v in obs_miss:
		if absf(v) > 1e-6:
			all_zero = false
	h.assert_true(all_zero, "all-miss -> zeros")

	# Hit-at-origin stub (distance 0) -> all ones
	var hit_fn := func(_o: Vector2, _d: Vector2) -> float:
		return 0.0
	s.set_cast_fn_for_test(hit_fn)
	var obs_hit: Array = s.get_observation()
	var all_one := true
	for v in obs_hit:
		if absf(v - 1.0) > 1e-6:
			all_one = false
	h.assert_true(all_one, "hit at origin -> ones")

	# Half-distance stub -> 0.5 closeness
	var half_fn := func(_o: Vector2, _d: Vector2) -> float:
		return 50.0
	s.set_cast_fn_for_test(half_fn)
	var obs_half: Array = s.get_observation()
	h.assert_true(absf(obs_half[0] - 0.5) < 1e-6, "distance 50 / length 100 -> 0.5")

	# Directions passed to the cast rotate with the node's rotation.
	var recorded := []
	s.rotation = PI / 2.0
	var record_fn := func(_o: Vector2, d: Vector2) -> float:
		recorded.append(d)
		return -1.0
	s.set_cast_fn_for_test(record_fn)
	s.get_observation()
	var expected: Array = RaycastMath.ray_directions_2d(4, 90.0, PI / 2.0)
	var dirs_match := recorded.size() == expected.size()
	for i in range(mini(recorded.size(), expected.size())):
		if (recorded[i] - expected[i]).length() > 1e-5:
			dirs_match = false
	h.assert_true(dirs_match, "cast directions rotate with node rotation (PI/2)")

	# --- class_sensor mode ---
	var cs = RaycastSensor2D.new()
	cs.n_rays = 2
	cs.ray_length = 100.0
	cs.cone_degrees = 90.0
	cs.class_sensor = true
	var cs_classes: Array[int] = [2, 3]    # bit values 2 and 4
	cs.detection_classes = cs_classes
	cs.include_other = true
	cs.include_distance = true
	# per ray = 2 classes + other + distance = 4; 2 rays -> 8
	h.assert_eq(cs.obs_size(), 8, "class obs_size = n_rays * (n_classes + other + distance)")

	# ray 0 hits layer 3 (bit value 4) at half distance; ray 1 misses
	var cs_calls := {"n": 0}
	var cs_class_fn := func(_o: Vector2, _d: Vector2) -> Dictionary:
		cs_calls["n"] += 1
		if cs_calls["n"] == 1:
			return {"distance": 50.0, "layer": 4}
		return {"distance": -1.0, "layer": 0}
	cs.set_class_cast_fn_for_test(cs_class_fn)
	var cs_obs: Array = cs.get_observation()
	h.assert_eq(cs_obs.size(), 8, "class obs length == obs_size")
	h.assert_eq(cs_obs, [0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], "class obs: ray0 layer-3 hit, ray1 miss")

	# class_sensor off -> distance-only path is byte-identical to before
	cs.class_sensor = false
	var cs_half_fn := func(_o: Vector2, _d: Vector2) -> float:
		return 50.0
	cs.set_cast_fn_for_test(cs_half_fn)
	h.assert_eq(cs.get_observation(), [0.5, 0.5], "class_sensor=false -> distance-only path unchanged")
	cs.free()

	# n_rays < 1 -> empty obs + obs_size 0
	s.n_rays = 0
	h.assert_eq(s.get_observation().size(), 0, "n_rays 0 -> empty obs")
	h.assert_eq(s.obs_size(), 0, "n_rays 0 -> obs_size 0")

	s.free()
	h.finish(self)

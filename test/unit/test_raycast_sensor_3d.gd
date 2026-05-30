extends SceneTree

const Harness = preload("res://test/harness.gd")
const RaycastSensor3D = preload("res://sensors/raycast_sensor_3d.gd")
const RaycastMath = preload("res://sensors/raycast_math.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RaycastSensor3D.new()
	s.n_rays_width = 4
	s.n_rays_height = 2
	s.ray_length = 20.0
	s.horizontal_fov = 90.0
	s.vertical_fov = 45.0

	# obs_size == n_w * n_h
	h.assert_eq(s.obs_size(), 8, "obs_size == n_w * n_h")

	# All-miss stub -> zeros, length == obs_size
	var miss_fn := func(_o: Vector3, _d: Vector3) -> float:
		return -1.0
	s.set_cast_fn_for_test(miss_fn)
	var obs_miss: Array = s.get_observation()
	h.assert_eq(obs_miss.size(), 8, "obs length == obs_size")
	var all_zero := true
	for v in obs_miss:
		if absf(v) > 1e-6:
			all_zero = false
	h.assert_true(all_zero, "all-miss -> zeros")

	# Hit-at-origin stub -> ones
	var hit_fn := func(_o: Vector3, _d: Vector3) -> float:
		return 0.0
	s.set_cast_fn_for_test(hit_fn)
	var obs_hit: Array = s.get_observation()
	var all_one := true
	for v in obs_hit:
		if absf(v - 1.0) > 1e-6:
			all_one = false
	h.assert_true(all_one, "hit at origin -> ones")

	# Half-distance stub -> 0.5
	var half_fn := func(_o: Vector3, _d: Vector3) -> float:
		return 10.0
	s.set_cast_fn_for_test(half_fn)
	var obs_half: Array = s.get_observation()
	h.assert_true(absf(obs_half[0] - 0.5) < 1e-6, "distance 10 / length 20 -> 0.5")

	# Directions rotate with the node's transform basis.
	var recorded := []
	s.rotation = Vector3(0.0, PI / 2.0, 0.0)
	var record_fn := func(_o: Vector3, d: Vector3) -> float:
		recorded.append(d)
		return -1.0
	s.set_cast_fn_for_test(record_fn)
	s.get_observation()
	var local_dirs: Array = RaycastMath.ray_directions_3d(4, 2, 90.0, 45.0)
	var node_basis: Basis = s.transform.basis
	var dirs_match := recorded.size() == local_dirs.size()
	for i in range(mini(recorded.size(), local_dirs.size())):
		var expected_dir: Vector3 = node_basis * local_dirs[i]
		if (recorded[i] - expected_dir).length() > 1e-5:
			dirs_match = false
	h.assert_true(dirs_match, "cast directions rotate with node basis (yaw PI/2)")

	# Degenerate counts -> empty
	s.n_rays_width = 0
	h.assert_eq(s.get_observation().size(), 0, "n_rays_width 0 -> empty obs")
	h.assert_eq(s.obs_size(), 0, "n_rays_width 0 -> obs_size 0")

	s.free()
	h.finish(self)

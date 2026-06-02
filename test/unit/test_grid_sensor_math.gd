extends SceneTree

const Harness = preload("res://test/harness.gd")
const GridSensorMath = preload("res://addons/godot_native_rl/sensors/grid_sensor_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# collision_mapping: each set bit -> sequential slot
	var m1: Dictionary = GridSensorMath.collision_mapping(0b101)
	h.assert_eq(m1.size(), 2, "mask 0b101 -> 2 layers")
	h.assert_eq(m1[0], 0, "bit 0 -> slot 0")
	h.assert_eq(m1[2], 1, "bit 2 -> slot 1")
	h.assert_eq(GridSensorMath.collision_mapping(0).size(), 0, "mask 0 -> empty mapping")

	# n_layers
	h.assert_eq(GridSensorMath.n_layers(0b101), 2, "n_layers(0b101) == 2")
	h.assert_eq(GridSensorMath.n_layers(0), 0, "n_layers(0) == 0")
	h.assert_eq(GridSensorMath.n_layers(1), 1, "n_layers(1) == 1")

	# obs_size = grid_a * grid_b * n_layers
	h.assert_eq(GridSensorMath.obs_size(3, 3, 1), 9, "3x3x1 -> 9")
	h.assert_eq(GridSensorMath.obs_size(3, 3, 0b101), 18, "3x3x2 -> 18")
	h.assert_eq(GridSensorMath.obs_size(0, 3, 1), 0, "zero grid -> 0")
	h.assert_eq(GridSensorMath.obs_size(3, 3, 0), 0, "zero mask -> 0")

	# obs_index: (i*grid_b*n) + (j*n) + slot
	h.assert_eq(GridSensorMath.obs_index(0, 0, 0, 3, 1), 0, "cell 0,0 slot 0")
	h.assert_eq(GridSensorMath.obs_index(1, 0, 0, 3, 1), 3, "cell 1,0 -> 3")
	h.assert_eq(GridSensorMath.obs_index(0, 2, 0, 3, 1), 2, "cell 0,2 -> 2")
	h.assert_eq(GridSensorMath.obs_index(1, 2, 1, 3, 2), 11, "cell 1,2 slot1 n2 -> 11")

	# cell_offsets: odd grid is symmetric about origin (shift = -(grid/2)*step)
	var off3: Array = GridSensorMath.cell_offsets(3, 3, 10.0, 10.0)
	h.assert_eq(off3.size(), 9, "3x3 -> 9 offsets")
	# i outer, j inner: index 0 = (i0,j0); shift = (-10,-10) since 3/2==1
	h.assert_true((off3[0] - Vector2(-10.0, -10.0)).length() < 1e-5, "cell 0,0 at (-10,-10)")
	# center cell (i1,j1) at index 1*3+1 = 4 sits on origin
	h.assert_true(off3[4].length() < 1e-5, "center cell at origin")
	# even grid offset by half-cell: 2/2==1 -> shift -(1)*10 = -10
	var off2: Array = GridSensorMath.cell_offsets(2, 2, 10.0, 10.0)
	h.assert_eq(off2.size(), 4, "2x2 -> 4 offsets")
	h.assert_true((off2[0] - Vector2(-10.0, -10.0)).length() < 1e-5, "even cell 0,0 at (-10,-10)")
	# asymmetric steps
	var offab: Array = GridSensorMath.cell_offsets(1, 2, 5.0, 7.0)
	h.assert_eq(offab.size(), 2, "1x2 -> 2 offsets")

	# build_obs: empty cells -> all zeros
	var empty_cells: Array = [[], [], [], [], [], [], [], [], []]
	var ob0: Array = GridSensorMath.build_obs(empty_cells, 3, 3, 1)
	h.assert_eq(ob0.size(), 9, "build_obs length == obs_size")
	var all_zero := true
	for v in ob0:
		if absf(v) > 1e-9:
			all_zero = false
	h.assert_true(all_zero, "empty -> zeros")

	# build_obs: one object on layer bit 0 in cell index 4 (i1,j1) -> count 1 at obs_index 4
	var cells1: Array = [[], [], [], [], [0b1], [], [], [], []]
	var ob1: Array = GridSensorMath.build_obs(cells1, 3, 3, 1)
	h.assert_eq(ob1[4], 1.0, "single hit -> count 1 at index 4")

	# build_obs: two objects same cell+layer -> count accumulates
	var cells2: Array = [[0b1, 0b1], [], [], [], [], [], [], [], []]
	var ob2: Array = GridSensorMath.build_obs(cells2, 3, 3, 1)
	h.assert_eq(ob2[0], 2.0, "two objects -> count 2")

	# build_obs: object on two mapped layers -> increments both slots
	var cells3: Array = [[0b101], [], [], [], [], [], [], [], []]
	var ob3: Array = GridSensorMath.build_obs(cells3, 3, 3, 0b101)
	h.assert_eq(ob3[0], 1.0, "slot 0 incremented")
	h.assert_eq(ob3[1], 1.0, "slot 1 incremented")

	# build_obs: layer bit outside the mask is ignored
	var cells4: Array = [[0b10], [], [], [], [], [], [], [], []]
	var ob4: Array = GridSensorMath.build_obs(cells4, 3, 3, 0b1)
	var ignored_zero := true
	for v in ob4:
		if absf(v) > 1e-9:
			ignored_zero = false
	h.assert_true(ignored_zero, "out-of-mask layer ignored")

	h.finish(self)

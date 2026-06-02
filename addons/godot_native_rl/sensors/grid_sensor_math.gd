class_name GridSensorMath
extends RefCounted

# Pure, stateless helpers for grid sensors. No physics, no node state — fully
# unit-testable headlessly. Encoding matches godot_rl's GridSensor: one float per
# active detection-layer bit per cell = count of overlapping objects on that layer.
# grid_a/grid_b are the two grid axes (2D: x,y; 3D: x,z). Buffer index = i*grid_b+j.

# Each set bit of detection_mask -> a sequential obs slot, low bit first.
static func collision_mapping(detection_mask: int) -> Dictionary:
	var mapping := {}
	var total := 0
	for i in range(32):
		if (detection_mask & (1 << i)) != 0:
			mapping[i] = total
			total += 1
	return mapping

static func n_layers(detection_mask: int) -> int:
	return collision_mapping(detection_mask).size()

static func obs_size(grid_a: int, grid_b: int, detection_mask: int) -> int:
	return maxi(grid_a, 0) * maxi(grid_b, 0) * n_layers(detection_mask)

# Flat-buffer index for a cell's layer slot (godot_rl formula).
static func obs_index(cell_i: int, cell_j: int, layer_slot: int, grid_b: int, layers: int) -> int:
	return (cell_i * grid_b * layers) + (cell_j * layers) + layer_slot

# Local cell-center offsets (Vector2) relative to the sensor origin. Integer-division
# shift matches godot_rl: odd grids are symmetric about origin; even grids place a cell
# boundary on the origin (shift = -(grid/2)*step, a whole cell for grid==2). Order is
# i outer / j inner -> element index = i*grid_b + j.
static func cell_offsets(grid_a: int, grid_b: int, step_a: float, step_b: float) -> Array:
	var offsets: Array = []
	if grid_a < 1 or grid_b < 1:
		return offsets
	var shift_a := -float(grid_a / 2) * step_a
	var shift_b := -float(grid_b / 2) * step_b
	for i in range(grid_a):
		for j in range(grid_b):
			offsets.append(Vector2(float(i) * step_a + shift_a, float(j) * step_b + shift_b))
	return offsets

# Build the flat float observation buffer from per-cell overlapping collision layers.
# cell_layers: flat Array (len grid_a*grid_b, index i*grid_b+j) of Array[int] of the
# collision_layer values overlapping each cell. Returns a fresh Array of floats.
static func build_obs(cell_layers: Array, grid_a: int, grid_b: int, detection_mask: int) -> Array:
	var mapping := collision_mapping(detection_mask)
	var layers := mapping.size()
	var out: Array = []
	out.resize(maxi(grid_a, 0) * maxi(grid_b, 0) * layers)
	out.fill(0.0)
	if layers == 0 or grid_a < 1 or grid_b < 1:
		return out
	for i in range(grid_a):
		for j in range(grid_b):
			var cell_index := i * grid_b + j
			if cell_index >= cell_layers.size():
				continue
			var layers_here: Array = cell_layers[cell_index]
			for collision_layer in layers_here:
				for bit in mapping:
					if (collision_layer & (1 << bit)) != 0:
						out[obs_index(i, j, mapping[bit], grid_b, layers)] += 1.0
	return out

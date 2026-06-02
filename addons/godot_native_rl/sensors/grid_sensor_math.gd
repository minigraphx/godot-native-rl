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

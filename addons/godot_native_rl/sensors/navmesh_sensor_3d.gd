class_name NavMeshSensor3D
extends "res://addons/godot_native_rl/sensors/i_sensor_3d.gd"

# 3D navigable-path observation toward a target (#20, novel-addons spec §3 A3) — the NavMeshSensor2D
# counterpart over NavigationServer3D. Emits [closeness, dir_x, dir_y, dir_z]: closeness from the
# actual walkable path length, direction toward the next waypoint (egocentric by default).
# No-map / freed target zero-fills; a fully disconnected target yields a *partial* path to the
# closest reachable point (set require_reachable to zero-fill those). map_get_path is a real
# pathfinding query per get_observation(). The query is isolated behind _path_fn for headless tests.

const NavMeshMath = preload("res://addons/godot_native_rl/sensors/navmesh_math.gd")

## Target to navigate toward; freed/invalid → zero-filled.
@export var target: Node3D
## Path-length normalizer for closeness: 1 at the target, →0 at/over this path length.
@export_range(0.01, 20000.0) var max_distance: float = 100.0
## Optimize the path (NavigationServer3D.map_get_path's `optimize`).
@export var optimize_path: bool = true
## Navigation layers mask for the query.
@export_flags_3d_navigation var navigation_layers: int = 1
## Emit the next-waypoint direction in the sensor's local (egocentric) frame. false = world frame.
@export var egocentric: bool = true
## When true, a target map_get_path can only partially reach (disconnected island) zero-fills, so a
## walled-off target reads as "far". Off by default (a partial path otherwise reads as high
## closeness). Tune reachable_tolerance.
@export var require_reachable: bool = false
## Distance (in 3D world units) within which the path's end must land on `target` to count as reached.
@export_range(0.0, 1000.0) var reachable_tolerance: float = 1.0

var _path_fn = null  # Callable(from: Vector3, to: Vector3) -> PackedVector3Array (test seam)

## Inject a path provider so get_observation() runs without a live navigation map (tests).
func set_path_fn_for_test(fn: Callable) -> void:
	_path_fn = fn

func obs_size() -> int:
	return 4

func get_observation() -> Array:
	if not is_instance_valid(target):
		return [0.0, 0.0, 0.0, 0.0]
	var from := global_position if is_inside_tree() else position
	var to := target.global_position if target.is_inside_tree() else target.position
	var path := _query_path(from, to)
	if require_reachable and not NavMeshMath.is_reachable(path, to, reachable_tolerance):
		return [0.0, 0.0, 0.0, 0.0]
	var basis := (global_basis if is_inside_tree() else transform.basis) if egocentric else Basis.IDENTITY
	return NavMeshMath.encode_3d(from, path, max_distance, basis)

func _query_path(from: Vector3, to: Vector3) -> Array:
	if _path_fn != null:
		return Array(_path_fn.call(from, to))
	if not is_inside_tree() or get_world_3d() == null:
		return []  # no map available -> unreachable
	var map := get_world_3d().get_navigation_map()
	return Array(NavigationServer3D.map_get_path(map, from, to, optimize_path, navigation_layers))

class_name NavMeshSensor2D
extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"

# Navigable-path observation toward a target (#20, novel-addons spec §3 A3): a Godot-native sensor
# with no godot_rl/Unity equivalent. Instead of straight-line distance/direction (RelativePosition),
# it queries the NavigationServer for the actual walkable path around obstacles and emits
# [closeness, dir_x, dir_y] — closeness from the *path* length, direction toward the *next waypoint*.
# Around-the-obstacle paths read longer than the straight line, so a wall the agent must detour
# around reads as "far". The direction is egocentric by default (set egocentric=false for world
# frame). No map / freed target zero-fills; note that map_get_path returns a *partial* path to the
# closest reachable point for a fully disconnected target — set require_reachable to zero-fill those.
# map_get_path is a real pathfinding query per get_observation() (per agent per physics step), so
# budget it on large maps.
#
# The NavigationServer query is isolated behind _path_fn so the full observation path is testable
# headlessly via set_path_fn_for_test (no baked navigation map needed) — same seam idiom as
# RaycastSensor2D._cast_fn. Composition into an agent's get_obs() is manual (or via
# NcnnControllerCore.collect_sensors, which is duck-typed on get_observation/obs_size).

const NavMeshMath = preload("res://addons/godot_native_rl/sensors/navmesh_math.gd")

## Target to navigate toward; freed/invalid → zero-filled.
@export var target: Node2D
## Path-length normalizer for closeness: 1 at the target, →0 at/over this path length.
@export_range(0.01, 20000.0) var max_distance: float = 1000.0
## Optimize the path (NavigationServer2D.map_get_path's `optimize`): smoother corners.
@export var optimize_path: bool = true
## Navigation layers mask for the query.
@export_flags_2d_navigation var navigation_layers: int = 1
## Emit the next-waypoint direction in the sensor's local (egocentric) frame — matches
## RelativePositionSensor and is the right default for a rotating agent. false = world frame.
@export var egocentric: bool = true
## When true, a target that map_get_path can only partially reach (disconnected island) zero-fills,
## so a walled-off target reads as "far". Off by default — map_get_path returns a partial path to
## the closest reachable point, which otherwise reads as high closeness. Tune reachable_tolerance.
@export var require_reachable: bool = false
## Distance (in 2D world units) within which the path's end must land on `target` to count as
## reached (set roughly to your agent radius / nav-cell size).
@export_range(0.0, 1000.0) var reachable_tolerance: float = 1.0

var _path_fn = null  # Callable(from: Vector2, to: Vector2) -> PackedVector2Array (test seam)

## Inject a path provider so get_observation() runs without a live navigation map (tests).
func set_path_fn_for_test(fn: Callable) -> void:
	_path_fn = fn

func obs_size() -> int:
	return 3

func get_observation() -> Array:
	if not is_instance_valid(target):
		return [0.0, 0.0, 0.0]
	var from := global_position if is_inside_tree() else position
	var to := target.global_position if target.is_inside_tree() else target.position
	var path := _query_path(from, to)
	if require_reachable and not NavMeshMath.is_reachable(path, to, reachable_tolerance):
		return [0.0, 0.0, 0.0]
	var rot := (global_rotation if is_inside_tree() else rotation) if egocentric else 0.0
	return NavMeshMath.encode_2d(from, path, max_distance, rot)

func _query_path(from: Vector2, to: Vector2) -> Array:
	if _path_fn != null:
		return Array(_path_fn.call(from, to))
	if not is_inside_tree() or get_world_2d() == null:
		return []  # no map available -> unreachable
	var map := get_world_2d().get_navigation_map()
	return Array(NavigationServer2D.map_get_path(map, from, to, optimize_path, navigation_layers))

class_name FlyByGame
extends Node3D
# Minimal 3D flight env: a plane flies at constant speed toward a ring of goals, steered by
# continuous pitch/turn. Motion is integrated manually on a Node3D (like RoverGame) so the env
# is headless-deterministic and the math is unit-testable as pure helpers. The cartoon_plane
# glTF is a cosmetic child of the plane body and does not affect the sim.

@export var arena_half := Vector3(60.0, 40.0, 60.0)  ## half-extent of the flight box (meters)
@export var flight_speed := 20.0  ## constant forward speed (m/s)
@export var turn_speed := 2.0     ## yaw rate scale (rad/s at |turn|=1)
@export var pitch_speed := 2.0    ## pitch rate scale (rad/s at |pitch|=1)
@export var goal_radius := 6.0    ## reach detection radius
@export var plane_body_path: NodePath
@export var goals_path: NodePath
@export var rng_seed := -1        ## >= 0 seeds the RNG at _ready for reproducible runs

signal goal_reached
signal exited_arena  ## emitted ONCE per arena-boundary crossing (re-armed when the plane returns inside)

var reaches := 0
var goal_index := 0
var _rng := RandomNumberGenerator.new()
var _plane: Node3D
var _goals: Array = []  # Array[Node3D], the ring, in order
var _at_boundary := false  # latch so a wall press costs one penalty, not one per frame
# Headless/test fallback when no plane body is attached.
var _plane_xform_override := Transform3D()

func _ready() -> void:
	_plane = get_node_or_null(plane_body_path) as Node3D
	_goals = collect_goals(get_node_or_null(goals_path))
	if rng_seed >= 0:
		_rng.seed = rng_seed
	reset_positions()

# --- Pure helpers (unit-tested) ---

# 8-dim observation in the plane-LOCAL frame: current-goal unit direction (3) + dist/50 (1),
# next-goal unit direction (3) + dist/50 (1). Local frame encodes heading (no separate orientation
# obs needed). Mirrors the upstream FlyBy obs layout.
func compute_obs(plane_xform: Transform3D, goal_pos: Vector3, next_goal_pos: Vector3) -> Array:
	var to_goal := plane_xform.affine_inverse() * goal_pos
	var to_next := plane_xform.affine_inverse() * next_goal_pos
	var gd := to_goal.length()
	var nd := to_next.length()
	var g := (to_goal / gd) if gd > 1e-6 else Vector3.ZERO
	var n := (to_next / nd) if nd > 1e-6 else Vector3.ZERO
	return [g.x, g.y, g.z, gd / 50.0, n.x, n.y, n.z, nd / 50.0]

# Rotate a basis by pitch (around its local X) then turn (around world UP), kept orthonormal.
func advance_basis(basis: Basis, pitch: float, turn: float, p_speed: float, t_speed: float, delta: float) -> Basis:
	var b := basis.rotated(basis.x.normalized(), pitch * p_speed * delta)
	b = b.rotated(Vector3.UP, turn * t_speed * delta)
	return b.orthonormalized()

# True iff pos is outside the centered box of the given half-extent.
func out_of_bounds(pos: Vector3, half: Vector3) -> bool:
	return absf(pos.x) > half.x or absf(pos.y) > half.y or absf(pos.z) > half.z

# Clamp a position into the centered box of the given half-extent.
func clamp_to_bounds(pos: Vector3, half: Vector3) -> Vector3:
	return Vector3(clampf(pos.x, -half.x, half.x), clampf(pos.y, -half.y, half.y), clampf(pos.z, -half.z, half.z))

func next_goal_index(i: int, count: int) -> int:
	return (i + 1) % count if count > 0 else 0

# --- Runtime accessors ---
func collect_goals(parent: Node) -> Array:
	var result: Array = []
	if parent == null:
		return result
	for child in parent.get_children():
		if child is Node3D:
			result.append(child)
	return result

func get_plane_xform() -> Transform3D:
	return _plane.transform if _plane != null else _plane_xform_override

func set_plane_xform_for_test(x: Transform3D) -> void:
	_plane_xform_override = x

func goal_count() -> int:
	return _goals.size()

func current_goal_pos() -> Vector3:
	if _goals.is_empty():
		return Vector3.ZERO
	return (_goals[goal_index] as Node3D).position

func next_goal_pos() -> Vector3:
	if _goals.is_empty():
		return Vector3.ZERO
	return (_goals[next_goal_index(goal_index, _goals.size())] as Node3D).position

func max_distance() -> float:
	return arena_half.length() * 2.0

# Distance from the plane to the CURRENT goal (used by the reward shaping Callable).
func distance() -> float:
	return get_plane_xform().origin.distance_to(current_goal_pos())

func get_obs_array() -> Array:
	return compute_obs(get_plane_xform(), current_goal_pos(), next_goal_pos())

# Integrate one step of flight, CLAMPED to the arena box. The plane can't leave (it slides along
# the boundary and must turn back), and exited_arena fires ONCE per crossing (re-armed on return) so
# a wall press costs a single penalty, not one per frame. No episode termination here — the bridge
# truncates at reset_after, matching chase/rover/ball_chase (which also only clamp + truncate).
func move_plane(pitch: float, turn: float, delta: float) -> void:
	var xform := get_plane_xform()
	var b := advance_basis(xform.basis, pitch, turn, pitch_speed, turn_speed, delta)
	var next_pos := xform.origin + (-b.z.normalized()) * flight_speed * delta
	var clamped := clamp_to_bounds(next_pos, arena_half)
	if clamped != next_pos:
		if not _at_boundary:
			_at_boundary = true
			exited_arena.emit()
	else:
		_at_boundary = false
	xform.basis = b
	xform.origin = clamped
	if _plane != null:
		_plane.transform = xform
	else:
		_plane_xform_override = xform

func try_reach_goal() -> void:
	if distance() < goal_radius and not _goals.is_empty():
		reaches += 1
		goal_index = next_goal_index(goal_index, _goals.size())
		goal_reached.emit()

func reset_positions() -> void:
	goal_index = 0
	_at_boundary = false
	var start := Transform3D(Basis(), Vector3.ZERO)
	# Random yaw so episodes don't all start identically.
	start.basis = start.basis.rotated(Vector3.UP, _rng.randf_range(-PI, PI))
	if _plane != null:
		_plane.transform = start
	else:
		_plane_xform_override = start

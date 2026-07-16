class_name GoToGoalGame
extends Node2D

# Goal-conditioned multi-target reach (#386): the GoalSensor worked example. K=3 differently-colored
# targets sit in the arena; exactly ONE is "signaled" (current_goal). One policy must reach whichever
# target is signaled — the signal is fed to the net as a one-hot obs channel (GoalSensor), so the
# same weights pursue different targets on demand. Reaching the signaled target scores a goal and
# re-rolls the signal; touching a wrong target is penalized. Kinematic + fully seeded (seed_rng), so
# canonical state lives INTERNALLY and mirrors to the scene nodes (like gridworld_game.gd) — that
# keeps the game exercisable headless with no scene through set_state_for_test() + resolve_touches().

const K := 3  ## number of targets (also the GoalSensor one-hot width)

@export var arena_size := Vector2(1000, 600)
@export var move_speed := 300.0  ## read by the agent controller to scale velocity
@export var target_radius := 40.0  ## reach/touch distance for any target
@export var agent_body_path: NodePath
@export var target_paths: Array[NodePath] = []  ## one NodePath per target (typed so a hand-authored .tscn resolves)

signal goal_reached  ## the signaled target was reached (goal scored, signal re-rolled)
signal wrong_target_touched  ## a non-signaled target was touched (penalty)

var current_goal: int = 0  ## the signaled target id in [0, K)
var goals_reached := 0
var wrong_touches := 0
var goals_reached_by_id: Array = [0, 0, 0]  ## per-goal reach count (length K — the >=K-distinct proof)

var _rng := RandomNumberGenerator.new()
var _agent_body: Node2D
var _targets: Array = []  ## resolved target nodes (Array[Node2D])
var _agent_pos := Vector2.ZERO  ## canonical agent position (authoritative; synced to _agent_body)
var _target_positions: Array = [Vector2.ZERO, Vector2.ZERO, Vector2.ZERO]  ## canonical target positions (length K)

func _ready() -> void:
	_agent_body = get_node_or_null(agent_body_path) as Node2D
	_targets.clear()
	for p in target_paths:
		_targets.append(get_node_or_null(p) as Node2D)
	reset_positions()

# --- Pure helpers (unit-tested) ---
func clamp_to_bounds(pos: Vector2) -> Vector2:
	return Vector2(clampf(pos.x, 0.0, arena_size.x), clampf(pos.y, 0.0, arena_size.y))

func max_distance() -> float:
	return arena_size.length()

## s must be a non-negative integer (RandomNumberGenerator.seed is uint64; negatives wrap).
func seed_rng(s: int) -> void:
	_rng.seed = s

func random_position() -> Vector2:
	return Vector2(_rng.randf_range(0.0, arena_size.x), _rng.randf_range(0.0, arena_size.y))

# --- Signaled-goal accessors (the GoalSensor pulls get_current_goal; reward shaping uses the rest) ---
func get_current_goal() -> int:
	return current_goal

func get_agent_pos() -> Vector2:
	return _agent_pos

func get_target_pos(i: int) -> Vector2:
	return _target_positions[i] if i >= 0 and i < _target_positions.size() else Vector2.ZERO

## Distance from the agent to the CURRENTLY-signaled target (progress-shaping target).
func signaled_distance() -> float:
	return _agent_pos.distance_to(_target_positions[current_goal])

# --- Movement (canonical state authoritative; mirrored to the scene node) ---
func move_agent(velocity: Vector2, delta: float) -> void:
	_agent_pos = clamp_to_bounds(_agent_pos + velocity * delta)
	_sync_positions()

func _sync_positions() -> void:
	if _agent_body != null:
		_agent_body.position = _agent_pos
	for i in range(_targets.size()):
		if i < _target_positions.size() and _targets[i] != null:
			_targets[i].position = _target_positions[i]

# --- Goal signalling ---
## Re-roll the signaled goal (seeded; may land on the same id — a repeated signal is legal).
func roll_goal() -> void:
	current_goal = _rng.randi_range(0, K - 1)

# --- Placement (seeded, non-overlapping) ---
func _min_separation() -> float:
	return target_radius * 2.0

## Draw a fresh position (seeded) that clears the agent + every position in `against` by the min
## separation. Bounded fallback: accept the last draw after a cap so a tight arena can't hang.
func _place_clear_of(against: Array) -> Vector2:
	var p := random_position()
	for _attempt in range(64):
		var ok := p.distance_to(_agent_pos) > _min_separation()
		if ok:
			for other in against:
				if p.distance_to(other) <= _min_separation():
					ok = false
					break
		if ok:
			return p
		p = random_position()
	return p

func reset_positions() -> void:
	_agent_pos = random_position()
	_target_positions = []
	for _i in range(K):
		_target_positions.append(_place_clear_of(_target_positions))
	roll_goal()
	_sync_positions()

## Relocate one target (seeded), keeping it clear of the agent + the other targets.
func _relocate_target(i: int) -> void:
	var against := []
	for j in range(_target_positions.size()):
		if j != i:
			against.append(_target_positions[j])
	_target_positions[i] = _place_clear_of(against)

func _relocate_all_targets() -> void:
	var fresh := []
	for _i in range(K):
		fresh.append(Vector2.ZERO)
	_target_positions = fresh
	for i in range(K):
		var against := []
		for j in range(K):
			if j != i:
				against.append(_target_positions[j])
		_target_positions[i] = _place_clear_of(against)

# --- Terminal resolution (model: seek's relocate_goal/trigger_hazard_hit + gridworld's resolve_terminal) ---
## Resolve the agent's contact with the targets this step. The signaled target has priority: reaching
## it scores a goal (counters credited BEFORE relocation — the seek baseline-rebase gotcha), relocates
## ALL targets and re-rolls the signal; touching a WRONG target penalizes and relocates only that
## target. Returns true if a scored event fired. Discrete + idempotent so overlap can't re-trigger
## every tick (the touched target is moved away).
func resolve_touches() -> bool:
	if signaled_distance() <= target_radius:
		goals_reached += 1
		goals_reached_by_id[current_goal] += 1
		_relocate_all_targets()
		roll_goal()
		_sync_positions()
		goal_reached.emit()
		return true
	for i in range(K):
		if i == current_goal:
			continue
		if _agent_pos.distance_to(_target_positions[i]) <= target_radius:
			wrong_touches += 1
			_relocate_target(i)
			_sync_positions()
			wrong_target_touched.emit()
			return true
	return false

# --- Test seam (pure logic; mirrors gridworld_game.gd:125) ---
func set_state_for_test(agent_pos: Vector2, target_positions: Array, goal: int) -> void:
	_agent_pos = agent_pos
	_target_positions = target_positions.duplicate()
	current_goal = goal
	_sync_positions()

# --- Lightweight visualizer (free headless: _draw never runs without a renderer) ---
func _process(_delta: float) -> void:
	queue_redraw()

func _draw() -> void:
	if not is_inside_tree():
		return  # cosmetic only; never runs headless, never touches obs/logic
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.10, 0.11, 0.15), true)
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.30, 0.32, 0.42), false, 2.0)
	var colors := [Color(0.95, 0.45, 0.35), Color(0.40, 0.85, 0.50), Color(0.45, 0.60, 0.95)]
	for i in range(_target_positions.size()):
		var pos: Vector2 = _target_positions[i]
		var col: Color = colors[i % colors.size()]
		if i == current_goal:
			# Highlight the signaled target with an outer glow ring.
			draw_circle(pos, target_radius + 12.0, Color(1.0, 1.0, 0.5, 0.35))
			draw_arc(pos, target_radius + 12.0, 0.0, TAU, 48, Color(1.0, 1.0, 0.6), 3.0)
		draw_circle(pos, target_radius, col)
	draw_circle(_agent_pos, 15.0, Color(0.30, 0.80, 1.0))

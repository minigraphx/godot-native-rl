class_name CoopCollectGame
extends Node2D

# Owns the Cooperative Collect world and one shared episode (MA-POCA scaffold, #30). All world
# mutation happens here in one prioritized _physics_process (runs BEFORE the agents via
# process_physics_priority), so the two agents never race on shared state: each agent only SETS its
# velocity and READS the cached team reward / collected / terminal state. The reward is a SHARED TEAM
# reward — both agents read the same per-frame value — which is the credit-assignment setting
# MA-POCA's centralized critic targets. Geometry is game-local (tile-offset-safe for ParallelArena2D).

const M = preload("res://examples/coop_collect/coop_collect_math.gd")

@export var arena_size := Vector2(1000, 600)
@export var move_speed := 300.0
@export var collect_radius := 40.0
@export var item_value := 1.0
@export var step_penalty := 0.01        ## flat per-frame penalty -> rewards finishing fast (coop pressure)
@export var max_steps := 400            ## episode timeout (frames)
@export var item_norm := 1200.0         ## normalizer for relative-position obs
@export var item_count := 4
@export var seed_value := 12345         ## deterministic item layout (seeded)
@export var agent_a_body_path: NodePath
@export var agent_b_body_path: NodePath
# Early-finish "bank and leave" mode (#30 M3). Off by default -> M2 behavior is byte-identical.
@export var early_finish := false
@export var bank_width := 120.0         ## right-edge bank zone width
@export var bank_bonus := 0.2           ## one-time shared bonus when an agent banks (gated on a team contribution)

var _bodies: Array[Node2D] = []
var _vels: Array[Vector2] = [Vector2.ZERO, Vector2.ZERO]
var _items: Array[Vector2] = []
var _collected: Array = []
var _banked: Array = [false, false]   ## per-agent: has this agent banked out and left?
var _step := 0
var _team_reward := 0.0   ## team reward produced THIS frame (read identically by every agent)
var _terminal := false
var _pending_reset := false
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	process_physics_priority = -10  # run before the agents so cached state reflects this frame
	var a := get_node_or_null(agent_a_body_path) as Node2D
	var b := get_node_or_null(agent_b_body_path) as Node2D
	_bodies = [a, b]
	_spawn_items()
	reset_positions()

func _spawn_items() -> void:
	# Deterministic, seeded item layout (so episodes/replays/tests are reproducible).
	_rng.seed = seed_value
	_items.clear()
	for i in range(item_count):
		_items.append(Vector2(
			_rng.randf_range(60.0, arena_size.x - 60.0),
			_rng.randf_range(60.0, arena_size.y - 60.0)))
	_collected = []
	_collected.resize(item_count)
	_collected.fill(false)

func reset_positions() -> void:
	# Agents start clustered on the left so they must spread out to reach items efficiently.
	if _bodies[0] != null:
		_bodies[0].position = Vector2(80.0, arena_size.y * 0.4)
	if _bodies[1] != null:
		_bodies[1].position = Vector2(80.0, arena_size.y * 0.6)
	_collected.fill(false)
	_banked = [false, false]
	_step = 0
	_team_reward = 0.0
	_terminal = false

func _physics_process(delta: float) -> void:
	if _pending_reset:
		reset_positions()
		_pending_reset = false
		return
	# Integrate velocities (kinematic), clamp to bounds. A banked agent is inert (parked, no move).
	for i in range(_bodies.size()):
		if _bodies[i] == null:
			continue
		if early_finish and _banked[i]:
			continue
		_bodies[i].position = clamp_to_bounds(_bodies[i].position + _vels[i] * delta)
	# Resolve collection against the ACTIVE agents' positions, compute the shared team reward.
	var newly := M.collect_step(_items, _collected, active_agent_positions(), collect_radius)
	_team_reward = M.team_step_reward(newly, item_value, step_penalty)
	# Early-finish: an active agent in the bank zone (after a team contribution) banks out, parks,
	# and adds a one-time bank bonus to the shared team reward. Its earlier collecting actions still
	# get credit for the team reward earned AFTER it leaves (posthumous credit — the trainer masks it).
	if early_finish:
		var collected_n := M.count_collected(_collected)
		for i in range(_bodies.size()):
			if _bodies[i] == null or _banked[i]:
				continue
			if M.should_bank(M.in_bank_zone(_bodies[i].position, arena_size.x, bank_width), collected_n, _banked[i]):
				_banked[i] = true
				_vels[i] = Vector2.ZERO
				_team_reward += bank_bonus
	_step += 1
	var done := M.all_collected(_collected) or _step >= max_steps
	if early_finish:
		done = done or M.all_banked(_banked)
	_terminal = done

func clamp_to_bounds(pos: Vector2) -> Vector2:
	return Vector2(clampf(pos.x, 0.0, arena_size.x), clampf(pos.y, 0.0, arena_size.y))

# --- Velocity setters (called by agents; applied next physics frame) ---
func set_agent_velocity(idx: int, v: Vector2) -> void:
	if idx >= 0 and idx < _vels.size():
		_vels[idx] = v

# --- Cached-state getters (read by agents) ---
func agent_positions() -> Array:
	var out: Array = []
	for b in _bodies:
		out.append(b.position if b != null else Vector2.ZERO)
	return out

# Positions of agents still in play (banked agents don't collect). In M2 mode (no banking) this is
# all agents, so collection is unchanged.
func active_agent_positions() -> Array:
	var out: Array = []
	for i in range(_bodies.size()):
		if _bodies[i] == null:
			continue
		if early_finish and _banked[i]:
			continue
		out.append(_bodies[i].position)
	return out

# Per-agent active flag (false once banked). Always true in M2 mode.
func agent_active(idx: int) -> bool:
	if idx < 0 or idx >= _banked.size():
		return true
	return not (early_finish and _banked[idx])

func banked() -> Array:
	return _banked

func agent_pos(idx: int) -> Vector2:
	return _bodies[idx].position if (idx >= 0 and idx < _bodies.size() and _bodies[idx] != null) else Vector2.ZERO

func teammate_pos(idx: int) -> Vector2:
	return agent_pos(1 - idx)  # 2-agent team

func items() -> Array:
	return _items

func collected() -> Array:
	return _collected

func team_reward() -> float:
	return _team_reward

func is_terminal() -> bool:
	return _terminal

func request_reset() -> void:
	_pending_reset = true

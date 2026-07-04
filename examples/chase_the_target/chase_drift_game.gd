# ChaseGame variant with a FLEEING target — the environment-shift half of the on-device
# fine-tuning story (#131 follow-up): the shipped chase_es net was trained against a STATIC
# target (it only jumps on catch). A target that runs AWAY is a genuinely changed task with the
# SAME observation/action contract — exactly the "game got patched, the shipped brain adapts
# in-engine" scenario warm_start_*_path exists for. (A plain random drift was tried first and
# rejected: the greedy go-to-target policy transfers to it unharmed — 21 catches vs 19 static —
# so it demonstrates nothing. Fleeing measurably degrades the shipped net.)
#
# The flee direction is the away-vector blended with seeded-RNG jitter (redrawn every
# drift_redraw_ticks; one RNG draw per redraw regardless of parameters, so the ES trainer's
# common-random-numbers seeding stays stream-stable across candidates). Walls REFLECT the
# direction (pure helper below) rather than clamp-pin it, so the target corners instead of
# camping — cornering is precisely where an adapted policy can beat the naive beeline.
extends "res://examples/chase_the_target/chase_game.gd"

@export var target_speed := 260.0  ## flee speed, px/s (agent move_speed is 300 — beeline barely gains; cornering wins)
@export var drift_redraw_ticks := 30  ## physics ticks between fresh jitter draws
@export var flee_jitter := 0.3  ## 0 = pure away-vector, 1 = pure random drift

var _drift_dir := Vector2.RIGHT
var _jitter_dir := Vector2.RIGHT
var _drift_ticks := 0


# --- Pure helpers (unit-tested) ---

## Fresh unit drift direction from the (seeded) RNG.
static func drift_dir_from(rng: RandomNumberGenerator) -> Vector2:
	return Vector2.from_angle(rng.randf_range(0.0, TAU))

## Flee direction: away from the chaser, blended with a jitter direction. Degenerate zero
## vectors (agent exactly on target, or opposing blend) fall back to the jitter/away component.
static func flee_dir(target: Vector2, agent: Vector2, jitter_dir: Vector2, jitter: float) -> Vector2:
	var away := target - agent
	if away.length_squared() < 1e-12:
		return jitter_dir
	var blended := away.normalized() * (1.0 - jitter) + jitter_dir * jitter
	if blended.length_squared() < 1e-12:
		return away.normalized()
	return blended.normalized()

## One drift step with wall reflection: returns [new_pos, new_dir]. A component that would
## leave [0, arena] is mirrored and its direction sign flipped, so motion stays lively at the
## boundary instead of pinning to it.
static func drift_step(pos: Vector2, dir: Vector2, speed: float, delta: float, arena: Vector2) -> Array:
	var p := pos + dir * speed * delta
	var d := dir
	if p.x < 0.0 or p.x > arena.x:
		p.x = clampf(p.x, 0.0, arena.x) * 2.0 - p.x
		p.x = clampf(p.x, 0.0, arena.x)  # guard a step larger than the arena itself
		d.x = -d.x
	if p.y < 0.0 or p.y > arena.y:
		p.y = clampf(p.y, 0.0, arena.y) * 2.0 - p.y
		p.y = clampf(p.y, 0.0, arena.y)
		d.y = -d.y
	return [p, d]


# --- Runtime ---

func _physics_process(delta: float) -> void:
	_drift_ticks += 1
	if _drift_ticks >= drift_redraw_ticks:
		_drift_ticks = 0
		_jitter_dir = drift_dir_from(_rng)
	_drift_dir = flee_dir(get_target_pos(), get_agent_pos(), _jitter_dir, flee_jitter)
	var stepped := drift_step(get_target_pos(), _drift_dir, target_speed, delta, arena_size)
	if _target != null:
		_target.position = stepped[0]

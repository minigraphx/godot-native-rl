extends SceneTree

const Harness = preload("res://test/harness.gd")
const RewardBuilder = preload("res://reward/reward_builder.gd")

const MAX_DIST := 100.0
const STEP_PENALTY := 0.001
const TOUCH_BONUS := 1.0
const TOUCH_RADIUS := 5.0
const INITIAL_DIST := 100.0
const POST_RESET_DIST := 95.0   # distance after reset_positions()

# Mutable distance holder feeding the new pipeline's value Callable.
class Holder:
	extends RefCounted
	var value := 0.0
	func dist() -> float:
		return value
	func maxd() -> float:
		return MAX_DIST

# A scripted trajectory of post-move distances. A "catch" happens whenever the
# distance dips below TOUCH_RADIUS; on catch the target relocates to `relocate_to`.
# A reset happens at the flagged step.
const TRAJECTORY := [
	{ "d": 90.0, "relocate_to": -1.0, "reset": false },
	{ "d": 70.0, "relocate_to": -1.0, "reset": false },
	{ "d": 4.0,  "relocate_to": 80.0, "reset": false },   # catch -> relocate to 80
	{ "d": 60.0, "relocate_to": -1.0, "reset": false },
	{ "d": 3.0,  "relocate_to": 50.0, "reset": false },   # catch -> relocate to 50
	{ "d": 40.0, "relocate_to": -1.0, "reset": true },    # episode reset after this step
	{ "d": 30.0, "relocate_to": -1.0, "reset": false },
]

# --- OLD inline formula (mirrors the pre-migration ChaseAgent) ---
# Baseline starts at INITIAL_DIST (the original primed `_prev_dist = _game.distance()` in _ready).
func _old_return() -> float:
	var prev := INITIAL_DIST
	var total := 0.0
	for entry in TRAJECTORY:
		var cur: float = entry["d"]
		var touched: bool = cur < TOUCH_RADIUS
		var progress := (prev - cur) / MAX_DIST
		var r := progress - STEP_PENALTY
		if touched:
			r += TOUCH_BONUS
		total += r
		if touched:
			prev = entry["relocate_to"]    # rebase to NEW target (original: _prev = distance())
		else:
			prev = cur
		if entry["reset"]:
			prev = POST_RESET_DIST          # original: _prev = distance() after reset_positions
	return total

# --- NEW pipeline ---
func _new_return() -> float:
	var holder := Holder.new()
	holder.value = INITIAL_DIST          # baseline primed here at build time
	var reward = RewardBuilder.new() \
		.add_progress_shaping(holder.dist, holder.maxd, ["target_caught"]) \
		.add_event_bonus("target_caught", TOUCH_BONUS) \
		.add_step_penalty(STEP_PENALTY) \
		.build()

	var total := 0.0
	for entry in TRAJECTORY:
		holder.value = entry["d"]          # post-move distance
		total += reward.evaluate(null)     # accumulate BEFORE relocate
		if entry["d"] < TOUCH_RADIUS:
			holder.value = entry["relocate_to"]   # relocate moves the target
			reward.trigger_event("target_caught") # signal-driven: rebase + queue bonus
		if entry["reset"]:
			holder.value = POST_RESET_DIST
			reward.reset()
	return total

func _initialize() -> void:
	var h := Harness.new()
	var old_total := _old_return()
	var new_total := _new_return()
	# Float-exact episode-return parity (same constants, same arithmetic order per term).
	h.assert_true(abs(old_total - new_total) < 1e-6, \
		"episode return parity (old=%f new=%f)" % [old_total, new_total])
	h.finish(self)

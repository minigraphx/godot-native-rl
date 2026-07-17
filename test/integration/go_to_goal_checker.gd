extends "res://test/integration/trained_checker_base.gd"
# Two-mode GoToGoal behavioral gate (#386): the M2 goal-conditioning proof.
#
# require_goal = true (trained net, goal channel live): the net must demonstrably FOLLOW the
# signal — enough signaled goals, ALL 3 distinct target ids reached (one policy, many goals),
# and wrong touches bounded to half the goals (2:1 goals:wrong; chance with 3 targets is ~1:2,
# so this is ~4x better than luck).
#
# require_goal = false (SAME committed net, GoalSensor.goal_blind = true zeroes the channel):
# the blind control must FAIL that bar — it cannot know which target is signaled, so signaled
# goals collapse to ~1-in-3 luck. It passes iff the trained criteria are NOT met, proving the
# goal channel is load-bearing (the chase_memory ablate_memory pattern).
#
# Calibrated against real headless runs (5400 frames, action_repeat 4, goals/wrong/by_id):
#   trained: seed 42: 17/5/[7,5,5]  7: 22/3/[7,10,5]  3: 21/7/[10,7,4]  11: 8/3/[5,1,2]  99: 12/4/[4,7,1]
#   blind:   seed 42: 4/16/[4,0,0]  7: 7/8/[4,2,1]    3: 7/10/[3,2,2]   11: 2/19/[2,0,0]  99: 5/20/[2,1,2]
# Across ALL five seeds the trained net clears every bound (worst goals 8 vs min 6; worst
# goals:wrong 2.67 vs required 2.0; distinct 3/3 every seed) and the blind net fails the ratio
# everywhere (best 0.875:1 — a >2x margin below the 2:1 bar), often also min_goals + distinct.
# The scenes pin seed 42, where the blind net fails ALL THREE criteria at once (redundancy) and
# the trained side keeps ~3x goals headroom — loose bounds for cross-machine divergence.

@export var require_goal := true  ## true = trained gate; false = goal-blind ablation control
@export var min_goals := 6  ## trained must reach at least this many SIGNALED goals
@export var min_distinct_ids := 3  ## trained must reach this many DISTINCT target ids
@export var max_wrong_ratio := 0.5  ## wrong touches must stay under this fraction of goals

func _distinct_ids() -> int:
	var n := 0
	for c in _game.goals_reached_by_id:
		if int(c) > 0:
			n += 1
	return n

func _follows_signal() -> bool:
	var goals := int(_game.goals_reached)
	return goals >= min_goals \
		and _distinct_ids() >= min_distinct_ids \
		and float(_game.wrong_touches) <= float(goals) * max_wrong_ratio

func _label() -> String:
	return "GOTOGOAL TRAINED" if require_goal else "GOTOGOAL BLIND"

func _passed() -> bool:
	if require_goal:
		return _follows_signal()
	# Blind control: the ablated net must NOT match the trained bar — if it does, the goal
	# channel was not load-bearing and this regression MUST fail loud.
	return not _follows_signal()

func _report() -> String:
	return "goals=%d wrong=%d by_id=%s distinct=%d in %d frames" % [
		int(_game.goals_reached), int(_game.wrong_touches),
		str(_game.goals_reached_by_id), _distinct_ids(), _frames]

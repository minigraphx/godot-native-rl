extends Node
# Goal-signal obs channel (#386): a num_goals-wide one-hot of the current goal, auto-discovered by
# collect_sensors() (has both get_observation() + obs_size()). Dimension-agnostic (a goal id has no
# 2D/3D split). The goal is pulled from `goal_source_path` (duck-typed get_current_goal()) when set,
# else from the pushed `current_goal`. `goal_blind` zeroes the channel — regression-only, NEVER set
# in a shipped scene (the goal-blind control that proves the signal is load-bearing).

const GoalMath = preload("res://addons/godot_native_rl/sensors/goal_math.gd")

@export var num_goals: int = 2
@export var goal_source_path: NodePath = ^""
@export var goal_blind: bool = false

var current_goal: int = -1
var _source: Node = null
var _warned := false

func set_goal(id: int) -> void:
	current_goal = id

func obs_size() -> int:
	return num_goals

func _resolve_goal() -> int:
	if goal_source_path != ^"":
		if _source == null:
			_source = get_node_or_null(goal_source_path)
		if _source != null and _source.has_method("get_current_goal"):
			return int(_source.get_current_goal())
		if not _warned:
			push_error("GoalSensor: goal_source_path %s has no get_current_goal(); emitting zeros." % str(goal_source_path))
			_warned = true
		return -1
	return current_goal

func get_observation() -> Array:
	return GoalMath.one_hot(_resolve_goal(), num_goals, goal_blind)

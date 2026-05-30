class_name ProgressShapingTerm
extends "res://reward/terms/reward_term.gd"

var _value_fn: Callable
var _scale            # float or Callable
var _rebase_on: Array
var _prev: float

func _init(value_fn: Callable, scale, rebase_on: Array = []) -> void:
	_value_fn = value_fn
	_scale = scale
	_rebase_on = rebase_on
	_prev = float(_value_fn.call())   # prime baseline at construction

func _current_scale() -> float:
	return float(_scale.call()) if _scale is Callable else float(_scale)

func evaluate(_ctx) -> float:
	var cur := float(_value_fn.call())
	var scale := _current_scale()
	var progress := (_prev - cur) / scale if scale != 0.0 else 0.0
	_prev = cur
	return progress

func on_event(event_name: String) -> void:
	# Rebase baseline to the value sampled NOW (e.g. right after a target relocate),
	# so the discontinuity is not scored as progress on the next step.
	if event_name in _rebase_on:
		_prev = float(_value_fn.call())

func reset() -> void:
	_prev = float(_value_fn.call())

class_name EventBonusTerm
extends "res://addons/godot_native_rl/reward/terms/reward_term.gd"

var _event_name: String
var _amount: float
var _pending := 0.0

func _init(event_name: String, amount: float) -> void:
	_event_name = event_name
	_amount = amount

func on_event(event_name: String) -> void:
	if event_name == _event_name:
		_pending += _amount

func evaluate(_ctx) -> float:
	var r := _pending
	_pending = 0.0
	return r

func reset() -> void:
	_pending = 0.0

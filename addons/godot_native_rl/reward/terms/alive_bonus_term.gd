class_name AliveBonusTerm
extends "res://addons/godot_native_rl/reward/terms/reward_term.gd"

var _amount: float

func _init(amount: float) -> void:
	_amount = amount

func evaluate(_ctx) -> float:
	return _amount

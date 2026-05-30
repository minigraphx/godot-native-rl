class_name AliveBonusTerm
extends "res://reward/terms/reward_term.gd"

var _amount: float

func _init(amount: float) -> void:
	_amount = amount

func evaluate(_ctx) -> float:
	return _amount

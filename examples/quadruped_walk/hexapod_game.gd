extends "res://examples/quadruped_walk/quadruped_game.gd"
# The hexapod morphology (#60 M3): identical world + obs/reward surface as the quadruped, just a
# 6-leg rig. The base game is leg-count-agnostic (reads _rig sizes), so only the builder changes.

const HexBuilder = preload("res://examples/quadruped_walk/hexapod_builder.gd")

func _make_rig() -> Dictionary:
	return HexBuilder.build(self)

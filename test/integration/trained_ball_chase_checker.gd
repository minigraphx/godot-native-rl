extends "res://test/integration/trained_checker_base.gd"
# Trained Ball_Chase behavioral gate: the trained policy must actually reach targets.
# Skeleton (node/model guards, pass/fail plumbing) lives in the base (#335).

@export var min_reaches := 3

func _label() -> String:
	return "TRAINED BALL_CHASE"

func _passed() -> bool:
	return int(_game.reaches) >= min_reaches

func _report() -> String:
	if _passed():
		return "%d reaches in %d frames" % [int(_game.reaches), _frames]
	return "only %d reaches in %d frames (need %d)" % [int(_game.reaches), _frames, min_reaches]

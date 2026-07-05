extends "res://test/integration/trained_checker_base.gd"
# Trained-chase behavioral gate: the ncnn policy must actually CATCH (a random policy almost
# never clears the threshold). Skeleton (seed + re-roll + guards) lives in the base (#335).

@export var min_catches := 5

func _label() -> String:
	return "TRAINED CHASE"

func _passed() -> bool:
	return int(_game.catches) >= min_catches

func _report() -> String:
	if _passed():
		return "%d catches in %d frames" % [int(_game.catches), _frames]
	return "only %d catches in %d frames (need %d) — agent did not learn to chase" % [int(_game.catches), _frames, min_catches]

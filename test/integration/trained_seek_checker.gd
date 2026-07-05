extends "res://test/integration/trained_checker_base.gd"
# Trained-seek behavioral gate (#38): BOTH halves of the task — reaching goals (a random policy
# almost never clears the threshold) AND bounded hazard contact (a goal-only greedy policy plows
# through the hazard more often). Skeleton lives in the base (#335).

@export var min_goals := 5
@export var max_hazard_hits := 6

func _label() -> String:
	return "TRAINED SEEK"

func _passed() -> bool:
	return int(_game.goals_reached) >= min_goals and int(_game.hazard_hits) <= max_hazard_hits

func _report() -> String:
	if _passed():
		return "%d goals, %d hazard hits in %d frames" % [int(_game.goals_reached), int(_game.hazard_hits), _frames]
	return "%d goals (need >= %d) with %d hazard hits (cap %d) in %d frames" % [
		int(_game.goals_reached), min_goals, int(_game.hazard_hits), max_hazard_hits, _frames]

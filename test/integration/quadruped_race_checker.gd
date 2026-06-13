extends Node
# Headless learning-arc check for the #60 M4 sequential generation race: one quadruped runs the
# 500k / 2.5M / 6M training-generation nets in turn (clean solo physics — no multi-ragdoll Jolt
# interference), and we assert the recorded distances show the arc: the latest generation (6M) walks
# substantially farther than the earliest (500k). Reads the SequentialRace controller's results once
# it has cycled through all generations.

@export var race_path: NodePath
@export var max_frames := 14000   ## safety budget (3 gens * frames_per_gen + slack)
@export var min_arc_gap := 8.0    ## 6M generation must out-walk 500k by at least this many metres

var _race
var _frames := 0

func _ready() -> void:
	_race = get_node_or_null(race_path)
	if _race == null:
		_fail("no SequentialRace node")

func _physics_process(_delta: float) -> void:
	if _race == null:
		return
	_frames += 1
	if _race.is_done():
		var results: Array = _race.results()
		var by_label := {}
		for r in results:
			by_label[r["label"]] = r["distance"]
		var early: float = by_label.get("gen-500k", 0.0)
		var late: float = by_label.get("gen-6M", 0.0)
		print("RACE DIAG: ", results)
		if late - early >= min_arc_gap:
			print("RACE LEARNING-ARC PASSED (6M %.2f m vs 500k %.2f m; +%.2f)" % [late, early, late - early])
			get_tree().quit(0)
		else:
			_fail("6M (%.2f) led 500k (%.2f) by only %.2f m (need %.1f)" % [late, early, late - early, min_arc_gap])
		return
	if _frames >= max_frames:
		_fail("race did not finish within %d frames" % max_frames)

func _fail(reason: String) -> void:
	printerr("RACE CHECK FAILED: %s" % reason)
	get_tree().quit(1)

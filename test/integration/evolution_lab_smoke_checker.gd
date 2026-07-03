extends Node
# Headless smoke for the Evolution Lab demo (#291): a tiny in-engine ES run inside the REAL demo
# scene. Asserts the demo's composition — generations fire, and the champion agent actually
# hot-swaps onto the trainer's blessed checkpoint (the checkpoint_saved → reload_model wiring),
# which is the scene's whole point. Visual polish is a human check, not CI's.

const Harness = preload("res://test/harness.gd")

const TIMEOUT_SEC := 60.0  # wall clock (the timer ignores the demo's time_scale)
const WANT_GENERATIONS := 3

var _h = Harness.new()
var _lab: Node = null
var _generations_seen := 0


func _ready() -> void:
	_lab = get_parent().get_node("EvolutionLab")
	var trainer: Node = _lab.get_node("ESTrainer")
	trainer.generation_finished.connect(_on_generation)
	var timeout := get_tree().create_timer(TIMEOUT_SEC, true, false, true)
	timeout.timeout.connect(func() -> void:
		printerr("FAIL: Evolution Lab smoke timed out after %.0fs wall clock (saw %d generations)"
			% [TIMEOUT_SEC, _generations_seen])
		get_tree().quit(1))


func _on_generation(_gen: int, _mean: float, _best: float) -> void:
	_generations_seen += 1
	if _generations_seen < WANT_GENERATIONS:
		return
	_h.assert_true(_lab.swap_count() >= 1,
		"champion hot-swapped onto a blessed net (swaps: %d)" % _lab.swap_count())
	var champion: Node = _lab.get_node("ChampionWorld/ChaseAgent")
	_h.assert_true(champion._ncnn_runner != null and champion._ncnn_runner.is_model_loaded(),
		"champion runner holds a live net")
	_h.finish(get_tree())

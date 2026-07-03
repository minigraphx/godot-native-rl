extends Node
# Headless smoke for the native in-engine ES trainer (#131): two tiny generations over the real
# chase world (k=2 seeded episodes per candidate). Asserts the WIRING — generations fire with
# finite fitness, θ actually updates, periodic + final checkpoints are deploy-ready ncnn pairs
# that load back into a live net. Whether ES learns chase well is a long-run behavioral question,
# deliberately NOT asserted here: improvement over 2 generations of a noisy env would flake.

const Harness = preload("res://test/harness.gd")

# Wall-clock seconds. The trainer sets Engine.time_scale (speed_up), so the watchdog timer MUST
# ignore time scale — a scaled SceneTreeTimer at speed_up=20 would fire after ~3 real seconds.
const TIMEOUT_SEC := 60.0

var _h = Harness.new()
var _trainer: Node = null
var _theta_before := PackedFloat32Array()
var _generations_seen := 0
var _fitness_finite := true


func _ready() -> void:
	_wipe_stale_checkpoints()
	_trainer = get_parent().get_node("ESTrainer")
	_trainer.generation_finished.connect(_on_generation)
	_trainer.training_finished.connect(_on_finished)
	_snapshot_theta.call_deferred()  # after the trainer's own deferred _late_init built θ
	var timeout := get_tree().create_timer(TIMEOUT_SEC, true, false, true)  # ignore_time_scale
	timeout.timeout.connect(func() -> void:
		printerr("FAIL: ES smoke timed out after %.0fs wall clock (saw %d generations)" % [TIMEOUT_SEC, _generations_seen])
		get_tree().quit(1))


# The checkpoint asserts below check files WRITTEN BY THE TRAINER during this run — wipe any
# left by a previous local run first, or a checkpoint-writing regression stays green forever
# on a developer machine (only a fresh CI container would catch it).
func _wipe_stale_checkpoints() -> void:
	var dir := DirAccess.open("user://es_smoke")
	if dir == null:
		return
	for f in dir.get_files():
		if f.begins_with("chase_es_smoke"):
			dir.remove(f)


func _snapshot_theta() -> void:
	_theta_before = _trainer.current_theta()
	_h.assert_true(_theta_before.size() > 0, "trainer built a non-empty theta")


func _on_generation(_gen: int, mean_fitness: float, best_fitness: float) -> void:
	_generations_seen += 1
	if not (is_finite(mean_fitness) and is_finite(best_fitness)):
		_fitness_finite = false


func _on_finished(_best_mean: float) -> void:
	_h.assert_eq(_generations_seen, _trainer.generations, "all generations reported")
	_h.assert_true(_fitness_finite, "all fitness values finite")
	_h.assert_true(_trainer.current_theta() != _theta_before, "theta updated by the ES step")

	# checkpoint_every=1 in the smoke scene: the periodic generation snapshot must exist too.
	_h.assert_true(FileAccess.file_exists(_trainer.out_dir.path_join(_trainer.checkpoint_stem + "_gen1.ncnn.param")),
		"periodic gen checkpoint written (checkpoint_every)")

	# The checkpoint is the deploy artifact: it must load back into a live ncnn net.
	var param_path: String = _trainer.out_dir.path_join(_trainer.checkpoint_stem + "_final.ncnn.param")
	var bin_path: String = _trainer.out_dir.path_join(_trainer.checkpoint_stem + "_final.ncnn.bin")
	_h.assert_true(FileAccess.file_exists(param_path) and FileAccess.file_exists(bin_path),
		"final checkpoint written")
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var loaded: bool = runner.load_model_from_buffers(
		FileAccess.get_file_as_bytes(param_path), FileAccess.get_file_as_bytes(bin_path))
	_h.assert_true(loaded, "checkpoint loads back into a live ncnn net")
	if loaded:
		var out := runner.run_inference(PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5]))
		_h.assert_eq(out.size(), 5, "checkpoint net runs inference (5 action logits)")
	runner.free()
	_h.finish(get_tree())

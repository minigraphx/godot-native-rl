extends Node
# Headless smoke for the native in-engine ES trainer (#131): two tiny generations over the real
# chase world. Asserts the WIRING — generations fire with finite fitness, θ actually updates, and
# the checkpoint is a deploy-ready ncnn pair that loads back into a live net. Whether ES learns
# chase well is a long-run behavioral question (follow-up on #131), deliberately NOT asserted
# here: improvement over 2 generations of a noisy env would flake.

const TIMEOUT_SEC := 60.0

var _trainer: Node = null
var _theta_before := PackedFloat32Array()
var _generations_seen := 0
var _fitness_finite := true
var _failures := 0


func _ready() -> void:
	_trainer = get_parent().get_node("ESTrainer")
	_trainer.generation_finished.connect(_on_generation)
	_trainer.training_finished.connect(_on_finished)
	_snapshot_theta.call_deferred()  # after the trainer's own deferred _late_init built θ
	var timeout := get_tree().create_timer(TIMEOUT_SEC)
	timeout.timeout.connect(func() -> void:
		printerr("FAIL: ES smoke timed out after %.0fs (saw %d generations)" % [TIMEOUT_SEC, _generations_seen])
		get_tree().quit(1))


func _snapshot_theta() -> void:
	_theta_before = _trainer.current_theta()
	_check(_theta_before.size() > 0, "trainer built a non-empty theta")


func _on_generation(_gen: int, mean_fitness: float, best_fitness: float) -> void:
	_generations_seen += 1
	if not (is_finite(mean_fitness) and is_finite(best_fitness)):
		_fitness_finite = false


func _check(cond: bool, label: String) -> void:
	if cond:
		print("  PASS: %s" % label)
	else:
		printerr("  FAIL: %s" % label)
		_failures += 1


func _on_finished(_best_mean: float) -> void:
	_check(_generations_seen == _trainer.generations,
		"all %d generations reported (saw %d)" % [_trainer.generations, _generations_seen])
	_check(_fitness_finite, "all fitness values finite")
	_check(_trainer.current_theta() != _theta_before, "theta updated by the ES step")

	# The checkpoint is the deploy artifact: it must load back into a live ncnn net.
	var param_path: String = _trainer.out_dir.path_join(_trainer.checkpoint_stem + "_final.ncnn.param")
	var bin_path: String = _trainer.out_dir.path_join(_trainer.checkpoint_stem + "_final.ncnn.bin")
	_check(FileAccess.file_exists(param_path) and FileAccess.file_exists(bin_path),
		"final checkpoint written (%s)" % param_path)
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var loaded: bool = runner.load_model_from_buffers(
		FileAccess.get_file_as_bytes(param_path), FileAccess.get_file_as_bytes(bin_path))
	_check(loaded, "checkpoint loads back into a live ncnn net")
	if loaded:
		var out := runner.run_inference(PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5]))
		_check(out.size() == 5, "checkpoint net runs inference (5 action logits)")
	runner.free()

	print("ES trainer smoke: %s" % ("OK" if _failures == 0 else "%d FAILURE(S)" % _failures))
	get_tree().quit(0 if _failures == 0 else 1)

extends Node
# #328 pt 1 live proof: ESTrainer warm-starts from the pnnx-exported PPO chase net (foreign
# layer names + fp16-tagged weights — the structural adapter path, NOT the strict canonical
# one). Asserts the adopted θ is the decoded PPO net bit-for-bit AND that it actually PLAYS:
# the trained PPO policy scores far above an untrained init in generation 1, so a decode that
# loaded garbage (misaligned blobs, wrong transpose) fails here even though θ sizes match.

const Harness = preload("res://test/harness.gd")
const NcnnWeights = preload("res://addons/godot_native_rl/training/ncnn_weights.gd")

const TIMEOUT_SEC := 90.0
const PPO_PARAM := "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
const PPO_BIN := "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"
## An untrained He-init policy hovers around -1..0 on this horizon; the shipped PPO net chases
## reliably. Well below its typical score, far above anything an accidental-garbage θ reaches.
const MIN_GEN1_BEST := 2.0

var _h = Harness.new()
var _trainer: Node = null
var _gen1_best := -INF
var _got_gen := false

func _ready() -> void:
	_trainer = get_parent().get_node("ESTrainer")
	_trainer.generation_finished.connect(_on_generation)
	_trainer.training_finished.connect(_on_finished)
	_check_adopted_theta.call_deferred()
	var timeout := get_tree().create_timer(TIMEOUT_SEC, true, false, true)  # ignore_time_scale
	timeout.timeout.connect(func() -> void:
		printerr("FAIL: warm-start smoke timed out after %.0fs" % TIMEOUT_SEC)
		get_tree().quit(1))

func _check_adopted_theta() -> void:
	var spec: Dictionary = NcnnWeights.mlp_spec([5, 64, 64, 5], "tanh")
	var want: PackedFloat32Array = NcnnWeights.warm_start_theta(spec, PPO_PARAM, PPO_BIN)
	_h.assert_eq(_trainer.current_theta().size(), want.size(), "trainer θ sized like the PPO net (4869 floats)")
	_h.assert_true(_trainer.current_theta() == want, "trainer θ IS the decoded PPO net (adapter path, bit-for-bit)")

func _on_generation(gen: int, _mean: float, best: float) -> void:
	if gen == 0:
		_gen1_best = best
		_got_gen = true

func _on_finished(_best_mean: float) -> void:
	_h.assert_true(_got_gen, "generation stats observed")
	_h.assert_true(_gen1_best > MIN_GEN1_BEST,
		"adopted PPO policy PLAYS in generation 1 (best fitness %.2f > %.1f — garbage θ can't reach this)" % [_gen1_best, MIN_GEN1_BEST])
	_h.finish(get_tree())

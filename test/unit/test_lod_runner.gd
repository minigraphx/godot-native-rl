extends SceneTree
# Integration test for NcnnLODRunner (#21): wires two real ncnn nets (the trained chase policy as
# the deliberative net, the untrained chase_dummy as the reflex net) and asserts that decide()
# alternates tiers on the LOD cadence and that each frame's logits come from the tier that ran —
# i.e. the deliberative net's expensive output appears only on deliberative frames, the cheap
# reflex output on the rest.

const Harness = preload("res://test/harness.gd")
const NcnnLODRunner = preload("res://addons/godot_native_rl/controllers/ncnn_lod_runner.gd")

const REFLEX_PARAM := "res://examples/chase_the_target/models/chase_dummy.ncnn.param"
const REFLEX_BIN := "res://examples/chase_the_target/models/chase_dummy.ncnn.bin"
const DELIB_PARAM := "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
const DELIB_BIN := "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"

var OBS := PackedFloat32Array([0.5479, -0.1222, 0.7172, 0.3947, -0.8116])

func _make_runner(param: String, bin: String):
	var r := NcnnRunner.new()
	r.input_blob_name = "in0"
	r.output_blob_name = "out0"
	var ok := r.load_model(ProjectSettings.globalize_path(param), ProjectSettings.globalize_path(bin))
	return r if ok else null

func _initialize() -> void:
	var h := Harness.new()

	var reflex = _make_runner(REFLEX_PARAM, REFLEX_BIN)
	var delib = _make_runner(DELIB_PARAM, DELIB_BIN)
	h.assert_true(reflex != null, "reflex (dummy) model loads")
	h.assert_true(delib != null, "deliberative (trained) model loads")
	if reflex == null or delib == null:
		h.finish(self)
		return

	# Reference logits from each net for the same obs — the LOD output must match these per tier.
	var reflex_logits: PackedFloat32Array = reflex.run_inference(OBS)
	var delib_logits: PackedFloat32Array = delib.run_inference(OBS)
	h.assert_true(reflex_logits != delib_logits, "the two nets produce different logits (distinct tiers)")

	var lod := NcnnLODRunner.new()
	lod.setup_for_test(reflex, delib, 3)  # deliberative every 3rd frame
	get_root().add_child(lod)

	# 7 frames at interval 3 -> deliberative on 0,3,6.
	var tiers: Array = []
	for i in range(7):
		var d: Dictionary = lod.decide(OBS)
		tiers.append(d["tier"])
		var expected: PackedFloat32Array = delib_logits if d["ran_deliberative"] else reflex_logits
		h.assert_eq(d["logits"], expected, "frame %d logits come from the %s net" % [i, d["tier"]])

	h.assert_eq(tiers,
		["deliberative", "reflex", "reflex", "deliberative", "reflex", "reflex", "deliberative"],
		"LOD cadence runs the deliberative net every 3rd frame")

	# last_deliberative_logits caches the accurate output across the cheap frames.
	h.assert_eq(lod.last_deliberative_logits(), delib_logits, "deliberative logits cached for reflex frames")

	# A state change forces the deliberative net immediately, off-cadence.
	var forced: Dictionary = lod.decide(OBS, true)
	h.assert_eq(forced["tier"], "deliberative", "state_changed forces the deliberative net")

	# reset() re-arms: the next frame deliberates again.
	lod.reset()
	var after_reset: Dictionary = lod.decide(OBS)
	h.assert_true(after_reset["ran_deliberative"], "reset() makes the next frame deliberative")

	lod.free()
	h.finish(self)

extends SceneTree
# Golden-inference regression for the committed visual-chase CNN (#35): fixed code-rasterized
# frames through the REAL deploy image route (run_inference_image) must reproduce the baked
# argmax actions. Catches a silently-corrupted/retrained net or an image-path layout change.
#
# Baked 2026-06-13 from the 1.5M-step net via record_visual_chase_golden.gd. Only probes whose
# top-2 logit gap is >= 3 are baked, so argmax survives cross-platform fp16 conv drift; logits
# themselves are NOT asserted (ncnn runs convs in fp16 on ARM — argmax is the deploy contract).

const Harness = preload("res://test/harness.gd")
const VObs = preload("res://examples/visual_chase/visual_chase_obs.gd")

const ARENA := Vector2(1000, 600)
# [agent, target, expected argmax]  (1=up, 2=down, 3=left, 4=right)
const GOLDEN := [
	[Vector2(500, 300), Vector2(500, 50), 1],
	[Vector2(100, 100), Vector2(950, 550), 4],
	[Vector2(900, 500), Vector2(100, 100), 1],
	[Vector2(500, 500), Vector2(500, 100), 1],
	[Vector2(800, 100), Vector2(200, 500), 2],
	[Vector2(100, 500), Vector2(900, 100), 1],
]

func _initialize() -> void:
	var h = Harness.new()
	var runner = NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var param_bytes := FileAccess.get_file_as_bytes("res://examples/visual_chase/models/visual_chase.ncnn.param")
	var bin_bytes := FileAccess.get_file_as_bytes("res://examples/visual_chase/models/visual_chase.ncnn.bin")
	h.assert_true(not param_bytes.is_empty() and not bin_bytes.is_empty(), "committed visual-chase net present")
	h.assert_true(runner.load_model_from_buffers(param_bytes, bin_bytes), "visual-chase net loads")

	for case in GOLDEN:
		var bytes: PackedByteArray = VObs.rasterize(case[0], case[1], ARENA, 36, 36)
		var img: Image = VObs.make_image(bytes, 36, 36)
		var out: PackedFloat32Array = runner.run_inference_image(img, true)
		h.assert_eq(out.size(), 5, "5 logits for agent=%s target=%s" % [case[0], case[1]])
		var best := 0
		for i in range(out.size()):
			if out[i] > out[best]:
				best = i
		h.assert_eq(best, case[2], "golden argmax for agent=%s target=%s" % [case[0], case[1]])

	h.finish(self)

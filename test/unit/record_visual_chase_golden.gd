extends SceneTree
# RECORD helper for the visual-chase golden test: runs the committed ncnn net on fixed
# synthetic frames (deploy image route) and prints logits + argmax per frame. Bake the
# printed actions into test_visual_chase_golden_inference.gd. Not part of run_tests.sh.

const VObs = preload("res://examples/visual_chase/visual_chase_obs.gd")

func _initialize() -> void:
	var runner = NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var param_bytes := FileAccess.get_file_as_bytes("res://examples/visual_chase/models/visual_chase.ncnn.param")
	var bin_bytes := FileAccess.get_file_as_bytes("res://examples/visual_chase/models/visual_chase.ncnn.bin")
	var ok: bool = runner.load_model_from_buffers(param_bytes, bin_bytes)
	if not ok:
		printerr("RECORD FAILED: model not loaded")
		quit(1)
		return
	var arena := Vector2(1000, 600)
	# Candidate probes (agent, target). Bake only cases whose top-2 logit gap is >= 2.5 so the
	# golden argmax survives cross-platform fp16 conv drift (observed up to ~3 on logits).
	var cases := [
		[Vector2(500, 300), Vector2(900, 300)],
		[Vector2(500, 300), Vector2(100, 300)],
		[Vector2(500, 300), Vector2(500, 50)],
		[Vector2(500, 300), Vector2(500, 550)],
		[Vector2(100, 100), Vector2(950, 550)],
		[Vector2(900, 500), Vector2(100, 100)],
		[Vector2(200, 300), Vector2(800, 300)],
		[Vector2(500, 500), Vector2(500, 100)],
		[Vector2(800, 100), Vector2(200, 500)],
		[Vector2(100, 500), Vector2(900, 100)],
	]
	for c in cases:
		var bytes: PackedByteArray = VObs.rasterize(c[0], c[1], arena, 36, 36)
		var img: Image = VObs.make_image(bytes, 36, 36)
		var out: PackedFloat32Array = runner.run_inference_image(img, true)
		var best := 0
		var second := -1
		for i in range(out.size()):
			if out[i] > out[best]:
				best = i
		for i in range(out.size()):
			if i != best and (second < 0 or out[i] > out[second]):
				second = i
		var gap: float = out[best] - out[second]
		print("agent=", c[0], " target=", c[1], " argmax=", best, " gap=", gap, " logits=", out)
	quit(0)

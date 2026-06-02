extends SceneTree
# Deploy smoke for INT8: loads the committed INT8 synthetic CNN through NcnnRunner and
# asserts run_inference_image runs and its argmax agrees with the fp32 synthetic CNN on the
# golden image. Proves native INT8 *deployment*, not just that conversion produced a file.
# Regenerate the int8 fixture with:
#   .venv-train/bin/python scripts/export_int8.py models/synthetic_cnn.ncnn.param \
#     models/synthetic_cnn.ncnn.bin --width 8 --height 8 --channels 3 --outdir models

const Harness = preload("res://test/harness.gd")
const GOLDEN := "res://models/synthetic_cnn_golden.json"
const FP32_PARAM := "res://models/synthetic_cnn.ncnn.param"
const FP32_BIN := "res://models/synthetic_cnn.ncnn.bin"
const INT8_PARAM := "res://models/synthetic_cnn_int8.ncnn.param"
const INT8_BIN := "res://models/synthetic_cnn_int8.ncnn.bin"

func _argmax(logits: PackedFloat32Array) -> int:
	var best := 0
	for i in range(1, logits.size()):
		if logits[i] > logits[best]:
			best = i
	return best

func _initialize() -> void:
	var h := Harness.new()

	var f := FileAccess.open(GOLDEN, FileAccess.READ)
	h.assert_true(f != null, "golden json opens")
	if f == null:
		h.finish(self)
		return
	var data: Dictionary = JSON.parse_string(f.get_as_text())
	var w := int(data["width"])
	var ht := int(data["height"])
	var img_bytes := PackedByteArray()
	for v in data["image_bytes"]:
		img_bytes.append(int(v))
	var img := Image.create_from_data(w, ht, false, Image.FORMAT_RGB8, img_bytes)

	var int8 := NcnnRunner.new()
	int8.input_blob_name = "in0"
	int8.output_blob_name = "out0"
	var ok := int8.load_model(ProjectSettings.globalize_path(INT8_PARAM), ProjectSettings.globalize_path(INT8_BIN))
	h.assert_true(ok, "INT8 synthetic CNN loads")
	if ok:
		var logits8: PackedFloat32Array = int8.run_inference_image(img, true)
		h.assert_eq(logits8.size(), 4, "INT8 produces 4 logits")

		var fp32 := NcnnRunner.new()
		fp32.input_blob_name = "in0"
		fp32.output_blob_name = "out0"
		fp32.load_model(ProjectSettings.globalize_path(FP32_PARAM), ProjectSettings.globalize_path(FP32_BIN))
		var logits32: PackedFloat32Array = fp32.run_inference_image(img, true)
		h.assert_eq(_argmax(logits8), _argmax(logits32), "INT8 argmax agrees with fp32 on golden image")

	h.finish(self)

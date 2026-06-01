extends SceneTree

const Harness = preload("res://test/harness.gd")
const CameraSensor = preload("res://addons/godot_native_rl/sensors/camera_sensor.gd")

func _make_image(w: int, h: int, fmt: int, fill: Color) -> Image:
	var img := Image.create(w, h, false, fmt)
	img.fill(fill)
	return img

func _initialize() -> void:
	var h := Harness.new()

	# --- RGB path: 2x2 image injected via the test seam ---
	var s = CameraSensor.new()
	var vp := SubViewport.new()
	vp.size = Vector2i(2, 2)
	s.viewport = vp
	# A real RGB8 image with known bytes (all red): each pixel = (255, 0, 0).
	var rgb := _make_image(2, 2, Image.FORMAT_RGB8, Color(1, 0, 0))
	s.set_image_for_test(rgb)

	var obs: String = s.get_observation()
	# 4 pixels * 3 channels = 12 bytes, each pixel "ff0000".
	h.assert_eq(obs, "ff0000ff0000ff0000ff0000", "RGB obs hex == red 2x2")
	h.assert_eq(s.get_obs_shape(), [2, 2, 3], "RGB obs_shape [H,W,3]")
	h.assert_eq(s.get_obs_space_entry(), {"space": "box", "size": [2, 2, 3]}, "RGB obs_space entry")
	h.assert_eq(s.get_observation_key(), "camera_2d", "default observation_key")

	# --- Grayscale path: same image, grayscale=true -> L8, 1 channel ---
	s.grayscale = true
	var gray_obs: String = s.get_observation()
	h.assert_eq(s.get_obs_shape(), [2, 2, 1], "grayscale obs_shape [H,W,1]")
	# 4 pixels * 1 channel = 4 bytes = 8 hex chars.
	h.assert_eq(gray_obs.length(), 8, "grayscale obs hex length == 4 bytes")

	s.free()
	vp.free()

	# --- Missing viewport -> stable empty obs, no crash ---
	var s2 = CameraSensor.new()
	s2.set_image_for_test(_make_image(2, 2, Image.FORMAT_RGB8, Color(1, 0, 0)))
	h.assert_eq(s2.get_observation(), "", "missing viewport -> empty obs")
	h.assert_eq(s2.get_obs_shape(), [0, 0, 3], "missing viewport -> zero shape")
	s2.free()

	# --- Key without "2d" is rejected (validation returns false) ---
	var s3 = CameraSensor.new()
	h.assert_true(s3.is_key_valid("camera_2d"), "key with 2d is valid")
	h.assert_true(not s3.is_key_valid("camera"), "key without 2d is invalid")
	s3.free()

	h.finish(self)

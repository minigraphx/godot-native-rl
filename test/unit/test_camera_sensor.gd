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

	# --- Grayscale path: white image -> L8, 1 channel, deterministic luminance ---
	# Pure white converts to L8 255 regardless of the luminance weights, so the exact
	# hex is stable: 4 pixels * 1 channel * 0xFF = "ffffffff".
	s.grayscale = true
	s.set_image_for_test(_make_image(2, 2, Image.FORMAT_RGB8, Color(1, 1, 1)))
	var gray_obs: String = s.get_observation()
	h.assert_eq(s.get_obs_shape(), [2, 2, 1], "grayscale obs_shape [H,W,1]")
	h.assert_eq(gray_obs, "ffffffff", "grayscale white obs == L8 255 x4")

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

	# --- get_image(): returns the raw captured Image (deploy path, no hex) ---
	var s4 = CameraSensor.new()
	var vp4 := SubViewport.new()
	vp4.size = Vector2i(2, 2)
	s4.viewport = vp4
	var src := _make_image(2, 2, Image.FORMAT_RGB8, Color(0, 1, 0))
	s4.set_image_for_test(src)
	var got: Image = s4.get_image()
	h.assert_true(got != null, "get_image returns the injected image")
	h.assert_eq(got.get_width(), 2, "get_image width")
	h.assert_eq(got.get_height(), 2, "get_image height")
	s4.free()
	vp4.free()

	# Missing viewport and no capture fn -> null (no crash).
	var s5 = CameraSensor.new()
	h.assert_true(s5.get_image() == null, "get_image with no viewport/capture -> null")
	s5.free()

	h.finish(self)

extends SceneTree

const Harness = preload("res://test/harness.gd")
const CameraObsMath = preload("res://addons/godot_native_rl/sensors/camera_obs_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# channels: RGB vs grayscale
	h.assert_eq(CameraObsMath.channels(false), 3, "channels RGB == 3")
	h.assert_eq(CameraObsMath.channels(true), 1, "channels grayscale == 1")

	# obs_shape is HWC: [height, width, channels]
	h.assert_eq(CameraObsMath.obs_shape(4, 2, false), [2, 4, 3], "obs_shape RGB [H,W,3]")
	h.assert_eq(CameraObsMath.obs_shape(4, 2, true), [2, 4, 1], "obs_shape grayscale [H,W,1]")

	# encode_image_bytes -> lowercase hex of raw bytes
	var bytes := PackedByteArray([0xAB, 0x01, 0x00, 0xFF])
	h.assert_eq(CameraObsMath.encode_image_bytes(bytes), "ab0100ff", "encode_image_bytes hex")
	h.assert_eq(CameraObsMath.encode_image_bytes(PackedByteArray()), "", "encode empty -> empty string")

	h.finish(self)

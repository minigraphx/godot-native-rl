class_name CameraSensor
extends Node

# Captures a SubViewport as a godot_rl-compatible image observation (raw uint8 HWC bytes,
# hex-encoded). Dimension-agnostic: the Camera2D/Camera3D lives inside the referenced
# SubViewport, not here. The live capture (viewport.get_texture().get_image()) needs a
# rendering context, so it is isolated behind _capture_fn for headless testing — inject a
# real Image with set_image_for_test. Composition into an agent's get_obs() is manual:
# obs[get_observation_key()] = get_observation(); merge get_obs_space_entry() into get_obs_space().

const CameraObsMath = preload("res://addons/godot_native_rl/sensors/camera_obs_math.gd")

@export var viewport: SubViewport = null
@export var grayscale: bool = false
# Must contain "2d" — godot_rl routes image obs on that substring.
@export var observation_key: String = "camera_2d"

# Test seam: a Callable() -> Image returning the captured frame. When null, the real
# viewport texture is read (only works with a rendering context, i.e. in-editor).
var _capture_fn = null
var _warned_no_viewport := false
var _validated_key := false

func _ready() -> void:
	if not is_key_valid(observation_key):
		push_error("CameraSensor: observation_key %r must contain \"2d\" (godot_rl routes image obs on that substring)." % observation_key)
	_validated_key = true

func set_capture_fn_for_test(fn: Callable) -> void:
	_capture_fn = fn

func set_image_for_test(img: Image) -> void:
	_capture_fn = func() -> Image: return img

func is_key_valid(key: String) -> bool:
	return key.contains("2d")

func get_observation_key() -> String:
	return observation_key

func get_obs_shape() -> Array:
	if viewport == null:
		return [0, 0, CameraObsMath.channels(grayscale)]
	return CameraObsMath.obs_shape(viewport.size.x, viewport.size.y, grayscale)

func get_obs_space_entry() -> Dictionary:
	return {"space": "box", "size": get_obs_shape()}

func get_observation() -> String:
	if viewport == null:
		if not _warned_no_viewport:
			push_warning("CameraSensor: no viewport set; returning empty observation.")
			_warned_no_viewport = true
		return ""
	_warned_no_viewport = false
	var img: Image = _capture()
	if img == null or img.is_empty():
		push_warning("CameraSensor: capture returned no image; returning empty observation.")
		return ""
	var target_format: int = Image.FORMAT_L8 if grayscale else Image.FORMAT_RGB8
	if img.get_format() != target_format:
		var converted: Image = img.duplicate()
		converted.convert(target_format)
		img = converted
	var bytes: PackedByteArray = img.get_data()
	var shape: Array = get_obs_shape()
	var expected: int = shape[0] * shape[1] * shape[2]
	if bytes.size() != expected:
		push_error("CameraSensor: byte count %d != expected %d for shape %s; returning empty." % [bytes.size(), expected, str(shape)])
		return ""
	return CameraObsMath.encode_image_bytes(bytes)

func _capture() -> Image:
	if _capture_fn != null:
		return _capture_fn.call()
	var tex: ViewportTexture = viewport.get_texture()
	if tex == null:
		return null
	return tex.get_image()

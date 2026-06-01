class_name CameraObsMath
extends RefCounted

# Pure, stateless helpers for camera (image) observations. No Node/Image capture here —
# that lives in CameraSensor. The wire format is godot_rl-compatible: raw uint8 bytes,
# HWC layout, hex-encoded. godot_rl decodes via np.frombuffer(bytes.fromhex(s), uint8).reshape(size).

static func channels(grayscale: bool) -> int:
	return 1 if grayscale else 3

# HWC order: matches Image.get_data() (row-major, channel-interleaved) and godot_rl's reshape(size).
static func obs_shape(width: int, height: int, grayscale: bool) -> Array:
	return [height, width, channels(grayscale)]

static func encode_image_bytes(bytes: PackedByteArray) -> String:
	return bytes.hex_encode()

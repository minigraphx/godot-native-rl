# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

const CameraSensor = preload("res://addons/godot_native_rl/sensors/camera_sensor.gd")

var _camera = null

func _ready() -> void:
	super._ready()
	_camera = CameraSensor.new()
	var vp := SubViewport.new()
	vp.size = Vector2i(2, 2)
	_camera.viewport = vp
	add_child(vp)
	add_child(_camera)
	# Inject a real all-red 2x2 RGB8 image (no rendering needed headless).
	var img := Image.create(2, 2, false, Image.FORMAT_RGB8)
	img.fill(Color(1, 0, 0))
	_camera.set_image_for_test(img)

func get_obs() -> Dictionary:
	return {"obs": [0.0, 0.0, 1.0, 0.0, 0.5], "camera_2d": _camera.get_observation()}

func get_obs_space() -> Dictionary:
	var space := NcnnControllerCore.obs_space_from_obs(get_obs())
	space[_camera.get_observation_key()] = _camera.get_obs_space_entry()
	return space

func get_reward() -> float:
	return reward

func get_info() -> Dictionary:
	return {"is_success": true}

func get_action_space() -> Dictionary:
	return {"move": {"size": 5, "action_type": "discrete"}}

func set_action(_action) -> void:
	pass

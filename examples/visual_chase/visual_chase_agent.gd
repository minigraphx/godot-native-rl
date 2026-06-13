class_name VisualChaseAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

# The chase task observed through PIXELS ONLY (#35): obs = a code-rasterized 36x36x3 frame on
# the camera_2d wire key (godot_rl maps "*2d" keys to uint8 Box -> SB3's CNN extractor).
# Movement/reward reuse the chase machinery wholesale; deploy uses the controllers' image route
# (get_inference_image() -> NcnnRunner.run_inference_image — item 36's glue, first trained user).

const ACTION_KEY := "move"
const ACTION_COUNT := 5
const IMG_W := 36
const IMG_H := 36
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
const ChaseObs = preload("res://examples/chase_the_target/chase_obs.gd")
const VObs = preload("res://examples/visual_chase/visual_chase_obs.gd")
# RewardAdapterScript is inherited from NcnnAIController2D — do not redeclare.

@export var game_path: NodePath
@export var step_penalty := 0.001
@export var touch_bonus := 1.0

var _game  # ChaseGame
var _action_index := 0

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("VisualChaseAgent: game_path not set — producing empty observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.distance, _game.max_distance, ["target_caught"]) \
		.add_event_bonus("target_caught", touch_bonus) \
		.add_step_penalty(step_penalty) \
		.build()
	var adapter := RewardAdapterScript.new()
	add_child(adapter)
	adapter.on_signal_event(_game, "target_caught", "target_caught")

func _frame_bytes() -> PackedByteArray:
	return VObs.rasterize(_game.get_agent_pos(), _game.get_target_pos(), _game.arena_size, IMG_W, IMG_H)

# --- godot_rl contract: image-only observation ---
func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func get_obs() -> Dictionary:
	if _game == null:
		return {"camera_2d": ""}
	return {"camera_2d": CameraObsMath.encode_image_bytes(_frame_bytes())}

func get_obs_space() -> Dictionary:
	return {"camera_2d": {"space": "box", "size": [IMG_H, IMG_W, 3]}}

# Deploy-side image route (item 36): the controller core calls this each decision and feeds the
# frame to NcnnRunner.run_inference_image instead of the float-vector path.
func get_inference_image() -> Image:
	if _game == null:
		return null
	return VObs.make_image(_frame_bytes(), IMG_W, IMG_H)

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "VisualChaseAgent: action %d out of range" % idx)
	_action_index = idx

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	var velocity: Vector2 = ChaseObs.action_index_to_velocity(_action_index, _game.move_speed)
	_game.move_agent(velocity, delta)
	accumulate_reward()
	if _game.distance() < _game.touch_radius:
		_game.relocate_target()
	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()

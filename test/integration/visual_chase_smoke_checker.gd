extends Node
# Integration smoke for the deployed visual-chase CNN (#35): the committed net loads, runs live
# through the image route (get_inference_image -> run_inference_image), and drives a RESPONSIVE
# policy — it emits several distinct discrete actions over the run (not stuck on one output).
#
# Why a smoke, not a catch-count behavioral gate: this is a DISCRETE-argmax CNN policy. ncnn runs
# convolutions in fp16 on ARM and fp32 on x86; the ~3-magnitude logit drift flips the argmax on a
# fraction of frames, so a full 1800-frame trajectory (and its catch count) diverges between
# architectures — locally (ARM) it catches 9-11/3600 frames under deterministic argmax, but CI
# (x86) takes a different categorical path. Continuous-control policies (quadruped) tolerate the
# same drift; a discrete argmax does not. Per-frame correctness IS asserted portably by
# test_visual_chase_golden_inference.gd (fixed frames, argmax, gap >= 3). This smoke covers the
# live wiring; the catch-count is a documented local result (see PR #35 / the example README).

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 600
@export var min_distinct_actions := 3

var _game
var _agent
var _frames := 0
var _seen := {}

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("could not resolve game/agent nodes")

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("trained ncnn model not loaded")
		return
	# The agent stores its last decoded discrete action index in _action_index.
	_seen[_agent._action_index] = true
	_frames += 1
	if _frames >= frames_to_run:
		var distinct := _seen.size()
		if distinct >= min_distinct_actions:
			print("VISUAL CHASE SMOKE PASSED (%d distinct actions over %d frames)" % [distinct, _frames])
			get_tree().quit(0)
		else:
			_fail("only %d distinct actions over %d frames (need %d) — policy not responding" % [distinct, _frames, min_distinct_actions])

func _fail(reason: String) -> void:
	printerr("VISUAL CHASE SMOKE FAILED: %s" % reason)
	get_tree().quit(1)

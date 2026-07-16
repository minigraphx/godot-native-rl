extends Node
# Trained GridWorld MASKED behavioral check (#385): the SAME shipped ncnn net now runs under the
# agent's action mask (GridWorldAgent.get_action_mask() → {"move": _game.current_action_mask()},
# off-grid moves masked; ActionDecode sets masked logits to -6e4 before argmax). The load-bearing
# property proven here: the deployed net NEVER selects a masked action. Goals are a sanity floor —
# masking must not lobotomize goal-reaching — but the zero-violations property is the point.
#
# Timing (see brief): inference_step fires synchronously inside the agent's super._physics_process()
# BEFORE gridworld_agent.gd applies the move (_game.move_agent() runs afterward). So inside the
# handler _game is still at the pre-move cell, and current_action_mask() is exactly the mask that
# gated this decision.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1800
@export var min_goals := 4  ## masking may shift trajectories vs the unmasked check's 5; floor, not the point
@export var max_pit_ratio := 0.5  ## pits_hit must stay under half the goals reached
## Anti-livelock (#293/#300): if no terminal lands within this window, reseed the episode.
@export var episode_timeout_frames := 300

var _game
var _agent
var _frames := 0
var _last_terminals := 0
var _since_terminal := 0
var _timeouts := 0
var _decisions := 0
var _violation := ""

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("missing game/agent")
		return
	if not _agent.has_signal("inference_step"):
		_fail("agent has no inference_step signal — cannot verify masking")
		return
	_agent.inference_step.connect(_on_inference_step)
	_game.seed_rng(3)
	# The game's _ready already rolled the FIRST layout from the unseeded RNG (tree order: world
	# before checker) — re-roll from the seeded stream (#300, the #298 chase gotcha).
	_game.reset_episode()

# Fires once per decision, pre-move (see timing note above). The mask read here is the one that
# gated the decoded action, so any masked/oob pick is a real violation of the deploy contract.
func _on_inference_step(payload: Dictionary) -> void:
	var mask: Array = _game.current_action_mask()
	var move: int = int(payload["action"]["move"])
	_decisions += 1
	if _violation != "":
		return
	if move < 0 or move >= mask.size() or int(mask[move]) == 0:
		_violation = "decision %d chose masked/oob action %d under mask %s" % [_decisions, move, str(mask)]

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("ncnn model not loaded")
		return
	_frames += 1
	var terminals: int = _game.goals_reached + _game.pits_hit
	if terminals != _last_terminals:
		_last_terminals = terminals
		_since_terminal = 0
	else:
		_since_terminal += 1
		if _since_terminal >= episode_timeout_frames:
			_timeouts += 1
			_since_terminal = 0
			_game.reset_episode()
	if _frames >= frames_to_run:
		if _violation != "":
			_fail(_violation)
			return
		if _decisions <= 0:
			_fail("0 decisions observed — inference_step never fired (false green)")
			return
		var goals: int = _game.goals_reached
		var pits: int = _game.pits_hit
		if goals < min_goals:
			_fail("only %d goals in %d frames (need %d; %d episode timeouts, %d decisions)" % [goals, _frames, min_goals, _timeouts, _decisions])
			return
		if float(pits) > float(goals) * max_pit_ratio:
			_fail("%d pits vs %d goals — not avoiding hazards (%d episode timeouts)" % [pits, goals, _timeouts])
			return
		print("TRAINED GRIDWORLD MASKED PASSED (%d decisions, 0 violations, %d goals, %d pits, %d episode timeouts in %d frames)" % [_decisions, goals, pits, _timeouts, _frames])
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("TRAINED GRIDWORLD MASKED FAILED: %s" % reason)
	get_tree().quit(1)

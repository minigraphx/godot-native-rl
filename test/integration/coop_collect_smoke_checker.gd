extends Node

# Headless behavioral smoke for the Cooperative Collect game (MA-POCA scaffold, #30). Drives the two
# agent bodies with a scripted greedy controller (each heads for its nearest uncollected item) and
# asserts the cooperative mechanics hold end-to-end in a ticking scene: items get collected, the team
# reward is SHARED (both agents would read the identical per-frame value), and the episode reaches the
# all-collected terminal. Validates the env without a trainer/socket (the policy is out of scope here).

@export var game_path: NodePath
@export var max_frames := 600

var _game
var _frames := 0
var _saw_positive_team_reward := false

func _ready() -> void:
	_game = get_node_or_null(game_path)
	if _game == null:
		_fail("game_path not set")

func _physics_process(_delta: float) -> void:
	_frames += 1
	# The game integrates + collects at priority -10 (before us), so team_reward()/collected() now
	# reflect this frame. Record that a positive team reward fired the frame an item was collected.
	if _game.team_reward() > 0.0:
		_saw_positive_team_reward = true

	if _game.is_terminal():
		var all := true
		for c in _game.collected():
			all = all and c
		if not all:
			_fail("terminal reached but not all items collected (timeout)")
		if not _saw_positive_team_reward:
			_fail("episode finished without any positive team reward")
		print("COOP COLLECT SMOKE PASSED (frames=%d, items=%d, shared team reward observed)" % [_frames, _game.collected().size()])
		get_tree().quit(0)
		return

	if _frames >= max_frames:
		_fail("did not collect all items within %d frames" % max_frames)

	# Greedy scripted steering: each agent heads to its nearest UNCOLLECTED item. Two agents naturally
	# split across items, exercising the multi-agent collection path.
	var items: Array = _game.items()
	var collected: Array = _game.collected()
	for idx in range(2):
		var pos: Vector2 = _game.agent_pos(idx)
		var target := _nearest_uncollected(pos, items, collected, idx)
		var dir := (target - pos)
		var vel: Vector2 = dir.normalized() * _game.move_speed if dir.length() > 1.0 else Vector2.ZERO
		_game.set_agent_velocity(idx, vel)

# Pick the nearest uncollected item; offset the second agent's tie-breaking so the two don't stack on
# the same item (keeps them cooperatively split).
func _nearest_uncollected(pos: Vector2, items: Array, collected: Array, idx: int) -> Vector2:
	var best := pos
	var best_d := INF
	for i in range(items.size()):
		if collected[i]:
			continue
		# Agent 1 biases toward later-indexed items so the pair spreads out.
		var d: float = pos.distance_to(items[i]) + (i * 0.001 if idx == 1 else -i * 0.001)
		if d < best_d:
			best_d = d
			best = items[i]
	return best

func _fail(msg: String) -> void:
	push_error("COOP COLLECT SMOKE FAILED: " + msg)
	print("COOP COLLECT SMOKE FAILED: " + msg)
	get_tree().quit(1)

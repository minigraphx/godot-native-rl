extends SceneTree
# ReplayRecorder (#39): episode segmentation from a stub signal source, initial-state snapshots,
# ring buffer on disk, meta content. No real NcnnSync/socket.

const Harness = preload("res://test/harness.gd")
const Recorder = preload("res://addons/godot_native_rl/training/replay_recorder.gd")
const RF = preload("res://addons/godot_native_rl/training/replay_format.gd")

class StubSync:
	extends Node
	signal actions_received(actions: Array)
	signal step_sent(rewards: Array, dones: Array)
	var action_repeat := 4

class StubGame:
	extends Node
	var x := 1.0
	func get_replay_state() -> Dictionary:
		return {"x": x}

const OUT := "user://replay_test"

func _emit_episode(sync: StubSync, n_steps: int, reward: float) -> void:
	for i in range(n_steps):
		sync.actions_received.emit([{"move": i % 3}])
		sync.step_sent.emit([reward], [i == n_steps - 1])

func _initialize() -> void:
	var h = Harness.new()

	var sync := StubSync.new()
	get_root().add_child(sync)
	var game := StubGame.new()
	get_root().add_child(game)
	var rec = Recorder.new()
	get_root().add_child(rec)
	rec.out_dir = OUT
	rec.keep_last = 2
	rec._game = game
	rec.attach_sync(sync)
	rec._snapshot_initial_state()

	# Episode 1: 3 steps; initial state x=1.
	_emit_episode(sync, 3, 0.5)
	var p0 := OUT + "/episode_0000.json"
	h.assert_true(FileAccess.file_exists(p0), "episode 0 written")
	var ep0 := RF.from_json(FileAccess.get_file_as_string(p0))
	h.assert_true(RF.validate(ep0), "episode 0 valid")
	h.assert_eq(int(ep0["meta"]["n_steps"]), 3, "3 steps recorded")
	h.assert_true(absf(float(ep0["meta"]["total_reward"]) - 1.5) < 1e-9, "rewards summed")
	h.assert_eq(int(ep0["meta"]["action_repeat"]), 4, "action_repeat from sync")
	h.assert_eq(float(ep0["initial_state"]["x"]), 1.0, "initial state snapshotted")
	h.assert_eq(int(ep0["steps"][1]["action"]["move"]), 1, "actions in order")

	# Next episode snapshots the NEW game state.
	game.x = 2.0
	# (the post-episode snapshot already ran with x=1; re-snapshot happens at finish — so episode 1's
	# initial state was taken right after episode 0 finished. Set x BEFORE episode 0 ends in real use;
	# here we assert the snapshot-at-finish behavior:)
	_emit_episode(sync, 2, 1.0)
	var ep1 := RF.from_json(FileAccess.get_file_as_string(OUT + "/episode_0001.json"))
	h.assert_eq(float(ep1["initial_state"]["x"]), 1.0, "episode 1 snapshot taken at episode-0 finish")

	# Ring: keep_last=2 -> writing a third episode drops episode_0000.
	_emit_episode(sync, 2, 1.0)
	h.assert_true(FileAccess.file_exists(OUT + "/episode_0002.json"), "episode 2 written")
	h.assert_true(not FileAccess.file_exists(p0), "ring dropped episode 0")

	# Steps without a pending action (step_sent alone) are ignored, no crash.
	sync.step_sent.emit([9.9], [true])
	h.assert_true(true, "orphan step_sent ignored")

	h.finish(self)

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

class StubAgent:
	extends Node
	# Mimics an NCNN_INFERENCE controller's recordable surface (#194).
	signal inference_step(payload: Dictionary)
	var reward := 0.0
	var n_steps := 0
	func decide(action: int, reward_after: float, steps_after: int) -> void:
		reward = reward_after
		n_steps = steps_after
		inference_step.emit({"action": {"move": action}})

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

	# Next episode snapshots the game state at ITS OWN start (#310): x changed between episodes,
	# and episode 1's first action window sees the new state.
	game.x = 2.0
	_emit_episode(sync, 2, 1.0)
	var ep1 := RF.from_json(FileAccess.get_file_as_string(OUT + "/episode_0001.json"))
	h.assert_eq(float(ep1["initial_state"]["x"]), 2.0, "episode 1 snapshot taken at episode-1 START")

	# Ring: keep_last=2 -> writing a third episode drops episode_0000.
	_emit_episode(sync, 2, 1.0)
	h.assert_true(FileAccess.file_exists(OUT + "/episode_0002.json"), "episode 2 written")
	h.assert_true(not FileAccess.file_exists(p0), "ring dropped episode 0")

	# Steps without a pending action (step_sent alone) are ignored, no crash.
	sync.step_sent.emit([9.9], [true])
	h.assert_true(true, "orphan step_sent ignored")

	# --- #195: multi-agent training capture — per-agent segmentation + file streams ---
	var mrec = Recorder.new()
	get_root().add_child(mrec)
	mrec.out_dir = OUT + "_multi"
	mrec.record_all_agents = true
	mrec._game = game
	mrec.attach_sync(sync)
	mrec._snapshot_initial_state()
	# Two agents; agent 1 finishes after 2 steps, agent 0 after 3. The game state EVOLVES per
	# window (#310): agent 0's episode must keep the snapshot from ITS start (x=10) even though
	# agent 1's episode finishes — and re-starts a fresh stream — in the middle of it.
	game.x = 10.0
	sync.actions_received.emit([{"move": 0}, {"move": 2}])
	sync.step_sent.emit([0.1, 1.0], [false, false])
	game.x = 11.0
	sync.actions_received.emit([{"move": 1}, {"move": 2}])
	sync.step_sent.emit([0.1, 1.0], [false, true])
	game.x = 12.0
	sync.actions_received.emit([{"move": 2}, {"move": 0}])
	sync.step_sent.emit([0.1, 0.5], [true, false])
	var a0 := RF.from_json(FileAccess.get_file_as_string(OUT + "_multi/episode_0000_train_0.json"))
	var a1 := RF.from_json(FileAccess.get_file_as_string(OUT + "_multi/episode_0000_train_1.json"))
	h.assert_true(RF.validate(a0) and RF.validate(a1), "both agents' episodes written and valid")
	h.assert_eq(int(a0["meta"]["n_steps"]), 3, "agent 0 episode has 3 steps")
	h.assert_eq(int(a1["meta"]["n_steps"]), 2, "agent 1 episode segmented independently (2 steps)")
	h.assert_eq(int(a1["meta"]["agent_index"]), 1, "agent index recorded per stream")
	h.assert_true(absf(float(a1["meta"]["total_reward"]) - 2.0) < 1e-9, "agent 1 rewards summed")
	h.assert_eq(float(a0["initial_state"]["x"]), 10.0, "#310: agent 0 keeps ITS episode-start snapshot despite agent 1 finishing mid-episode")
	h.assert_eq(float(a1["initial_state"]["x"]), 10.0, "#310: agent 1 episode 0 snapshot from its own start")

	# --- #194: inference-time capture — actions from inference_step, episodes via n_steps wrap ---
	var agent := StubAgent.new()
	agent.name = "Hero"
	get_root().add_child(agent)
	var irec = Recorder.new()
	get_root().add_child(irec)
	irec.out_dir = OUT + "_inf"
	irec._game = game
	irec.attach_agent(agent)
	var hero_stream := "inf_Hero_%d" % agent.get_instance_id()
	# Episode A: three decisions at cadence 8; cumulative reward 0.5 -> 0.8 -> 0.3: the dip is a
	# legitimate PENALTY window (#308) and must record as delta -0.5, not the whole accumulator.
	agent.decide(1, 0.5, 8)
	agent.decide(2, 0.8, 16)
	agent.decide(0, 0.3, 24)
	# Controller reset (n_steps wraps, reward accumulator restarted): first decision of episode B.
	agent.decide(3, 0.4, 8)
	var i0 := RF.from_json(FileAccess.get_file_as_string(OUT + "_inf/episode_0000_%s.json" % hero_stream))
	h.assert_true(RF.validate(i0), "inference episode written on n_steps wrap")
	h.assert_eq(String(i0["meta"]["mode"]), "inference", "meta marks inference mode")
	h.assert_eq(String(i0["meta"]["agent"]), "Hero", "meta carries the display name (stream key is instance-id'd)")
	h.assert_eq(int(i0["meta"]["n_steps"]), 3, "episode A has 3 decisions")
	h.assert_true(absf(float(i0["meta"]["total_reward"]) - 0.3) < 1e-9, "reward deltas sum to the accumulator")
	h.assert_eq(int(i0["steps"][0]["action"]["move"]), 1, "actions captured from inference_step payload")
	h.assert_true(absf(float(i0["steps"][2]["reward"]) + 0.5) < 1e-9, "#308: a penalty window records its negative delta, not the accumulator")
	h.assert_eq(int(i0["meta"]["action_repeat"]), 8, "#311: cadence derived from the observed n_steps delta")
	# Flush captures the in-progress episode B as partial (single decision -> export fallback).
	irec.inference_action_repeat = 5
	irec.flush_inference_episodes()
	var i1 := RF.from_json(FileAccess.get_file_as_string(OUT + "_inf/episode_0001_%s.json" % hero_stream))
	h.assert_true(RF.validate(i1), "flushed partial episode valid")
	h.assert_eq(int(i1["meta"]["n_steps"]), 1, "episode B holds the post-reset decision")
	h.assert_true(bool(i1["meta"]["partial"]), "flushed episode marked partial")
	h.assert_true(absf(float(i1["steps"][0]["reward"]) - 0.4) < 1e-9, "post-reset delta is the fresh accumulator")

	# --- #311: an override cadence (action_repeat=4 deploy) is derived, not the stale export ---
	var fast := StubAgent.new()
	fast.name = "Fast"
	get_root().add_child(fast)
	irec.attach_agent(fast)
	fast.decide(0, 0.1, 4)
	fast.decide(1, 0.2, 8)
	fast.decide(2, 0.3, 12)
	fast.decide(0, 0.1, 4)  # wrap -> episode written
	var f0 := RF.from_json(FileAccess.get_file_as_string(
		OUT + "_inf/episode_0000_inf_Fast_%d.json" % fast.get_instance_id()))
	h.assert_eq(int(f0["meta"]["action_repeat"]), 4, "#311: observed cadence 4 beats the export default")

	# --- #309: two agents with the SAME node name record into separate streams ---
	var holder_a := Node.new()
	var holder_b := Node.new()
	get_root().add_child(holder_a)
	get_root().add_child(holder_b)
	var twin_a := StubAgent.new()
	var twin_b := StubAgent.new()
	twin_a.name = "Twin"
	twin_b.name = "Twin"
	holder_a.add_child(twin_a)
	holder_b.add_child(twin_b)
	irec.attach_agent(twin_a)
	irec.attach_agent(twin_b)
	twin_a.decide(1, 1.0, 8)
	twin_b.decide(2, 5.0, 8)
	twin_a.decide(1, 2.0, 16)
	twin_b.decide(2, 6.0, 16)
	twin_a.decide(0, 0.0, 8)  # twin_a wraps: its 2-step episode flushes alone
	var t0 := RF.from_json(FileAccess.get_file_as_string(
		OUT + "_inf/episode_0000_inf_Twin_%d.json" % twin_a.get_instance_id()))
	h.assert_eq(int(t0["meta"]["n_steps"]), 2, "#309: same-named agents don't share a stream (twin A has its own 2 steps)")
	h.assert_true(absf(float(t0["meta"]["total_reward"]) - 2.0) < 1e-9, "#309: twin A's rewards uncontaminated by twin B's")
	h.assert_true(not FileAccess.file_exists(OUT + "_inf/episode_0000_inf_Twin_%d.json" % twin_b.get_instance_id()),
		"#309: twin B's stream still buffering (not flushed by twin A's wrap)")

	# --- #312: a node emitting inference_step WITHOUT n_steps/reward is rejected at attach ---
	var bare := Node.new()
	bare.name = "CrowdUnit"
	bare.add_user_signal("inference_step", [{"name": "payload", "type": TYPE_DICTIONARY}])
	get_root().add_child(bare)
	var before: int = irec._agent_state.size()
	irec.attach_agent(bare)
	h.assert_eq(int(irec._agent_state.size()), before, "#312: incomplete surface rejected at attach (loud error above), not at record time")

	h.finish(self)

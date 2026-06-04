extends SceneTree

const Harness = preload("res://test/harness.gd")
const DemoRecorder = preload("res://addons/godot_native_rl/training/demo_recorder.gd")

func _initialize() -> void:
	var h := Harness.new()
	var r := DemoRecorder.new()

	# Two non-terminal steps, then a terminal step => one trajectory, 3 obs / 2 acts.
	r.record_step([0.0, 1.0], [1.0], false)
	r.record_step([0.1, 1.1], [2.0], false)
	r.record_step([0.2, 1.2], [9.0], true)  # terminal: obs kept, action dropped
	h.assert_eq(r.trajectory_count(), 1, "terminal step finalizes one trajectory")
	h.assert_eq(r.step_count(), 2, "two actions recorded (terminal action dropped)")

	# gnrl_v1 envelope.
	var parsed = JSON.parse_string(
		r.to_json("gnrl_v1", {"move": {"size": 5, "action_type": "discrete"}}))
	h.assert_eq(parsed["format_version"], "gnrl_v1", "envelope carries format_version")
	h.assert_true(parsed.has("action_space"), "envelope carries action_space")
	var traj = parsed["demo_trajectories"][0]
	h.assert_eq(traj[0].size(), 3, "obs list keeps terminal frame (len acts + 1)")
	h.assert_eq(traj[1].size(), 2, "acts list excludes terminal action")

	# Legacy godot_rl format is the bare trajectory array.
	var bare = JSON.parse_string(r.to_json("godot_rl", {}))
	h.assert_true(bare is Array, "godot_rl format is a bare top-level array")
	h.assert_eq(bare.size(), 1, "one trajectory in bare array")

	# Input arrays must not be aliased: mutating the caller's array after record_step
	# must not change recorded data.
	var obs_in := [5.0]
	var act_in := [6.0]
	r.record_step(obs_in, act_in, false)
	obs_in[0] = 999.0
	act_in[0] = 999.0
	r.record_step([7.0], [8.0], true)
	var t2 = JSON.parse_string(r.to_json("godot_rl", {}))[1]
	h.assert_eq(t2[1][0][0], 6.0, "recorded action is a copy, not aliased to caller")
	h.assert_eq(t2[0][0][0], 5.0, "recorded obs is a copy, not aliased to caller")

	# Degenerate 1-frame episode: terminal on the very first step -> 1 obs, 0 acts.
	var r1 := DemoRecorder.new()
	r1.record_step([3.0], [4.0], true)
	h.assert_eq(r1.trajectory_count(), 1, "1-frame episode finalizes a trajectory")
	h.assert_eq(r1.step_count(), 0, "1-frame episode records zero actions")
	var solo = JSON.parse_string(r1.to_json("godot_rl", {}))[0]
	h.assert_eq(solo[0].size(), 1, "1-frame episode keeps the single obs")
	h.assert_eq(solo[1].size(), 0, "1-frame episode has no actions")

	# remove_last_episode pops; guarded on empty.
	r.remove_last_episode()
	r.remove_last_episode()
	h.assert_eq(r.trajectory_count(), 0, "remove_last_episode pops both, then guards empty")
	r.remove_last_episode()  # must not crash when empty

	h.finish(self)

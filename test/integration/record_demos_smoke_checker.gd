extends Node
# Drives the RECORD_EXPERT_DEMOS scene headlessly: waits until the recorder has enough
# completed trajectories, saves, re-loads the file, asserts gnrl_v1 shape, and quits.
# Save path / trajectory count are overridable via user cmdline args so the same scene
# both runs the suite smoke (user://) and generates the committed sample (res://).

@export var sync_path: NodePath
@export var save_path: String = "user://chase_demos_smoke.json"
@export var target_trajectories: int = 2
@export var max_frames_to_run := 600  # hard ceiling so a regression fails loudly, not hangs

var _sync = null
var _done := false
var _frames := 0

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS  # tick through NcnnSync's ~1s startup pause
	_sync = get_node_or_null(sync_path)
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--demo-out="):
			save_path = arg.substr("--demo-out=".length())
		elif arg.begins_with("--demo-trajectories="):
			target_trajectories = arg.substr("--demo-trajectories=".length()).to_int()
	if _sync == null:
		_fail("sync_path '%s' did not resolve to a node" % str(sync_path))
		return
	_sync.expert_demo_save_path = save_path

func _physics_process(_delta) -> void:
	if _done or _sync == null or _sync._recorder == null:
		return
	_frames += 1
	if _frames > max_frames_to_run:
		_done = true
		_fail("timed out waiting for %d trajectories" % target_trajectories)
		return
	if _sync._recorder.trajectory_count() < target_trajectories:
		return
	_done = true
	_sync.save_expert_demos()
	_verify_and_quit()

func _verify_and_quit() -> void:
	var text := FileAccess.get_file_as_string(ProjectSettings.globalize_path(save_path))
	if text.is_empty():
		_fail("output file not found at '%s'" % save_path)
		return
	var parsed = JSON.parse_string(text)
	var ok: bool = parsed is Dictionary \
		and parsed.get("format_version") == "gnrl_v1" \
		and parsed.get("demo_trajectories", []).size() >= target_trajectories
	if ok:
		var traj = parsed["demo_trajectories"][0]
		ok = traj[0].size() == traj[1].size() + 1  # obs keeps the terminal frame
	if not ok:
		_fail("recorded demo file failed gnrl_v1 shape check")
		return
	print("RECORD DEMOS SMOKE PASSED (%d trajectories)" % parsed["demo_trajectories"].size())
	get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("RECORD DEMOS SMOKE FAILED: %s" % reason)
	get_tree().quit(1)

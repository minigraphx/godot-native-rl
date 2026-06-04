extends RefCounted
# Pure accumulator for expert demonstrations in godot_rl trajectory layout.
# Each trajectory is [obs_list, acts_list] with len(obs_list) == len(acts_list) + 1
# (the terminal observation has no action), mirroring godot_rl_agents' recorder.
# No file I/O here — serialization returns a String; NcnnSync writes it.

const FORMAT_GNRL_V1 := "gnrl_v1"
const FORMAT_GODOT_RL := "godot_rl"

var _trajectories: Array = []      # Array of [obs_list, acts_list]
var _current: Array = [[], []]     # in-progress [obs_list, acts_list]

func record_step(obs: Array, action: Array, done: bool) -> void:
	_current[0].append(obs.duplicate())  # copy so callers can't mutate recorded data
	if done:
		_trajectories.append(_current.duplicate(true))
		_current[0] = []
		_current[1] = []
	else:
		_current[1].append(action.duplicate())

func remove_last_episode() -> void:
	if _trajectories.size() > 0:
		_trajectories.remove_at(_trajectories.size() - 1)

func trajectory_count() -> int:
	return _trajectories.size()

# Total recorded actions (transitions): every completed trajectory's acts_list plus the
# in-progress episode's acts_list so far.
func step_count() -> int:
	var n: int = _current[1].size()
	for traj in _trajectories:
		n += traj[1].size()
	return n

func to_json(demo_format: String, action_space: Dictionary) -> String:
	if demo_format == FORMAT_GODOT_RL:
		return JSON.stringify(_trajectories.duplicate(true), "", false)
	assert(demo_format == FORMAT_GNRL_V1,
		"DemoRecorder: unknown demo_format '%s'" % demo_format)
	var envelope := {
		"format_version": FORMAT_GNRL_V1,
		"action_space": action_space,
		"demo_trajectories": _trajectories.duplicate(true),
	}
	return JSON.stringify(envelope, "", false)

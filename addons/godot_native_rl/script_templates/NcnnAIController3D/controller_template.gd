# meta-name: NCNN AI Controller
# meta-description: Agent scaffold for Godot Native RL — implement the obs/reward/action contract
# meta-default: true
extends _BASE_

# The four methods below define your agent. Each stub fails loud (push_error) so a
# forgotten override surfaces immediately instead of silently training on garbage.

func get_obs() -> Dictionary:
	# Compose a flat float Array. With ISensor2D/3D children, auto-discovery
	# concatenates them for you — collect_sensors() returns the flat float Array:
	#     return {"obs": collect_sensors()}
	# Mix in your own features by appending to it before returning.
	push_error("get_obs() not implemented — return {\"obs\": [floats...]}")
	return {"obs": []}

func get_reward() -> float:
	# Return the reward accumulated since the last step — e.g. a RewardBuilder total,
	# or a hand-computed shaping term (distance delta, goal bonus, time penalty).
	push_error("get_reward() not implemented")
	return 0.0

func get_action_space() -> Dictionary:
	# Describe each action head. Examples:
	#     "move": {"size": 4, "action_type": "discrete"}     # one of 4 choices
	#     "steer": {"size": 2, "action_type": "continuous"}  # 2 floats in [-1, 1]
	push_error("get_action_space() not implemented")
	return {}

func set_action(action) -> void:
	# Apply the chosen action to your agent, e.g.:
	#     var idx := int(action["move"])
	#     velocity = DIRECTIONS[idx] * speed
	push_error("set_action() not implemented")

#func get_obs_space() -> Dictionary:
#	# Override only for complex obs (images, multiple keys). The base class derives
#	# {"obs": {"size": [len], "space": "box"}} from get_obs() automatically.
#	return super.get_obs_space()

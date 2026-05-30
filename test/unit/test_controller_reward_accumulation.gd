extends SceneTree

const Harness = preload("res://test/harness.gd")
const RewardBuilder = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
const RewardAdapter = preload("res://addons/godot_native_rl/reward/reward_adapter.gd")
const Stub = preload("res://test/unit/stub_agent.gd")

class Emitter:
	extends Node
	signal pinged

func _initialize() -> void:
	var h := Harness.new()

	# Agent with no reward_source and no adapters -> accumulate_reward() is a no-op.
	var plain := Stub.new()
	get_root().add_child(plain)
	plain.reward = 0.0
	plain.accumulate_reward()
	h.assert_eq(plain.reward, 0.0, "no reward_source + no adapters: accumulate is a no-op")
	plain.free()

	# Agent with a reward_source: accumulate_reward adds the evaluated reward.
	var agent := Stub.new()
	get_root().add_child(agent)
	agent.reward_source = RewardBuilder.new().add_alive_bonus(0.01).build()
	agent.reward = 0.0
	agent.accumulate_reward()
	h.assert_eq(agent.reward, 0.01, "accumulate adds reward_source.evaluate")

	# A child RewardAdapter's scalar reward is drained into the agent.
	var adapter := RewardAdapter.new()
	agent.add_child(adapter)
	var emitter := Emitter.new()
	get_root().add_child(emitter)
	adapter.on_signal(emitter, "pinged", 0.5)
	# collect_reward_adapters() is called automatically by the adapter's _ready(), but
	# _ready() is deferred in synchronous _initialize() test contexts; call it explicitly
	# here to prove the public API works (real game scenes get auto-registration for free).
	agent.collect_reward_adapters()
	emitter.pinged.emit()
	agent.reward = 0.0
	agent.accumulate_reward()
	# reward_source alive bonus (0.01) + drained adapter scalar (0.5) = 0.51
	h.assert_eq(agent.reward, 0.51, "accumulate also drains child adapters")

	agent.free()
	emitter.free()
	h.finish(self)

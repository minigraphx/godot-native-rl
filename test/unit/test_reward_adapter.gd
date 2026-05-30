extends SceneTree

const Harness = preload("res://test/harness.gd")
const RewardAdapter = preload("res://reward/reward_adapter.gd")
const Reward = preload("res://reward/reward.gd")
const EventBonusTerm = preload("res://reward/terms/event_bonus_term.gd")

# Emitters with signals of varying arity.
class Emitter0:
	extends Node
	signal boom
class Emitter1:
	extends Node
	signal hit(amount)
class Emitter2:
	extends Node
	signal pair(a, b)

func _initialize() -> void:
	var h := Harness.new()
	var root := Node.new()
	get_root().add_child(root)

	# on_signal: fire-and-forget scalar accumulation, drained on demand.
	var adapter := RewardAdapter.new()
	root.add_child(adapter)

	var e0 := Emitter0.new()
	var e1 := Emitter1.new()
	var e2 := Emitter2.new()
	root.add_child(e0)
	root.add_child(e1)
	root.add_child(e2)

	adapter.on_signal(e0, "boom", 1.0)
	adapter.on_signal(e1, "hit", 0.5)
	adapter.on_signal(e2, "pair", -0.25)

	e0.boom.emit()
	e1.hit.emit(99)        # 1-arg signal; handler ignores the arg
	e2.pair.emit(1, 2)     # 2-arg signal; handler ignores the args
	h.assert_eq(adapter.drain(), 1.25, "0/1/2-arg signals all accumulate (1.0+0.5-0.25)")
	h.assert_eq(adapter.drain(), 0.0, "drain resets accumulator")

	# on_signal_event: routes to the bound Reward's trigger_event.
	var bonus := EventBonusTerm.new("caught", 2.0)
	var reward := Reward.new([bonus])
	var adapter2 := RewardAdapter.new()
	root.add_child(adapter2)
	adapter2.bind_reward(reward)        # explicit binding (not a child of a controller here)

	var e1b := Emitter1.new()
	root.add_child(e1b)
	adapter2.on_signal_event(e1b, "hit", "caught")
	e1b.hit.emit(0)
	h.assert_eq(reward.evaluate(null), 2.0, "on_signal_event triggers the bound Reward event")

	root.free()
	h.finish(self)

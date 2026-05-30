extends SceneTree

const Harness = preload("res://test/harness.gd")
const RewardBuilder = preload("res://reward/reward_builder.gd")

class Holder:
	extends RefCounted
	var value := 10.0
	func dist() -> float:
		return value

func _initialize() -> void:
	var h := Harness.new()
	var holder := Holder.new()

	var base := RewardBuilder.new()
	var extended := base.add_step_penalty(0.001)

	# Immutability: add_* returns a NEW builder; the original is unchanged.
	h.assert_true(extended != base, "add_* returns a new builder instance")
	h.assert_eq(base.term_count(), 0, "original builder unchanged after add")
	h.assert_eq(extended.term_count(), 1, "new builder has the added term")

	# Full chain builds a working Reward with summed terms.
	var reward = RewardBuilder.new() \
		.add_progress_shaping(holder.dist, 100.0, ["caught"]) \
		.add_event_bonus("caught", 1.0) \
		.add_step_penalty(0.001) \
		.add_alive_bonus(0.01) \
		.build()

	# Step 1 after build: baseline primed at 10; move to 8 -> progress 0.02.
	holder.value = 8.0
	# total = progress(0.02) + bonus(0) - penalty(0.001) + alive(0.01) = 0.029
	h.assert_eq(reward.evaluate(null), 0.029, "built Reward sums all configured terms")

	# Event flows through the built Reward.
	holder.value = 50.0
	reward.trigger_event("caught")    # rebase progress baseline to 50, queue bonus
	holder.value = 48.0
	# total = progress((50-48)/100=0.02) + bonus(1.0) - 0.001 + 0.01 = 1.029
	h.assert_eq(reward.evaluate(null), 1.029, "events flow through built Reward")

	h.finish(self)

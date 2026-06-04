extends SceneTree

const Harness = preload("res://test/harness.gd")
const ChaseExpertAgent = preload("res://examples/chase_the_target/chase_expert_agent.gd")

func _initialize() -> void:
	var h := Harness.new()

	# expert_action_index maps the relative offset (target - agent) to a chase direction.
	# Indices match ChaseAgent.action_index_to_velocity: 1=up,2=down,3=left,4=right.
	h.assert_eq(ChaseExpertAgent.expert_action_index(Vector2(10.0, 1.0)), 4, "target right -> move right")
	h.assert_eq(ChaseExpertAgent.expert_action_index(Vector2(-10.0, 1.0)), 3, "target left -> move left")
	h.assert_eq(ChaseExpertAgent.expert_action_index(Vector2(1.0, 10.0)), 2, "target below -> move down")
	h.assert_eq(ChaseExpertAgent.expert_action_index(Vector2(1.0, -10.0)), 1, "target above -> move up")
	# Ties on |x| >= |y| pick the horizontal axis.
	h.assert_eq(ChaseExpertAgent.expert_action_index(Vector2(5.0, 5.0)), 4, "tie favors horizontal")

	h.finish(self)

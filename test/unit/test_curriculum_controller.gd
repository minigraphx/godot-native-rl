extends SceneTree
# CurriculumController: applies stage params to the game at episode boundaries, emits
# stage_changed, honors external control, loads stages from JSON, errors loud on a missing
# apply method. No NcnnSync/socket involved.
#
# NOTE: nodes added under get_root() in a SceneTree script do NOT get _ready() fired
# automatically (see memory/CLAUDE notes) — the controller's _ready() is idempotent and the
# test drives it manually after configuration.

const Harness = preload("res://test/harness.gd")
const Controller = preload("res://addons/godot_native_rl/training/curriculum_controller.gd")

class StubGame:
	extends Node
	var applied: Array = []
	func apply_curriculum(params: Dictionary) -> void:
		applied.append(params)

var _signal_log: Array = []

func _on_stage_changed(index: int, name: String, params: Dictionary) -> void:
	_signal_log.append([index, name, params])

func _stages() -> Array:
	return [
		{"name": "easy", "params": {"touch_radius": 120.0},
			"promote": {"metric": "mean_reward", "threshold": 5.0, "window": 2, "min_episodes": 2}},
		{"name": "hard", "params": {"touch_radius": 40.0}},
	]

func _initialize() -> void:
	var h = Harness.new()

	var game := StubGame.new()
	get_root().add_child(game)
	var ctrl = Controller.new()
	get_root().add_child(ctrl)
	ctrl.game_path = ctrl.get_path_to(game)
	ctrl.set_stages(_stages())
	ctrl.stage_changed.connect(_on_stage_changed)
	ctrl._ready()

	h.assert_true(ctrl.is_in_group("CURRICULUM"), "joins CURRICULUM group")
	h.assert_eq(game.applied.size(), 1, "initial stage params applied")
	h.assert_eq(game.applied[0]["touch_radius"], 120.0, "initial params are stage 0")

	# Two good episodes -> promotion applies stage 1 params at the SAME record boundary
	ctrl.record_episode(10.0, true)
	h.assert_eq(game.applied.size(), 1, "no promotion after 1 episode")
	ctrl.record_episode(10.0, true)
	h.assert_eq(game.applied.size(), 2, "promotion applied at episode boundary")
	h.assert_eq(game.applied[1]["touch_radius"], 40.0, "stage 1 params applied")
	h.assert_eq(_signal_log.size(), 1, "stage_changed emitted once")
	h.assert_eq(_signal_log[0][0], 1, "signal carries new index")
	h.assert_eq(ctrl.stage_index(), 1, "controller reports stage 1")

	# External control disables auto-promotion and supports direct jumps
	var ctrl2 = Controller.new()
	get_root().add_child(ctrl2)
	ctrl2.game_path = ctrl2.get_path_to(game)
	ctrl2.set_stages(_stages())
	ctrl2._ready()
	game.applied.clear()
	ctrl2.set_external_control(true)
	ctrl2.record_episode(100.0, true)
	ctrl2.record_episode(100.0, true)
	h.assert_eq(ctrl2.stage_index(), 0, "external control blocks auto-promotion")
	h.assert_true(ctrl2.jump_to_stage(1), "external jump works")
	h.assert_eq(game.applied.back()["touch_radius"], 40.0, "jump applied params")
	h.assert_true(not ctrl2.jump_to_stage(9), "out-of-range jump refused")

	# Direct params injection (trainer 'params' override)
	ctrl2.apply_external_params({"touch_radius": 7.0})
	h.assert_eq(game.applied.back()["touch_radius"], 7.0, "external params applied")

	# JSON loading
	var ctrl3 = Controller.new()
	get_root().add_child(ctrl3)
	ctrl3.game_path = ctrl3.get_path_to(game)
	ctrl3.stages_json_path = "res://test/unit/fixtures/curriculum_two_stage.json"
	ctrl3._ready()
	h.assert_eq(ctrl3.stage_count(), 2, "stages loaded from JSON")

	# Missing apply method: loud but not crashing
	var bare := Node.new()
	get_root().add_child(bare)
	var ctrl4 = Controller.new()
	get_root().add_child(ctrl4)
	ctrl4.game_path = ctrl4.get_path_to(bare)
	ctrl4.set_stages(_stages())
	ctrl4._ready()
	h.assert_true(true, "missing apply method did not crash")

	h.finish(self)

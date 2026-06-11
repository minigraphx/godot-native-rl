extends SceneTree
# Structure tests for the BallChase training scenes (#82). ball_chase_world.tscn is the
# replicable unit (game + agent, NO Sync); ball_chase_train.tscn composes one world + Sync;
# ball_chase_train_parallel.tscn tiles N worlds via ParallelArena2D + Sync. Scenes are
# instantiated WITHOUT entering the tree so _ready() (and Sync's trainer connection) never
# fires — exported properties are applied at instantiate(), which is all we assert on.

const Harness = preload("res://test/harness.gd")

const WORLD_PATH := "res://examples/ball_chase/ball_chase_world.tscn"
const TRAIN_PATH := "res://examples/ball_chase/ball_chase_train.tscn"
const PARALLEL_PATH := "res://examples/ball_chase/ball_chase_train_parallel.tscn"
const GAME_SCRIPT := "res://examples/ball_chase/ball_chase_game.gd"
const AGENT_SCRIPT := "res://examples/ball_chase/ball_chase_agent.gd"
const SYNC_SCRIPT := "res://addons/godot_native_rl/sync.gd"
const ARENA_SCRIPT := "res://addons/godot_native_rl/training/parallel_arena_2d.gd"

func _initialize() -> void:
	var h := Harness.new()
	_test_world(h)
	_test_train(h)
	h.finish(self)

func _script_path(node: Node) -> String:
	var s: Variant = node.get_script()
	return s.resource_path if s != null else ""

func _test_world(h) -> void:
	var packed := load(WORLD_PATH) as PackedScene
	h.assert_true(packed != null, "world scene loads")
	if packed == null:
		return
	var world := packed.instantiate()
	h.assert_eq(_script_path(world), GAME_SCRIPT, "world root runs BallChaseGame")
	h.assert_true(world.get_node_or_null("AgentBody") != null, "world has AgentBody")
	h.assert_true(world.get_node_or_null("Target") != null, "world has Target")
	var agent := world.get_node_or_null("BallChaseAgent")
	h.assert_true(agent != null, "world has BallChaseAgent")
	if agent != null:
		h.assert_eq(_script_path(agent), AGENT_SCRIPT, "agent runs BallChaseAgent script")
		h.assert_eq(agent.game_path, NodePath(".."), "agent game_path points at world root")
		h.assert_eq(agent.control_mode, 2, "agent control_mode matches the old train scene (2)")
	h.assert_true(world.get_node_or_null("Sync") == null, "world has NO Sync (replicable unit)")
	world.free()

func _test_train(h) -> void:
	var packed := load(TRAIN_PATH) as PackedScene
	h.assert_true(packed != null, "train scene loads")
	if packed == null:
		return
	var train := packed.instantiate()
	var world := train.get_node_or_null("BallChaseWorld")
	h.assert_true(world != null, "train scene instances the world sub-scene")
	if world != null:
		h.assert_eq(world.scene_file_path, WORLD_PATH, "world child comes from ball_chase_world.tscn")
	var sync := train.get_node_or_null("Sync")
	h.assert_true(sync != null, "train scene has Sync")
	if sync != null:
		h.assert_eq(_script_path(sync), SYNC_SCRIPT, "Sync runs NcnnSync")
		h.assert_eq(sync.control_mode, 1, "Sync control_mode = TRAINING (1)")
	train.free()

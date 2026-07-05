extends Node
# #313 regression: under ParallelArena2D tiling, each world's EntitySensor2D must see ONLY its
# own world's tiles. Scene-tree groups are global, so the pre-fix shared literal group leaked
# every world's tiles into every sensor — presence flags saturated to max_entities and neighbor
# tiles displaced local ones in the nearest-N. Two worlds are forced to DIFFERENT tile counts;
# each agent's flag sum must equal its own world's count.

const Harness = preload("res://test/harness.gd")

const MAX_ENTITIES := 6
const EXPECT_OBS := MAX_ENTITIES * 4 + MAX_ENTITIES

var _h = Harness.new()
var _frames := 0
var _configured := false

func _physics_process(_delta: float) -> void:
	_frames += 1
	if not _configured:
		# Worlds exist now (ParallelArena2D instanced them in its _ready): pin distinct counts.
		_configured = true
		var games := _games()
		_h.assert_eq(games.size(), 2, "two tiled worlds found")
		var count := 2
		for game in games:
			game.min_tiles = count
			game.max_tiles = count
			game.seed_rng(7)
			game.reset_episode()
			count += 2  # world 0 -> 2 tiles, world 1 -> 4 tiles
		return
	if _frames < 5:
		return
	for agent in get_tree().get_nodes_in_group("AGENT"):
		var game: Node = agent.get_node(agent.game_path)
		var obs: Array = agent.get_obs()["obs"]
		_h.assert_eq(obs.size(), EXPECT_OBS, "obs width fixed")
		var flag_sum := 0.0
		for i in range(EXPECT_OBS - MAX_ENTITIES, EXPECT_OBS):
			flag_sum += float(obs[i])
		_h.assert_eq(int(round(flag_sum)), int(game.tile_count()),
			"agent sees exactly its OWN world's %d tiles (flag sum %d — a leak would see both worlds')"
			% [int(game.tile_count()), int(round(flag_sum))])
	_h.finish(get_tree())

func _games() -> Array:
	var out: Array = []
	for agent in get_tree().get_nodes_in_group("AGENT"):
		out.append(agent.get_node(agent.game_path))
	return out

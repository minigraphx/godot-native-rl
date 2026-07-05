class_name SorterGame
extends Node2D
# Sorter env (#46 M2, spec §2): each episode spawns a VARIABLE number of numbered tiles
# (min_tiles..max_tiles); the agent must visit them in ascending order. A wrong-order visit
# penalizes (signal) on tile ENTER only and does NOT consume the tile; the episode ends when all
# tiles are visited in order (the agent's horizon owns the timeout). The variable count is the
# whole point: the policy sees the tiles through an EntitySensor2D block whose presence flags
# carry the count — the attention encoder's input contract.
#
# Tiles the sensor should see this episode are members of `tile_group` (inactive slots leave the
# group, so the candidate set matches the spawned count). Scene-tree groups are GLOBAL, so under
# ParallelArena2D tiling every world's tiles share it — per-world isolation is the SENSOR's job:
# the world scene sets EntitySensor2D.scope_root to this world's root (#336, generalizing the
# #313 fix out of per-example instance-group suffixing and into shared infra).

signal correct_visit   ## the next-in-order tile was entered (and consumed)
signal wrong_visit     ## a not-next tile was entered (not consumed)
signal all_sorted      ## every tile visited in order — episode solved

const SorterMath = preload("res://examples/sorter/sorter_math.gd")
const TileScript = preload("res://examples/sorter/sorter_tile.gd")

@export var arena_size := Vector2(1000, 600)
@export var move_speed := 300.0     ## read by the agent controller to scale velocity
@export var visit_radius := 45.0
@export var min_tiles := 2
@export var max_tiles := 6          ## must match the agent's EntitySensor2D max_entities
@export var edge_margin := 70.0
@export var agent_body_path: NodePath
@export var tile_group := "SORTER_TILES"

var episodes_solved := 0
var correct_visits := 0
var wrong_visits := 0

var _agent_body: Node2D
var _tiles: Array = []          # max_tiles pre-built tile nodes (active subset used per episode)
var _tile_count := 0
var _prev_overlaps: Array = []
var _rng := RandomNumberGenerator.new()


func _ready() -> void:
	_agent_body = get_node_or_null(agent_body_path)
	for i in range(max_tiles):
		var tile := Node2D.new()
		tile.set_script(TileScript)
		tile.name = "Tile%d" % (i + 1)
		add_child(tile)
		_tiles.append(tile)
	reset_episode()


## s must be a non-negative integer (RandomNumberGenerator.seed is uint64; negatives wrap).
func seed_rng(s: int) -> void:
	_rng.seed = s


func reset_episode() -> void:
	# Clamp to the PRE-BUILT slot count and iterate ALL slots below: max_tiles can be changed at
	# runtime (a curriculum, a test pinning the count), and iterating `range(max_tiles)` after a
	# shrink left the higher slots active AND in the sensor group with stale layouts — orphaned
	# ghost tiles the policy could see (caught by the #313 parallel smoke's flakiness).
	_tile_count = clampi(_rng.randi_range(min_tiles, max_tiles), 1, _tiles.size())
	var layout: Array = SorterMath.tile_layout(_rng, _tile_count, arena_size, edge_margin)
	_prev_overlaps = []
	for i in range(_tiles.size()):
		var tile: Node2D = _tiles[i]
		var live := i < _tile_count
		tile.active = live
		tile.visited = false
		tile.number = i + 1
		tile.total = _tile_count
		var group := StringName(tile_group)
		if live:
			tile.position = layout[i]
			if not tile.is_in_group(group):
				tile.add_to_group(group)
			_prev_overlaps.append(false)
		else:
			tile.position = Vector2(-10000, -10000)  # parked far outside; also out of the group
			if tile.is_in_group(group):
				tile.remove_from_group(group)
	if _agent_body != null:
		# Reject spawns inside a tile's visit radius (#315): _prev_overlaps starts all-false, so
		# spawning ON a tile would fire a phantom ENTER (free reward / undeserved penalty) on the
		# first step. Rejection (rather than priming the overlap state) keeps the enter semantics
		# clean — a primed spawn-on-target would stall a greedy policy that never re-enters.
		var spawned_clear := false
		for _attempt in range(20):
			var pos := Vector2(
				_rng.randf_range(edge_margin, arena_size.x - edge_margin),
				_rng.randf_range(edge_margin, arena_size.y - edge_margin))
			_agent_body.position = pos
			spawned_clear = true
			for i in range(_tile_count):
				if pos.distance_to(layout[i]) <= visit_radius + 5.0:
					spawned_clear = false
					break
			if spawned_clear:
				break
		if not spawned_clear:
			# 20 blocked draws (vanishingly rare at shipped params; reachable with a tiny arena /
			# huge radius): keep the last position but PRIME the overlap state, or the first step
			# fires exactly the phantom ENTER the rejection loop exists to prevent (#333).
			push_warning("SorterGame: could not place the agent clear of all tiles after 20 attempts — priming overlap state instead.")
			var tile_positions: Array = []
			for i in range(_tile_count):
				tile_positions.append(layout[i])
			_prev_overlaps = SorterMath.overlap_flags(_agent_body.position, tile_positions, visit_radius)


func move_agent(velocity: Vector2, delta: float) -> void:
	if _agent_body == null:
		return
	var p := _agent_body.position + velocity * delta
	_agent_body.position = Vector2(clampf(p.x, 0.0, arena_size.x), clampf(p.y, 0.0, arena_size.y))
	_process_visits()


func _process_visits() -> void:
	var tile_positions: Array = []
	for i in range(_tile_count):
		tile_positions.append((_tiles[i] as Node2D).position)
	var overlaps: Array = SorterMath.overlap_flags(_agent_body.position, tile_positions, visit_radius)
	var visited := _visited_flags()
	for idx in SorterMath.entered_tiles(_prev_overlaps, overlaps):
		if visited[idx]:
			continue  # standing on / re-entering an already-sorted tile is neutral
		if idx + 1 == SorterMath.next_target(visited):
			# The tiles ARE the single source of truth (#317/#333): write there, re-derive the
			# snapshot — no shadow mutation to keep in lockstep.
			(_tiles[idx] as Node2D).visited = true
			visited = _visited_flags()
			correct_visits += 1
			correct_visit.emit()
			if SorterMath.all_visited(visited):
				episodes_solved += 1
				all_sorted.emit()
		else:
			wrong_visits += 1
			wrong_visit.emit()
	_prev_overlaps = overlaps


## The per-tile `visited` bools ARE the single source of truth (the sensor obs reads them via
## get_entity_features); this derives the flat array SorterMath consumes — a second stored copy
## previously had to be kept in lockstep by hand (#317).
func _visited_flags() -> Array:
	var out: Array = []
	for i in range(_tile_count):
		out.append(bool((_tiles[i] as Node2D).visited))
	return out


func tile_count() -> int:
	return _tile_count


func next_target() -> int:
	return SorterMath.next_target(_visited_flags())


func get_agent_pos() -> Vector2:
	return _agent_body.position if _agent_body != null else Vector2.ZERO


func tile_position(i: int) -> Vector2:
	return (_tiles[i] as Node2D).position


# --- Lightweight visualizer (chase pattern; free headless) ---
func _process(_delta: float) -> void:
	queue_redraw()


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.10, 0.11, 0.15), true)
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.30, 0.32, 0.42), false, 2.0)
	var nxt := next_target()
	var font := ThemeDB.fallback_font
	for i in range(_tile_count):
		var tile: Node2D = _tiles[i]
		var color := Color(0.25, 0.75, 0.35) if tile.visited else \
			(Color(0.95, 0.85, 0.30) if i + 1 == nxt else Color(0.45, 0.50, 0.65))
		draw_circle(tile.position, visit_radius, Color(color, 0.25))
		draw_circle(tile.position, 16.0, color)
		draw_string(font, tile.position + Vector2(-5, 6), str(i + 1), HORIZONTAL_ALIGNMENT_CENTER, -1, 18, Color.BLACK)
	draw_circle(get_agent_pos(), 12.0, Color(0.3, 0.8, 1.0))

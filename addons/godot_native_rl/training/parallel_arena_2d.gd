class_name ParallelArena2D
extends Node2D

## 2D sibling of ParallelArena (Node3D): tiles N copies of a 2D agent "world" sub-scene in one
## shared space so a single Godot process trains many agents at once. NcnnSync collects every
## AGENT-group node the worlds spawn; godot-rl auto-detects n_agents and vectorizes over them. A
## hide & seek world spawns 2 agents, so N worlds -> 2N agents under one shared policy. Isolation is
## spatial: worlds sit on a square XY grid `spacing` units apart (must exceed a world's reach).

@export var world_scene: PackedScene  ## world to replicate (its AGENT-group agents must be tile-offset-safe)
@export var count: int = 8            ## number of parallel worlds
@export var spacing: float = 1400.0   ## distance between tile origins (must exceed arena extent + ray_length)

func _ready() -> void:
	if world_scene == null:
		push_error("ParallelArena2D: world_scene is not set — nothing to spawn.")
		return
	if count < 1:
		push_warning("ParallelArena2D: count < 1 (%d) — nothing to spawn." % count)
		return
	var cols := _cols()
	for i in range(count):
		var world: Node2D = world_scene.instantiate()
		world.position = tile_offset(i, spacing, cols)
		add_child(world)

func _cols() -> int:
	return int(ceil(sqrt(float(count))))

# Lays tiles in a roughly-square grid on the XY plane. Pure + unit-tested.
static func tile_offset(index: int, spacing: float, cols: int) -> Vector2:
	if cols < 1:
		return Vector2.ZERO
	return Vector2((index % cols) * spacing, (index / cols) * spacing)

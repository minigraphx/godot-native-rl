extends Node
# Live race leaderboard (#60 M4): each frame reads every lane creature's forward distance, ranks
# them, and writes a standings + timer string to a HUD Label. Lanes are independent creatures in one
# physics space, each driven by its own trained net (different training generations) — so the board
# shows the learning arc as a race. Ranking math is the pure, unit-tested race_math.gd.

const R = preload("res://examples/quadruped_walk/race_math.gd")

@export var lane_game_paths: Array[NodePath] = []
@export var lane_labels: Array[String] = []
@export var label_path: NodePath          ## a Label to write standings into (optional)
@export var finish_z := 40.0

var _games: Array = []
var _label: Label
var _elapsed := 0.0

func _ready() -> void:
	for p in lane_game_paths:
		_games.append(get_node_or_null(p))
	_label = get_node_or_null(label_path) as Label

func _lane_distance(i: int) -> float:
	# Forward progress = torso Z minus the lane's own start Z (lanes share Z=0 start; offset is X).
	if i < _games.size() and _games[i] != null:
		return _games[i].torso_pos().z
	return 0.0

func distances() -> Array:
	var out: Array = []
	for i in range(_games.size()):
		out.append(_lane_distance(i))
	return out

func _physics_process(delta: float) -> void:
	_elapsed += delta
	if _label == null:
		return
	var d := distances()
	var order := R.standings(d)
	var lines: Array = ["RACE  t=%.1fs" % _elapsed]
	for rank in range(order.size()):
		var lane: int = order[rank]
		var name: String = lane_labels[lane] if lane < lane_labels.size() else ("lane %d" % lane)
		var tag := name
		if R.finished(d[lane], finish_z):
			tag += "  [FINISH]"
		lines.append(R.format_row(rank + 1, tag, d[lane]))
	_label.text = "\n".join(lines)

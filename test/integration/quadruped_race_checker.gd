extends Node
# Headless behavioral check for the #60 M4 locomotion race: three quadruped lanes driven by the
# 500k / 2.5M / 6M training-generation nets race in one physics space. Asserts the LEARNING ARC
# holds — the latest generation (lane 2, 6M) out-distances the earliest (lane 0, 500k) — and that
# every lane's net loaded and inferred. This is the showcase's "it actually got better" proof.

@export var leaderboard_path: NodePath
@export var agent_paths: Array[NodePath] = []
@export var frames_to_run := 4000
@export var min_arc_gap := 5.0   ## 6M lane must lead the 500k lane by at least this many metres

var _board
var _agents: Array = []
var _frames := 0

func _ready() -> void:
	_board = get_node_or_null(leaderboard_path)
	for p in agent_paths:
		_agents.append(get_node_or_null(p))
	if _board == null or _agents.size() < 3:
		_fail("missing leaderboard or < 3 lane agents")

func _physics_process(_delta: float) -> void:
	if _board == null:
		return
	for a in _agents:
		if a == null or a._ncnn_runner == null or not a._ncnn_runner.is_model_loaded():
			_fail("a lane's trained net is not loaded")
			return
	_frames += 1
	if _frames >= frames_to_run:
		var d: Array = _board.distances()
		print("RACE DIAG: gen-500k=%.2f  gen-2.5M=%.2f  gen-6M=%.2f" % [d[0], d[1], d[2]])
		if d[2] - d[0] >= min_arc_gap:
			print("RACE LEARNING-ARC PASSED (6M ahead of 500k by %.2f m)" % (d[2] - d[0]))
			get_tree().quit(0)
		else:
			_fail("6M lane led 500k by only %.2f m (need %.1f) — learning arc not shown" % [d[2] - d[0], min_arc_gap])

func _fail(reason: String) -> void:
	printerr("RACE CHECK FAILED: %s" % reason)
	get_tree().quit(1)

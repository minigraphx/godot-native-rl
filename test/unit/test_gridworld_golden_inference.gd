extends SceneTree
# Golden inference regression for the shipped GridWorld ncnn model (#48). Discrete argmax over
# fixed 52-dim observations (50 zeros for the empty sensor window + a goal vector) — argmax is
# exact cross-platform (rover pattern). If retrained, flip RECORD, rerun, paste values.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://examples/gridworld/models/gridworld.ncnn.param"
const BIN := "res://examples/gridworld/models/gridworld.ncnn.bin"

const RECORD := false

# Empty 5x5x2 sensor window + goal direction: right / left / down / up / far diagonal.
static func _obs(goal_dx: float, goal_dy: float) -> Array:
	var o: Array = []
	o.resize(50)
	o.fill(0.0)
	return o + [goal_dx, goal_dy]

var CASES: Array = [
	_obs(0.5, 0.0),    # goal right
	_obs(-0.5, 0.0),   # goal left
	_obs(0.0, 0.5),    # goal below
	_obs(0.0, -0.5),   # goal above
	_obs(0.75, 0.75),  # goal far diagonal
]

# Semantically meaningful baseline: right/left/down/up actions for the matching goal directions.
const GOLDEN: Array = [4, 3, 2, 1, 4]

func _initialize() -> void:
	var h := Harness.new()
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "gridworld model loads")
	if ok:
		for i in range(CASES.size()):
			var got := runner.run_discrete_action(PackedFloat32Array(CASES[i]))
			if RECORD:
				print("GOLDEN_%d = %d" % [i, got])
				continue
			h.assert_eq(got, int(GOLDEN[i]), "golden case %d argmax" % i)
	runner.free()
	h.finish(self)

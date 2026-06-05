extends Node
# Headless behavioral floor: runs BOTH trained policies (seeker via its model, hider via its) under
# ncnn inference and asserts the SEEKER keeps the hider in line-of-sight for at least
# `min_los_fraction` of the run -- its learned objective (it is rewarded per-frame for LOS). Against
# an actively-evading *trained* hider a random seeker scores low LOS, so clearing the floor shows the
# seeker policy learned to pursue. Catches are reported but NOT gated: in this walls-favor-the-hider
# self-play they're sparse (0-1 per run) and too flaky for a floor. The golden-inference test
# (test/unit/test_hide_seek_multipolicy_golden_inference.gd) is the precise, deterministic regression;
# this is the generous behavioral sanity floor (threshold well below observed ~20-44% LOS).

@export var game_path: NodePath
@export var seeker_path: NodePath
@export var hider_path: NodePath
@export var frames_to_run := 3000
@export var min_los_fraction := 0.08
@export var rng_seed := 1  ## seed the game's spawn RNG so the run (and LOS fraction) is deterministic

var _game
var _frames := 0
var _catches := 0
var _los_frames := 0
var _was_caught_last := false

func _ready() -> void:
	_game = get_node_or_null(game_path)
	var seeker = get_node_or_null(seeker_path)
	var hider = get_node_or_null(hider_path)
	if _game == null or seeker == null or hider == null:
		_fail("could not resolve game/seeker/hider nodes")
		return
	for a in [seeker, hider]:
		if a._ncnn_runner == null or not a._ncnn_runner.is_model_loaded():
			_fail("a trained ncnn model is not loaded")
			return
	# Deterministic run: seed the spawn RNG and re-spawn so the LOS fraction is reproducible
	# (the policies are argmax-deterministic; spawn positions were the only nondeterminism).
	if _game.has_method("seed_rng"):
		_game.seed_rng(rng_seed)
		_game.reset_positions()

func _physics_process(_delta: float) -> void:
	if _game == null:
		return
	# Rising-edge catch count (reported only); LOS fraction is the gated signal.
	var caught: bool = _game.was_caught()
	if caught and not _was_caught_last:
		_catches += 1
	_was_caught_last = caught
	if _game.has_los():
		_los_frames += 1
	_frames += 1
	if _frames >= frames_to_run:
		var los_frac := float(_los_frames) / float(_frames)
		var report := "los=%.1f%% (%d/%d), catches=%d" % [100.0 * los_frac, _los_frames, _frames, _catches]
		if los_frac >= min_los_fraction:
			print("MULTI-POLICY HIDE&SEEK PASSED (%s; floor %.0f%%)" % [report, 100.0 * min_los_fraction])
			get_tree().quit(0)
		else:
			_fail("%s below LOS floor %.0f%% — seeker did not learn to pursue" % [report, 100.0 * min_los_fraction])

func _fail(reason: String) -> void:
	printerr("MULTI-POLICY HIDE&SEEK FAILED: %s" % reason)
	get_tree().quit(1)

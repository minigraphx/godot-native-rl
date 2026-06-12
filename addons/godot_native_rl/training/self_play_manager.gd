extends Node
# League self-play coordinator (#29): owns the opponent pool + ELO ledger, swaps the ghost
# agent's frozen ncnn snapshot at episode boundaries, and records match outcomes.
# The ghost is an ordinary NCNN_INFERENCE controller — invisible to the trainer (NcnnSync's
# n_agents counts only TRAINING agents), so any stock single-policy backend trains against it.
# Spec: docs/superpowers/specs/2026-06-12-competitive-selfplay-design.md

const OpponentPool = preload("res://addons/godot_native_rl/training/opponent_pool.gd")

signal opponent_changed(name: String)
signal ratings_updated(learner_rating: float)

@export var pool_dir := "user://selfplay_pool"
@export var ghost_agent_path: NodePath
@export var pick_mode := "uniform"  ## "uniform" | "latest"
@export var elo_k := 32.0
@export var rng_seed := 0

const BASELINE := "__baseline__"

var _pool = OpponentPool.new()
var _ghost: Node
var _rng := RandomNumberGenerator.new()
var _current := BASELINE

func _ready() -> void:
	if not is_in_group("SELF_PLAY"):
		add_to_group("SELF_PLAY")
	_rng.seed = rng_seed
	if _ghost == null:
		_ghost = get_node_or_null(ghost_agent_path)
	if _ghost == null:
		push_error("SelfPlayManager: ghost_agent_path not set/invalid.")
		return
	_load_ledger()
	if _pool.is_empty():
		# Rated placeholder for the ghost's preconfigured (scene-set) model.
		_pool.add_member(BASELINE)
		push_warning("SelfPlayManager: pool '%s' is empty — playing the ghost's preconfigured model as %s." % [pool_dir, BASELINE])
	else:
		_assign_next_opponent()

func set_ghost_for_test(g: Node) -> void:
	_ghost = g

# Re-read the pool dir/ledger and (re)assign an opponent — for pools that grow after _ready
# (e.g. the alternating-phase orchestrator registering fresh snapshots, or tests priming a
# temp pool).
func rescan_pool() -> void:
	_load_ledger()
	if not _pool.is_empty():
		_assign_next_opponent()

# Called by the TRAINING-side agent at each episode end (one line, null-guarded — same
# integration pattern as the curriculum's record_episode).
func report_match(learner_won: bool, draw := false) -> void:
	if _pool.record_match(_current, learner_won, draw, elo_k):
		print("SelfPlay: match vs %s -> learner %s | learner_rating=%.1f opp_rating=%.1f" % [
			_current, ("draw" if draw else ("won" if learner_won else "lost")),
			_pool.learner_rating(), _pool.member_rating(_current)])
		_save_ledger()
		ratings_updated.emit(_pool.learner_rating())
	_assign_next_opponent()

func learner_rating() -> float:
	return _pool.learner_rating()

func current_opponent() -> String:
	return _current

func _assign_next_opponent() -> void:
	var pick: String = _pool.pick_opponent(_rng, pick_mode)
	if pick == "" or pick == BASELINE:
		_current = BASELINE
		return
	var param := pool_dir.path_join(pick + ".ncnn.param")
	var bin := pool_dir.path_join(pick + ".ncnn.bin")
	if _ghost.has_method("reload_model") and _ghost.reload_model(param, bin):
		if pick != _current:
			print("SelfPlay: ghost now plays snapshot '%s' (rating %.1f)" % [pick, _pool.member_rating(pick)])
			opponent_changed.emit(pick)
		_current = pick
	else:
		push_error("SelfPlayManager: could not load snapshot '%s'; keeping '%s'." % [pick, _current])

func _ledger_path() -> String:
	return pool_dir.path_join("pool.json")

func _load_ledger() -> void:
	var f := FileAccess.open(_ledger_path(), FileAccess.READ)
	if f != null:
		_pool.load_ledger(f.get_as_text())

func _save_ledger() -> void:
	DirAccess.make_dir_recursive_absolute(pool_dir)
	var f := FileAccess.open(_ledger_path(), FileAccess.WRITE)
	if f == null:
		push_error("SelfPlayManager: cannot write ledger '%s'." % _ledger_path())
		return
	f.store_string(_pool.ledger_to_json())

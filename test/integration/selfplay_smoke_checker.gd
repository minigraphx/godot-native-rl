extends Node
# Self-play integration smoke (#29): primes a real 2-member opponent pool (copies of the
# committed hider fixture), then drives match outcomes through the REAL selfplay scene —
# asserting ghost snapshot swaps (real ncnn reloads), ledger persistence + rating motion,
# and the n_agents invariant (the ghost is invisible to the trainer: 1 training + 1 inference).

@export var manager_path: NodePath
@export var ghost_path: NodePath
@export var sync_path: NodePath

const SRC_PARAM := "res://examples/hide_and_seek/models/hide_seek_hider.ncnn.param"
const SRC_BIN := "res://examples/hide_and_seek/models/hide_seek_hider.ncnn.bin"

var _manager
var _ghost
var _sync
var _changes := 0

func _ready() -> void:
	_manager = get_node_or_null(manager_path)
	_ghost = get_node_or_null(ghost_path)
	_sync = get_node_or_null(sync_path)
	if _manager == null or _ghost == null or _sync == null:
		_fail("could not resolve manager/ghost/sync")
		return
	_manager.opponent_changed.connect(func(_n): _changes += 1)
	# Defer past the sync's deferred init (it awaits root.ready + a 1s timer).
	var t := get_tree().create_timer(2.0)
	t.timeout.connect(_run)

func _prime_pool(dir: String) -> bool:
	DirAccess.make_dir_recursive_absolute(dir)
	for gen in ["gen1", "gen2"]:
		var p := FileAccess.get_file_as_bytes(SRC_PARAM)
		var b := FileAccess.get_file_as_bytes(SRC_BIN)
		if p.is_empty() or b.is_empty():
			return false
		var fp := FileAccess.open(dir.path_join(gen + ".ncnn.param"), FileAccess.WRITE)
		fp.store_buffer(p)
		fp.close()
		var fb := FileAccess.open(dir.path_join(gen + ".ncnn.bin"), FileAccess.WRITE)
		fb.store_buffer(b)
		fb.close()
	var ledger := {"members": {"gen1": {"rating": 1200.0, "games": 0}, "gen2": {"rating": 1200.0, "games": 0}}, "learner_rating": 1200.0}
	var fl := FileAccess.open(dir.path_join("pool.json"), FileAccess.WRITE)
	fl.store_string(JSON.stringify(ledger))
	fl.close()
	return true

func _run() -> void:
	# n_agents invariant: ghost in inference list, learner in training list.
	if _sync.agents_training.size() != 1:
		_fail("agents_training != 1 (got %d)" % _sync.agents_training.size())
		return
	if _sync.agents_inference.size() != 1:
		_fail("agents_inference != 1 (got %d)" % _sync.agents_inference.size())
		return

	# Redirect the manager to a test-local pool so the smoke never pollutes the real user:// pool.
	_manager.pool_dir = "user://selfplay_smoke_pool"
	if not _prime_pool(_manager.pool_dir):
		_fail("could not prime pool (committed hider fixture missing?)")
		return
	_manager.rescan_pool()
	if not (_manager.current_opponent() in ["gen1", "gen2"]):
		_fail("no pool opponent assigned after rescan (got %s)" % _manager.current_opponent())
		return
	if _ghost._ncnn_runner == null or not _ghost._ncnn_runner.is_model_loaded():
		_fail("ghost has no loaded net after snapshot assignment")
		return

	var lr0: float = _manager.learner_rating()
	for i in range(20):
		_manager.report_match(i % 3 != 0)  # mostly learner wins
	if absf(_manager.learner_rating() - lr0) < 1e-6:
		_fail("learner rating did not move over 20 matches")
		return
	if _changes == 0:
		_fail("opponent never changed over 20 uniform picks of 2 members")
		return
	if not FileAccess.file_exists(_manager.pool_dir.path_join("pool.json")):
		_fail("ledger not persisted")
		return
	print("SELFPLAY SMOKE PASSED (%d opponent swaps, learner_rating %.1f -> %.1f)" % [_changes, lr0, _manager.learner_rating()])
	get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("SELFPLAY SMOKE FAILED: %s" % reason)
	get_tree().quit(1)

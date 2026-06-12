# Competitive Self-Play Implementation Plan (#29)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** League self-play for Hide & Seek: a learner trains against frozen ncnn "ghost" snapshots picked from an opponent pool with ELO tracking, via alternating-role phases driven by stock single-policy training.

**Architecture:** Pure `elo.gd` + `opponent_pool.gd` (ledger/selection) under a thin `SelfPlayManager` node that swaps the ghost's model at episode boundaries via a new `reload_model()` on the controllers. The ghost is an `NCNN_INFERENCE` agent — invisible to the trainer (`n_agents` counts only training agents), so any backend works. `train_selfplay.sh` alternates roles, exporting each phase's learner into the pool.

**Tech Stack:** GDScript (TAB, path-based extends), `test/harness.gd`, stdlib Python, existing TorchScript→ncnn export chain.

**Spec:** `docs/superpowers/specs/2026-06-12-competitive-selfplay-design.md`
**Run tests with:** `GODOT=/opt/homebrew/bin/godot-mono`; under classifier outage use the allowlisted `GODOT="/Applications/Godot_mono.app/Contents/MacOS/Godot" ./test/run_tests.sh`.

---

## File structure

- Create `addons/godot_native_rl/training/elo.gd` (+ `test/unit/test_elo.gd`)
- Create `addons/godot_native_rl/training/opponent_pool.gd` (+ `test/unit/test_opponent_pool.gd`)
- Create `addons/godot_native_rl/training/self_play_manager.gd` (+ `test/unit/test_self_play_manager.gd`)
- Modify `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` + `_3d.gd` — extract `_setup_ncnn_runner` body into public `reload_model(param_path, bin_path) -> bool` (+ `test/unit/test_controller_reload_model.gd`)
- Modify `examples/hide_and_seek/hide_seek_agent.gd` — null-guarded `report_match` hook in the terminal branch
- Create `examples/hide_and_seek/hide_and_seek_selfplay_seeker.tscn` / `_hider.tscn`
- Create `scripts/selfplay_phase.py` (+ `test/python/test_selfplay_phase.py`), `scripts/train_selfplay.sh`
- Create `test/integration/selfplay_smoke_checker.gd` + `selfplay_smoke_scene.tscn`; register in `test/run_tests.sh`
- Docs: README, CLAUDE.md, gap-analysis, BACKLOG item 53; `Closes #29`; file follow-ups (two-sided simultaneous, ELO matchmaking, inference-time recording — last one belongs to #39 but file together).

---

### Task 1: Pure ELO math

Test `test/unit/test_elo.gd`:

```gdscript
extends SceneTree
const Harness = preload("res://test/harness.gd")
const Elo = preload("res://addons/godot_native_rl/training/elo.gd")

func _initialize() -> void:
	var h = Harness.new()
	h.assert_eq(Elo.expected_score(1200.0, 1200.0), 0.5, "equal ratings -> 0.5")
	h.assert_true(Elo.expected_score(1400.0, 1200.0) > 0.75, "200pt favorite > 0.75")
	h.assert_true(absf(Elo.expected_score(1400.0, 1200.0) + Elo.expected_score(1200.0, 1400.0) - 1.0) < 1e-9, "expectations sum to 1")
	# win moves rating up by k*(1-e); loss down by k*e
	var r := Elo.update(1200.0, 0.5, 1.0, 32.0)
	h.assert_eq(r, 1216.0, "k=32 win from even -> +16")
	h.assert_eq(Elo.update(1200.0, 0.5, 0.0, 32.0), 1184.0, "k=32 loss from even -> -16")
	h.assert_eq(Elo.update(1200.0, 0.5, 0.5, 32.0), 1200.0, "draw from even -> unchanged")
	# zero-sum pair update
	var pair := Elo.update_pair(1300.0, 1100.0, 0.0, 32.0)  # upset: low-rated wins
	h.assert_true(pair[0] < 1300.0 and pair[1] > 1100.0, "upset moves both")
	h.assert_true(absf((pair[0] - 1300.0) + (pair[1] - 1100.0)) < 1e-9, "zero-sum")
	h.finish(self)
```

Implementation `addons/godot_native_rl/training/elo.gd`:

```gdscript
extends RefCounted
# Pure ELO rating math (static; no state). Used by the self-play opponent pool (#29).

static func expected_score(rating_a: float, rating_b: float) -> float:
	return 1.0 / (1.0 + pow(10.0, (rating_b - rating_a) / 400.0))

static func update(rating: float, expected: float, actual: float, k := 32.0) -> float:
	return rating + k * (actual - expected)

# score_a: 1.0 a wins, 0.5 draw, 0.0 b wins. Returns [new_a, new_b] (zero-sum).
static func update_pair(rating_a: float, rating_b: float, score_a: float, k := 32.0) -> Array:
	var ea := expected_score(rating_a, rating_b)
	return [update(rating_a, ea, score_a, k), update(rating_b, 1.0 - ea, 1.0 - score_a, k)]
```

Run, commit: `feat: pure ELO rating math (#29)`.

### Task 2: Opponent pool + ledger (pure)

Test `test/unit/test_opponent_pool.gd` — covers: empty pool behavior, `add_member` initial rating
= learner rating, `pick_opponent` uniform (seeded RNG reproducibility) + `latest`, `record_match`
rating evolution (learner beats member → learner up/member down, zero-sum), ledger JSON round-trip,
unknown member fail-loud (returns false), `members()` listing.

Implementation `addons/godot_native_rl/training/opponent_pool.gd`:

```gdscript
extends RefCounted
# Opponent pool + ELO ledger for league self-play (#29). Pure logic: file I/O stays in the
# SelfPlayManager node. Ledger shape:
# {"members": {"<name>": {"rating": float, "games": int}}, "learner_rating": float}

const Elo = preload("res://addons/godot_native_rl/training/elo.gd")
const DEFAULT_RATING := 1200.0

var _members: Dictionary = {}      # name -> {"rating": float, "games": int}
var _learner_rating := DEFAULT_RATING

func learner_rating() -> float: return _learner_rating
func members() -> Array: return _members.keys()
func member_rating(name: String) -> float: return float(_members.get(name, {}).get("rating", -1.0))
func is_empty() -> bool: return _members.is_empty()

func add_member(name: String) -> void:
	_members[name] = {"rating": _learner_rating, "games": 0}

func pick_opponent(rng: RandomNumberGenerator, mode := "uniform") -> String:
	if _members.is_empty(): return ""
	var names: Array = _members.keys()
	match mode:
		"latest": return names.back()
		_: return names[rng.randi_range(0, names.size() - 1)]

func record_match(member_name: String, learner_won: bool, draw := false, k := 32.0) -> bool:
	if not _members.has(member_name):
		push_error("OpponentPool: unknown member '%s'." % member_name)
		return false
	var score := 0.5 if draw else (1.0 if learner_won else 0.0)
	var pair := Elo.update_pair(_learner_rating, member_rating(member_name), score, k)
	_learner_rating = pair[0]
	_members[member_name]["rating"] = pair[1]
	_members[member_name]["games"] = int(_members[member_name]["games"]) + 1
	return true

func ledger_to_json() -> String:
	return JSON.stringify({"members": _members, "learner_rating": _learner_rating}, "\t")

func load_ledger(json_text: String) -> bool:
	var parsed = JSON.parse_string(json_text)
	if not (parsed is Dictionary) or not (parsed.get("members") is Dictionary):
		push_error("OpponentPool: malformed ledger JSON.")
		return false
	_members = parsed["members"]
	_learner_rating = float(parsed.get("learner_rating", DEFAULT_RATING))
	return true
```

Commit: `feat: opponent pool + ELO ledger (pure) (#29)`.

### Task 3: Controller `reload_model`

Refactor `_setup_ncnn_runner` in BOTH controllers so the byte-load block becomes:

```gdscript
func reload_model(param_path: String, bin_path: String) -> bool:
	model_param_path = param_path
	model_bin_path = bin_path
	if _ncnn_runner == null:
		_ncnn_runner = NcnnRunner.new()
		_ncnn_runner.input_blob_name = input_blob_name
		_ncnn_runner.output_blob_name = output_blob_name
		add_child(_ncnn_runner)
	var param_bytes := FileAccess.get_file_as_bytes(param_path)
	var bin_bytes := FileAccess.get_file_as_bytes(bin_path)
	if param_bytes.is_empty() or bin_bytes.is_empty():
		push_error("%s: cannot read model files '%s' / '%s'." % [name, param_path, bin_path])
		return false
	if not _ncnn_runner.load_model_from_buffers(param_bytes, bin_bytes):
		push_error("%s: failed to load ncnn model '%s'." % [name, param_path])
		return false
	_core.init_recurrent_state()  # fresh memory for a fresh policy
	return true
```

with `_setup_ncnn_runner` reduced to the empty-path guard + `reload_model(model_param_path, model_bin_path)` (cleanup of a failed runner stays as-is — on a false return with a fresh runner, free it like today). Test `test/unit/test_controller_reload_model.gd`: load the committed chase fixture, `reload_model` to the dummy fixture (`chase_dummy.ncnn.*`) → true + inference still runs; `reload_model` to a bad path → false, loud, runner survives with the old net.
Commit: `feat: runtime reload_model on NcnnAIController2D/3D (#29)`.

### Task 4: SelfPlayManager node

Test `test/unit/test_self_play_manager.gd` — stub ghost Node with `reload_model(p, b)` recorder
returning true; temp pool dir under `user://selfplay_test_pool/` with two fake member files +
ledger; assertions: `_ready` loads ledger; `report_match(true)` updates ratings + persists JSON +
assigns a new opponent (ghost recorder called with a pool member's paths); empty-pool path records
vs `__baseline__` without reload; `opponent_changed`/`ratings_updated` signals fire.

Implementation `addons/godot_native_rl/training/self_play_manager.gd`:

```gdscript
extends Node
# League self-play coordinator (#29): owns the opponent pool + ELO ledger, swaps the ghost
# agent's frozen ncnn snapshot at episode boundaries, and records match outcomes.
# The ghost is an ordinary NCNN_INFERENCE controller — invisible to the trainer.

const OpponentPool = preload("res://addons/godot_native_rl/training/opponent_pool.gd")

signal opponent_changed(name: String)
signal ratings_updated(ledger_learner_rating: float)

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
	_ghost = get_node_or_null(ghost_agent_path)
	if _ghost == null:
		push_error("SelfPlayManager: ghost_agent_path not set/invalid.")
		return
	_load_ledger()
	if _pool.is_empty():
		_pool.add_member(BASELINE)  # rated placeholder for the ghost's preconfigured model
		push_warning("SelfPlayManager: pool '%s' empty — playing the ghost's preconfigured model as %s." % [pool_dir, BASELINE])
	else:
		_assign_next_opponent()

func report_match(learner_won: bool, draw := false) -> void:
	if _pool.record_match(_current, learner_won, draw, elo_k):
		print("SelfPlay: match vs %s -> learner %s | learner_rating=%.1f opp_rating=%.1f" % [
			_current, ("draw" if draw else ("won" if learner_won else "lost")),
			_pool.learner_rating(), _pool.member_rating(_current)])
		_save_ledger()
		ratings_updated.emit(_pool.learner_rating())
	_assign_next_opponent()

func learner_rating() -> float: return _pool.learner_rating()
func current_opponent() -> String: return _current

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

func _ledger_path() -> String: return pool_dir.path_join("pool.json")

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
```

Commit: `feat: SelfPlayManager — ghost snapshot swapping + ELO ledger (#29)`.

### Task 5: Hide & Seek hook + selfplay scenes

`hide_seek_agent.gd`: resolve `_selfplay = get_tree().get_first_node_in_group("SELF_PLAY")` in
`_ready` (null-guarded). In the terminal branch (find the agent's `needs_reset`/terminal handling —
read the current code first), the **training-side** agent reports:
`_selfplay.report_match(<learner_won>)` where learner_won = `_game.was_caught()` if the learner is
the seeker else `not _game.was_caught()` (timeout = hider win; no draws). Only the TRAINING-mode
agent reports (guard: `control_mode != ControlModes.NCNN_INFERENCE`), so the ghost never
double-reports.

Scenes: `hide_and_seek_selfplay_seeker.tscn` = the hide_and_seek_train scene with: seeker agent
`control_mode = 2` (TRAINING), hider agent `control_mode = 3` + the committed
`hide_seek_hider.ncnn.*` as its preconfigured model, plus a `SelfPlayManager`
(`ghost_agent_path` → hider, `pool_dir = "user://selfplay_pool/hider"`). `_hider.tscn` mirrored.
Verify both load headless.
Commit: `feat: hide&seek self-play scenes + match reporting (#29)`.

### Task 6: Integration smoke

`test/integration/selfplay_smoke_checker.gd` + scene: real hide&seek game + a TRAINING seeker
stub-driven + a real ghost hider with the committed fixture; checker primes a 2-member pool dir
under `user://` (copies the committed hider fixture twice via `DirAccess`), instantiates ledger,
then calls `manager.report_match(...)` with a scripted win/loss stream; asserts: opponent swaps
happened (`opponent_changed` count > 0 across 20 matches with uniform pick over 2 members),
ledger file exists + learner rating moved, ghost runner still loaded, and the scene's
`agents_training.size()`-equivalent invariant: the ghost is NOT in the sync's training list
(instantiate a bare NcnnSync, call its `_get_agents()`, assert counts 1/1).
Register in `run_tests.sh` after the curriculum smoke (note: that line exists only on the #28
branch — on this branch insert after the quadruped smoke; merge order resolves the neighbors).
Commit: `test: self-play integration smoke (#29)`.

### Task 7: Phase orchestration

`scripts/selfplay_phase.py` — stdlib CLI: `register-snapshot --pool-dir D --name N [--rating R]`
appends member N to D/pool.json at the given rating (default: current learner_rating in the
ledger, else 1200) — pure function `register_snapshot(ledger: dict, name: str, rating=None) -> dict`
unit-tested in `test/python/test_selfplay_phase.py` (new member at learner rating; explicit rating
honored; existing member errors).
`scripts/train_selfplay.sh` — loop `PHASES` (default 4): role = seeker on odd, hider on even;
runs `train_hide_seek.py` with `SCENE=` the role's selfplay scene and `--policy-filter <role>`…
**verify** how train_hide_seek.py names/filters policies first — if it trains the single shared
policy of whatever agents are in TRAINING mode (it does: stock single-policy over the wire),
no filter is needed. After each phase: export learner → TorchScript → ncnn into the OPPOSITE
role's pool dir as `<role>_gen<K>` + `selfplay_phase.py register-snapshot`. `TIMESTEPS_PER_PHASE`,
`PHASES`, `OUTDIR` overrides; `bash -n` + a tiny `--dry-run` echo mode for the unit test.
Commit: `feat: alternating-phase self-play orchestration (#29)`.

### Task 8: Suite green + trained run + docs + PR

- Full suite green (allowlisted runner under outage).
- Real `train_selfplay.sh` run (4 short phases) → pool of ≥4 snapshots + evolving `pool.json` +
  assignment/rating log lines; capture for the PR (post as comment if tooling is gated at the time).
- Docs: README bullet (lead with native ghosts), CLAUDE.md key command, gap-analysis row
  (Unity ML-Agents self-play parity), BACKLOG item 53 ✅.
- File follow-up issues: simultaneous two-sided self-play; ELO-proximity matchmaking.
- Push, `gh pr create` (`Closes #29`).

---

## Self-review

- Spec coverage: elo (T1), pool/ledger (T2), reload_model (T3), manager (T4), game hook + scenes
  (T5), smoke incl. n_agents invariant (T6), orchestration + register-snapshot (T7), trained run +
  docs + follow-ups (T8). ✔
- Placeholders: T5/T7 carry explicit "read the current code first" verification steps where
  insertion points depend on live code; all new-file code is complete. ✔
- Naming consistency: `reload_model`, `report_match`, `pick_opponent`, `record_match`,
  `learner_rating`, `member_rating`, `add_member`, `load_ledger`/`ledger_to_json`, groups
  `SELF_PLAY`/`AGENT`, `__baseline__`. ✔

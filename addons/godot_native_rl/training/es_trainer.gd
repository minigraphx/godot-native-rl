extends Node
# ESTrainer (#131): a native, in-engine training loop — no Python, no socket, no backprop.
#
# A drop-in replacement for the NcnnSync node that trains instead of bridging: it drives the same
# agent contract NcnnSync does (group "AGENT"; get_obs()["obs"], set_action(dict),
# get_reward()/zero_reward(), get_done()/set_done_false(), needs_reset, get_action_space()) at the
# same action_repeat cadence — honouring the same `speedup=`/`action_repeat=` cmdline overrides —
# but the actions come from CANDIDATE nets the trainer itself evolves: an OpenAI-ES optimizer
# (es_math.gd; or sep-CMA-ES via `optimizer = "cma_es"`, cma_math.gd — self-adapting step size +
# per-coordinate variances) perturbs a flat weight vector θ, each candidate becomes a live ncnn net via the
# θ ⇄ buffers codec (ncnn_weights.gd) + NcnnRunner.load_model_from_buffers, and episodic return
# (the existing reward system) is its fitness. Multi-agent scenes (e.g. ParallelArena tiles)
# evaluate one candidate per agent slot in parallel waves.
#
# The trainer OWNS the episode horizon (episode_decisions, counted in decision steps): each
# agent's reset_after is overridden to effectively-infinite (and restored when training ends) so
# the controller never HORIZON-self-resets — a self-reset zeroes the agent's reward accumulator
# before the trainer can read it, silently deleting the final window's reward (catches!) from the
# fitness. Rewards are drained EVERY physics tick (not just on decision ticks) and `done` is acted
# on the tick it appears, so agents that terminate-and-self-reset game-side (3dball-style falls)
# can't leak their next episode's reward into the finished candidate either.
#
# Fitness comparability safeguards (ES lives or dies on the between-candidate ranking):
# - Common random numbers: with seed_games on, every candidate's episode k in generation g is
#   seeded identically (EsMath.episode_seed) via the agent's `game_path` game's `seed_rng()` —
#   spawn luck cancels out of the ranking. episode_starting is emitted for custom seeding.
# - Episode starts are TWO-PHASE: on episode end the slot first freezes for one decision window
#   under a neutral action (decoded all-zero logits), letting any old-stream game event (e.g. a
#   chase catch resolving on the pre-reset tick) consume the OLD RNG stream; only then is the
#   game seeded and the reset requested — so the fresh stream's first draws always belong to the
#   reset, identically for every candidate.
# - k-episode averaging (episodes_per_candidate) cuts residual per-episode noise.
# - A terminal `done` before the candidate's first decision (hostile spawn) restarts the episode
#   instead of recording a phantom 0 fitness.
#
# The training artifact IS the deploy artifact: checkpoints are ncnn .param/.bin pairs, loadable
# by every controller unchanged — there is no export step. Warm-starting from a shipped net is the
# same codec in reverse (warm_start_*_path; the .param file is validated against this scene's
# architecture, fail loud).
#
# Scope/trade-offs: flat float obs only (fail loud otherwise); ES is sample-inefficient — this is
# for small nets, dense rewards, and self-contained/on-device learning, not a PPO/SAC replacement.
# Place the trainer AFTER the agents in the scene tree (like Sync) so agents step first each tick.

const EsMath = preload("res://addons/godot_native_rl/training/es_math.gd")
const CmaMath = preload("res://addons/godot_native_rl/training/cma_math.gd")
const NcnnWeights = preload("res://addons/godot_native_rl/training/ncnn_weights.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")
const RunSpeed = preload("res://addons/godot_native_rl/training/run_speed.gd")

signal generation_finished(generation: int, mean_fitness: float, best_fitness: float)
signal training_finished(best_mean_fitness: float)
## Emitted after a checkpoint pair is written (stem is e.g. "<checkpoint_stem>_best"). Lets
## consumers react to a blessed net without re-deriving the blessing rule — e.g. the Evolution
## Lab demo hot-swaps its champion agent onto every new best.
signal checkpoint_saved(stem: String, param_path: String, bin_path: String)
## Emitted right before an episode reset is requested — hook for CUSTOM per-episode game setup
## beyond the built-in seed_games seeding (curriculum stages, layout variants, ...).
signal episode_starting(slot: int, candidate_index: int, generation: int, episode: int)

@export var agent_group := "AGENT"
@export var hidden_dims: Array[int] = [16]
@export var hidden_activation := "relu"  # relu | tanh | sigmoid | "" (see ncnn_weights.gd)
## "openai_es" (antithetic gradient estimate, es_math.gd) or "cma_es" (sep-CMA-ES, cma_math.gd:
## top-μ recombination + step-size and per-coordinate variance adaptation — sigma becomes the
## INITIAL step size and self-adapts; alpha is unused). Cmdline `optimizer=` overrides.
@export var optimizer := "openai_es"
@export var half_population := 8  ## candidates per generation = 2 * half_population (antithetic)
@export var sigma := 0.1  ## perturbation stddev (openai_es: fixed; cma_es: initial, self-adapts)
@export var alpha := 0.05  ## learning rate (openai_es only)
@export var generations := 100
@export var episode_decisions := 100  ## episode horizon in DECISION steps (× action_repeat ticks)
@export var episodes_per_candidate := 1  ## fitness = mean return over k seeded episodes
@export var action_repeat := 8  ## physics ticks between decisions (cmdline `action_repeat=` overrides)
@export var rng_seed := 0
@export var speed_up := 1.0  ## same mechanism as NcnnSync (cmdline `speedup=` overrides)
## Seed each agent's game (duck-typed: the node at the agent's `game_path` with a `seed_rng()`
## method) per (generation, episode) so all candidates face identical draws — common random
## numbers. Warns once when nothing seedable is found; connect episode_starting to seed manually.
@export var seed_games := true
@export var out_dir := "user://es_checkpoints"
@export var checkpoint_stem := "es_policy"
@export var checkpoint_every := 0  ## write <stem>_gen<N> every N generations (0 = final/best only)
## Warm-start θ from a shipped net (both paths or neither). The .param file must match the
## architecture this scene builds ([obs_dim] + hidden_dims + [action_dim], activations) — the
## trainer compares it against its own generated param text and aborts on mismatch.
@export var warm_start_param_path := ""
@export var warm_start_bin_path := ""
@export var input_blob_name := "in0"
@export var output_blob_name := "out0"
## Quit the SceneTree when training finishes (exit 0) or aborts on an error (exit 1) — for
## headless CLI runs. Leave off when the trainer is embedded in a larger scene (demos, tests).
@export var exit_on_finish := false

## A terminal done before the candidate's first decision restarts the episode this many times
## before scoring what accrued (a broken always-terminal env must not loop forever).
const MAX_PHANTOM_RESTARTS := 3

# Per-slot episode phases: FREEZE holds the neutral action for one decision window (old-stream
# game events resolve here), SEEDED has requested the seeded reset and awaits the first decision,
# RUNNING is the scored episode. IDLE = no candidate assigned (population exhausted this wave).
enum Phase { IDLE, FREEZE, SEEDED, RUNNING }

var _agents: Array = []
var _runners: Array = []
var _slot_game: Array = []  # per slot: the seedable game node (null when none found)
var _saved_reset_after: Array = []  # restored when training ends (embedded scenes keep running)
var _spec: Dictionary = {}
var _param_bytes := PackedByteArray()
var _theta := PackedFloat32Array()
var _action_space: Dictionary = {}
var _neutral_action: Dictionary = {}  # decoded all-zero logits: the candidate-independent reset action
var _rng := RandomNumberGenerator.new()
var _cma = null  # CmaMath instance when optimizer == "cma_es"
var _epsilons: Array = []
var _fitness: Array = []
var _next_candidate := 0
var _n_candidates := 0
var _slot_phase: Array = []
var _slot_candidate: Array = []  # candidate index under evaluation per agent slot
var _slot_episode: Array = []  # 0-based episode index within the candidate's k episodes
var _slot_decisions: Array = []  # decision steps taken in the current episode
var _slot_accum: Array = []  # summed return across the candidate's completed episodes
var _slot_restarts: Array = []  # phantom-done restarts of the current episode
var _generation := 0
var _tick := 0
var _best_mean := -INF


func _ready() -> void:
	var args := RunSpeed.parse_cmdline_args()
	speed_up = float(args.get("speedup", str(speed_up)))
	action_repeat = int(args.get("action_repeat", str(action_repeat)))
	optimizer = str(args.get("optimizer", optimizer))
	generations = int(args.get("generations", str(generations)))
	_rng.seed = rng_seed
	set_physics_process(false)
	# Agents register their group in their own _ready; initialize (and validate the possibly
	# cmdline-overridden config) after the whole scene is in.
	_late_init.call_deferred()


func _late_init() -> void:
	if episode_decisions < 1 or episodes_per_candidate < 1 or half_population < 1:
		_abort("episode_decisions, episodes_per_candidate and half_population must all be >= 1.")
		return
	if optimizer != "openai_es" and optimizer != "cma_es":
		_abort("unknown optimizer '%s' (openai_es | cma_es — a bad cmdline override?)." % optimizer)
		return
	if optimizer == "cma_es" and half_population < 2:
		_abort("cma_es needs a population of >= 4, so half_population >= 2.")
		return
	if generations < 1:
		_abort("generations must be >= 1 (got %d — a bad cmdline override?)." % generations)
		return
	# int("garbage") is 0, so an action_repeat=typo cmdline override would otherwise reach the
	# `_tick % action_repeat` hot path — a hang in debug builds and a SIGFPE (% 0) in release.
	if action_repeat < 1 or speed_up <= 0.0:
		_abort("action_repeat must be >= 1 and speed_up > 0 (got %d / %s — a bad cmdline override?)"
			% [action_repeat, str(speed_up)])
		return
	RunSpeed.apply(speed_up)
	if not _discover_agents():
		return
	var first_obs: Dictionary = _agents[0].get_obs()
	if not first_obs.has("obs"):
		_abort("agent get_obs() has no flat 'obs' key; image obs are unsupported.")
		return
	var obs_dim: int = (first_obs["obs"] as Array).size()
	_action_space = _agents[0].get_action_space()
	var action_dim := 0
	for key in _action_space:
		action_dim += int(_action_space[key]["size"])
	var dims: Array = [obs_dim]
	dims.append_array(hidden_dims)
	dims.append(action_dim)
	_spec = NcnnWeights.mlp_spec(dims, hidden_activation)
	if _spec.is_empty():
		_abort("invalid MLP spec (see the error above).")
		return
	_param_bytes = NcnnWeights.param_text(_spec).to_utf8_buffer()
	_theta = _initial_theta()
	if _theta.is_empty():
		return  # _initial_theta aborted with the specific reason
	var zero_out := PackedFloat32Array()
	zero_out.resize(action_dim)
	_neutral_action = ActionDecode.decode_actions(zero_out, _action_space)
	_n_candidates = 2 * half_population
	if optimizer == "cma_es":
		_cma = CmaMath.new()
		if not _cma.setup(_theta, sigma, _n_candidates):
			_abort("CMA-ES setup rejected the configuration (see the error above).")
			return
	if _n_candidates % _agents.size() != 0:
		push_warning("ESTrainer: %d candidates across %d agent slot(s) leaves %d idle slot-episodes per generation — pick half_population so 2*half_population is a multiple of the slot count."
			% [_n_candidates, _agents.size(), _agents.size() - (_n_candidates % _agents.size())])
	for agent in _agents:
		# The trainer owns the episode horizon (see the header comment): the controller must
		# never HORIZON-self-reset mid-run, or the tail window's reward is zeroed before we read
		# it. Saved and restored when training ends. Optional in the duck contract — an agent
		# without reset_after just keeps its own horizon (warned: its self-reset wipes reward).
		if "reset_after" in agent:
			_saved_reset_after.append(agent.reset_after)
			agent.reset_after = 1 << 30
		else:
			_saved_reset_after.append(null)
			push_warning("ESTrainer: agent '%s' has no reset_after — its own horizon self-reset will zero reward the trainer can't read; fitness may undercount." % agent.name)
		var runner := NcnnRunner.new()
		runner.input_blob_name = input_blob_name
		runner.output_blob_name = output_blob_name
		add_child(runner)
		_runners.append(runner)
		_slot_game.append(_seedable_game_of(agent))
	if seed_games and _slot_game.count(null) == _slot_game.size():
		push_warning("ESTrainer: seed_games is on but no agent exposes a seedable game (game_path -> node with seed_rng()); candidates will face DIFFERENT random draws, which can drown the fitness ranking. Connect episode_starting to seed manually, or set seed_games = false to silence this.")
	var n_slots := _agents.size()
	for arr in [_slot_phase, _slot_candidate, _slot_episode, _slot_decisions, _slot_accum, _slot_restarts]:
		arr.resize(n_slots)
		arr.fill(0)  # Array.resize pads with null, which would poison int/float arithmetic
	print("[ESTrainer] net %s (θ=%d floats) · %s · population %d × %d episode(s) · %d agent slot(s) · %d generations"
		% [str(dims), _theta.size(), optimizer, _n_candidates, episodes_per_candidate, n_slots, generations])
	_start_generation()
	set_physics_process(true)


## Mirror NcnnSync's partition: drive TRAINING agents (adopting INHERIT_FROM_SYNC ones), leave
## HUMAN and NCNN_INFERENCE agents alone — a frozen self-play ghost or a player avatar in the
## scene must not be hijacked as a candidate slot.
func _discover_agents() -> bool:
	_agents = []
	for agent in get_tree().get_nodes_in_group(agent_group):
		var mode: int = agent.get("control_mode")
		if mode == agent.ControlModes.INHERIT_FROM_SYNC:
			agent.control_mode = agent.ControlModes.TRAINING
			mode = agent.ControlModes.TRAINING
		if mode == agent.ControlModes.TRAINING:
			_agents.append(agent)
	if _agents.is_empty():
		_abort("no TRAINING-mode agents found in group '%s'." % agent_group)
		return false
	return true


func _seedable_game_of(agent: Node) -> Node:
	if not ("game_path" in agent):
		return null
	var game: Node = agent.get_node_or_null(agent.game_path)
	if game != null and game.has_method("seed_rng"):
		return game
	return null


func _initial_theta() -> PackedFloat32Array:
	if warm_start_param_path.is_empty() != warm_start_bin_path.is_empty():
		_abort("set both warm_start paths or neither.")
		return PackedFloat32Array()
	if warm_start_bin_path.is_empty():
		return NcnnWeights.init_theta(_spec, _rng)
	# The .bin alone can't reveal an architecture mismatch (same float count, different
	# activations/layout decodes to garbage θ) — the .param can. Fail loud on any difference.
	var want_param := NcnnWeights.param_text(_spec).strip_edges()
	var have_param := FileAccess.get_file_as_string(warm_start_param_path).strip_edges()
	if have_param != want_param:
		_abort("warm-start net '%s' does not match this scene's architecture (param text differs)." % warm_start_param_path)
		return PackedFloat32Array()
	var bin := FileAccess.get_file_as_bytes(warm_start_bin_path)
	var theta: PackedFloat32Array = NcnnWeights.theta_from_bin(_spec, bin)
	if theta.is_empty():
		_abort("warm-start bin '%s' does not decode against this scene's spec." % warm_start_bin_path)
	return theta


func _start_generation() -> void:
	if _cma != null:
		_cma.sample(_rng)
	else:
		_epsilons = EsMath.sample_epsilons(_rng, half_population, _theta.size())
	_fitness = []
	_fitness.resize(_n_candidates)
	_fitness.fill(0.0)
	_next_candidate = 0
	for slot in range(_agents.size()):
		_assign_next(slot)


func _assign_next(slot: int) -> void:
	if _next_candidate >= _n_candidates:
		_slot_phase[slot] = Phase.IDLE
		return
	var candidate: PackedFloat32Array = _cma.candidate_at(_next_candidate) if _cma != null \
		else EsMath.candidate_at(_theta, sigma, _epsilons, _next_candidate)
	var runner: NcnnRunner = _runners[slot]
	if not runner.load_model_from_buffers(_param_bytes, NcnnWeights.bin_bytes(_spec, candidate)):
		_abort("candidate net failed to load.")
		return
	_slot_candidate[slot] = _next_candidate
	_next_candidate += 1
	_slot_episode[slot] = 0
	_slot_accum[slot] = 0.0
	_enter_freeze(slot)


## Phase 1 of an episode start: hold the candidate-independent neutral action for one decision
## window so any old-stream game event (a chase catch resolving on the pre-reset tick, a
## still-falling ragdoll) plays out against the OLD RNG stream — seeding now would let such an
## event consume the fresh stream's first draws for SOME candidates and not others, exactly the
## spawn-luck ranking noise seed_games exists to cancel.
func _enter_freeze(slot: int) -> void:
	_agents[slot].set_action(_neutral_action)
	_slot_phase[slot] = Phase.FREEZE


## Phase 2: seed the game (common random numbers), announce, and request the reset. The agent
## applies needs_reset on ITS next tick; the first decision follows on the next decision tick.
func _seed_and_reset(slot: int) -> void:
	var agent: Node = _agents[slot]
	if seed_games and _slot_game[slot] != null:
		_slot_game[slot].seed_rng(EsMath.episode_seed(rng_seed, _generation, _slot_episode[slot]))
	episode_starting.emit(slot, _slot_candidate[slot], _generation, _slot_episode[slot])
	agent.set_action(_neutral_action)
	agent.needs_reset = true
	agent.set_done_false()
	_slot_decisions[slot] = 0
	_slot_phase[slot] = Phase.SEEDED


func _physics_process(_delta: float) -> void:
	var decision_tick := _tick % action_repeat == 0
	_tick += 1
	var any_active := false
	for slot in range(_agents.size()):
		if _slot_phase[slot] == Phase.IDLE:
			continue
		any_active = true
		_step_slot(slot, decision_tick)
		if not is_physics_processing():
			return  # _step_slot aborted mid-generation
	if not any_active:
		_finish_generation()


func _step_slot(slot: int, decision_tick: bool) -> void:
	var agent: Node = _agents[slot]
	# Drain the reward EVERY tick: only RUNNING episodes score it. Draining per tick (rather
	# than per decision window) means a game-side terminal that self-resets the agent mid-window
	# (3dball-style) can't leak its NEXT episode's reward into the finished candidate — we see
	# `done` on the tick it appears and stop scoring immediately.
	var r: float = agent.get_reward()
	agent.zero_reward()
	match _slot_phase[slot]:
		Phase.FREEZE:
			if decision_tick:
				_seed_and_reset(slot)
		Phase.SEEDED:
			if agent.get_done():
				# Terminal before the candidate's first decision — hostile spawn or leftover
				# momentum the candidate never played. Retry the SAME seeded episode (capped).
				agent.set_done_false()
				if _slot_restarts[slot] < MAX_PHANTOM_RESTARTS:
					_slot_restarts[slot] += 1
					_seed_and_reset(slot)
					return
			if decision_tick:
				_decide(slot)  # first decision of the episode
				_slot_phase[slot] = Phase.RUNNING
		Phase.RUNNING:
			_slot_accum[slot] += r
			if agent.get_done():
				agent.set_done_false()
				_end_episode(slot)
				return
			if decision_tick:
				if _slot_decisions[slot] >= episode_decisions:
					_end_episode(slot)
					return
				_decide(slot)


func _decide(slot: int) -> void:
	var obs: Array = _agents[slot].get_obs()["obs"]
	var out: PackedFloat32Array = _runners[slot].run_inference(PackedFloat32Array(obs))
	if out.is_empty():
		_abort("inference failed.")
		return
	_agents[slot].set_action(ActionDecode.decode_actions(out, _action_space))
	_slot_decisions[slot] += 1


func _end_episode(slot: int) -> void:
	_slot_restarts[slot] = 0
	_slot_episode[slot] += 1
	if _slot_episode[slot] < episodes_per_candidate:
		_enter_freeze(slot)
		return
	_fitness[_slot_candidate[slot]] = _slot_accum[slot] / float(episodes_per_candidate)
	_assign_next(slot)


func _finish_generation() -> void:
	var mean := 0.0
	var best := -INF
	for f in _fitness:
		mean += float(f)
		best = maxf(best, float(f))
	mean /= float(_fitness.size())
	print("[ESTrainer] generation %d/%d · mean fitness %.3f · best %.3f"
		% [_generation + 1, generations, mean, best])
	# Stats first, THEN blessing: generation_finished carries this generation's mean, and any
	# checkpoint_saved that follows belongs to it — consumers that pair the two (the Evolution
	# Lab HUD's best-so-far) would otherwise read a one-generation-stale mean at blessing time.
	generation_finished.emit(_generation, mean, best)
	# Bless BEFORE the update: the measured mean belongs to the population around the CURRENT θ;
	# the post-update θ has never been evaluated and may be worse (noisy ranks, big alpha step).
	if mean > _best_mean:
		_best_mean = mean
		_save_checkpoint("%s_best" % checkpoint_stem)
	if _cma != null:
		_cma.update(_fitness)
		_theta = _cma.mean_vector()
	else:
		var shaped: Array = EsMath.centered_ranks(_fitness)
		_theta = EsMath.es_update(_theta, _epsilons, shaped, sigma, alpha)
	if checkpoint_every > 0 and (_generation + 1) % checkpoint_every == 0:
		_save_checkpoint("%s_gen%d" % [checkpoint_stem, _generation + 1])
	_generation += 1
	if _generation >= generations:
		_save_checkpoint("%s_final" % checkpoint_stem)
		set_physics_process(false)
		_restore_agents()
		print("[ESTrainer] done · best mean fitness %.3f · checkpoints in %s" % [_best_mean, out_dir])
		training_finished.emit(_best_mean)
		if exit_on_finish:
			get_tree().quit(0)
		return
	_start_generation()


## Hand the agents back the horizon this trainer overrode — an embedded scene (demo, test) keeps
## running after training and its agents must self-reset normally again.
func _restore_agents() -> void:
	for i in range(_saved_reset_after.size()):
		if _saved_reset_after[i] != null:
			_agents[i].reset_after = _saved_reset_after[i]


## Write the current θ as a deploy-ready ncnn pair: <out_dir>/<stem>.ncnn.{param,bin}.
func _save_checkpoint(stem: String) -> void:
	var err := DirAccess.make_dir_recursive_absolute(out_dir)
	if err != OK and err != ERR_ALREADY_EXISTS:
		push_error("ESTrainer: cannot create out_dir '%s' (err %d)." % [out_dir, err])
		return
	var param_path := out_dir.path_join(stem + ".ncnn.param")
	var bin_path := out_dir.path_join(stem + ".ncnn.bin")
	var pf := FileAccess.open(param_path, FileAccess.WRITE)
	var bf := FileAccess.open(bin_path, FileAccess.WRITE)
	if pf == null or bf == null:
		push_error("ESTrainer: cannot write checkpoint '%s'." % param_path)
		return
	pf.store_buffer(_param_bytes)
	bf.store_buffer(NcnnWeights.bin_bytes(_spec, _theta))
	pf = null  # close before announcing: consumers may reload the pair from the signal
	bf = null
	checkpoint_saved.emit(stem, param_path, bin_path)


## Fail loud and STOP: push the error, halt the loop, restore the agents, and (for CLI runs)
## exit nonzero so a headless training run can never hang silently on a configuration mistake.
func _abort(message: String) -> void:
	push_error("ESTrainer: " + message)
	set_physics_process(false)
	_restore_agents()
	if exit_on_finish:
		get_tree().quit(1)


# --- Introspection (tests, HUDs) ---

func current_theta() -> PackedFloat32Array:
	return _theta.duplicate()

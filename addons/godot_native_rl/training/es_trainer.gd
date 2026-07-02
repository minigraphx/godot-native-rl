extends Node
# ESTrainer (#131): a native, in-engine training loop — no Python, no socket, no backprop.
#
# A drop-in replacement for the NcnnSync node that trains instead of bridging: it drives the same
# agent contract NcnnSync does (group "AGENT"; get_obs()["obs"], set_action(dict),
# get_reward()/zero_reward(), get_done()/set_done_false(), needs_reset, get_action_space()) at the
# same action_repeat cadence, but the actions come from CANDIDATE nets the trainer itself evolves:
# an OpenAI-ES optimizer (es_math.gd) perturbs a flat weight vector θ, each candidate becomes a
# live ncnn net via the θ ⇄ buffers codec (ncnn_weights.gd) + NcnnRunner.load_model_from_buffers,
# and one episode's return (the existing reward system) is its fitness. Multi-agent scenes (e.g.
# ParallelArena tiles) evaluate one candidate per agent slot in parallel waves.
#
# The training artifact IS the deploy artifact: checkpoints are ncnn .param/.bin pairs, loadable
# by every controller unchanged — there is no export step. Warm-starting from a shipped net is the
# same codec in reverse (warm_start_*_path).
#
# Scope/trade-offs: episodes are the fixed horizon the controllers already implement
# (`reset_after`); flat float obs only (fail loud otherwise); ES is sample-inefficient — this is
# for small nets, dense rewards, and self-contained/on-device learning, not a PPO/SAC replacement.
# Place the trainer AFTER the agents in the scene tree (like Sync) so agents step first each tick.

const EsMath = preload("res://addons/godot_native_rl/training/es_math.gd")
const NcnnWeights = preload("res://addons/godot_native_rl/training/ncnn_weights.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")

signal generation_finished(generation: int, mean_fitness: float, best_fitness: float)
signal training_finished(best_mean_fitness: float)
## Emitted right before a candidate's episode reset is requested. Connect this to re-seed the
## game's RNG per generation (common random numbers): giving every candidate in a generation the
## SAME spawn sequence removes environment luck from the fitness comparison — without it, spawn
## noise can drown the between-candidate signal entirely (flat fitness, no learning).
signal candidate_starting(slot: int, candidate_index: int, generation: int)

@export var agent_group := "AGENT"
@export var hidden_dims: Array[int] = [16]
@export var hidden_activation := "relu"  # relu | tanh | sigmoid | "" (see ncnn_weights.gd)
@export var half_population := 8  ## candidates per generation = 2 * half_population (antithetic)
@export var sigma := 0.1  ## perturbation stddev
@export var alpha := 0.05  ## learning rate
@export var generations := 100
@export var action_repeat := 8  ## physics ticks between control decisions (NcnnSync convention)
@export var rng_seed := 0
@export var speed_up := 1.0  ## same mechanism as NcnnSync: scales physics ticks + time_scale
@export var out_dir := "user://es_checkpoints"
@export var checkpoint_stem := "es_policy"
@export var checkpoint_every := 0  ## write <stem>_gen<N> every N generations (0 = final/best only)
## Warm-start θ from a shipped net (both paths or neither). The net must match the spec this
## trainer builds ([obs_dim] + hidden_dims + [action_dim], hidden_activation) — fail loud if not.
@export var warm_start_param_path := ""
@export var warm_start_bin_path := ""

var _agents: Array = []
var _runners: Array = []
var _spec: Dictionary = {}
var _param_bytes := PackedByteArray()
var _theta := PackedFloat32Array()
var _action_space: Dictionary = {}
var _rng := RandomNumberGenerator.new()
var _epsilons: Array = []
var _candidates: Array = []
var _fitness: Array = []
var _next_candidate := 0
var _slot_candidate: Array = []  # candidate index under evaluation per agent slot; -1 = idle
var _slot_fitness: Array = []
var _slot_fresh: Array = []  # true right after a reset boundary: discard that stale reward read
var _generation := 0
var _tick := 0
var _best_mean := -INF
var _finished := false


func _ready() -> void:
	# Same speedup mechanism as NcnnSync (headless training runs faster than real time).
	if speed_up != 1.0:
		Engine.physics_ticks_per_second = int(speed_up * 60.0)
		Engine.time_scale = speed_up
	_rng.seed = rng_seed
	set_physics_process(false)
	# Agents register their group in their own _ready; initialize after the whole scene is in.
	_late_init.call_deferred()


func _late_init() -> void:
	_agents = get_tree().get_nodes_in_group(agent_group)
	if _agents.is_empty():
		push_error("ESTrainer: no agents found in group '%s'." % agent_group)
		return
	var first_obs: Dictionary = _agents[0].get_obs()
	if not first_obs.has("obs"):
		push_error("ESTrainer: agent get_obs() has no flat 'obs' key; image obs are unsupported.")
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
		return  # mlp_spec already pushed the error
	_param_bytes = NcnnWeights.param_text(_spec).to_utf8_buffer()
	_theta = _initial_theta()
	if _theta.is_empty():
		return
	for agent in _agents:
		var runner := NcnnRunner.new()
		runner.input_blob_name = "in0"
		runner.output_blob_name = "out0"
		add_child(runner)
		_runners.append(runner)
	_slot_candidate.resize(_agents.size())
	_slot_fitness.resize(_agents.size())
	_slot_fresh.resize(_agents.size())
	print("[ESTrainer] net %s (θ=%d floats) · population %d · %d agent slot(s) · %d generations"
		% [str(dims), _theta.size(), 2 * half_population, _agents.size(), generations])
	_start_generation()
	set_physics_process(true)


func _initial_theta() -> PackedFloat32Array:
	if warm_start_param_path.is_empty() != warm_start_bin_path.is_empty():
		push_error("ESTrainer: set both warm_start paths or neither.")
		return PackedFloat32Array()
	if warm_start_bin_path.is_empty():
		return NcnnWeights.init_theta(_spec, _rng)
	var bin := FileAccess.get_file_as_bytes(warm_start_bin_path)
	var theta: PackedFloat32Array = NcnnWeights.theta_from_bin(_spec, bin)
	if theta.is_empty():
		push_error("ESTrainer: warm-start net '%s' does not match the spec this scene builds."
			% warm_start_bin_path)
	return theta


func _start_generation() -> void:
	_epsilons = EsMath.sample_epsilons(_rng, half_population, _theta.size())
	_candidates = EsMath.antithetic_candidates(_theta, sigma, _epsilons)
	_fitness = []
	_fitness.resize(_candidates.size())
	_fitness.fill(0.0)
	_next_candidate = 0
	for slot in range(_agents.size()):
		_slot_fitness[slot] = 0.0
		_assign_next(slot)


func _assign_next(slot: int) -> void:
	if _next_candidate >= _candidates.size():
		_slot_candidate[slot] = -1
		return
	var runner: NcnnRunner = _runners[slot]
	if not runner.load_model_from_buffers(_param_bytes, NcnnWeights.bin_bytes(_spec, _candidates[_next_candidate])):
		push_error("ESTrainer: candidate net failed to load — aborting training.")
		set_physics_process(false)
		return
	_slot_candidate[slot] = _next_candidate
	_slot_fitness[slot] = 0.0
	_slot_fresh[slot] = true
	_next_candidate += 1
	candidate_starting.emit(slot, _next_candidate - 1, _generation)
	var agent: Node = _agents[slot]
	agent.needs_reset = true  # the agent applies it on ITS next tick (fresh episode + zeroed reward)
	agent.set_done_false()


func _physics_process(_delta: float) -> void:
	if _tick % action_repeat != 0:
		_tick += 1
		return
	_tick += 1
	var any_active := false
	for slot in range(_agents.size()):
		var cand: int = _slot_candidate[slot]
		if cand < 0:
			continue
		any_active = true
		var agent: Node = _agents[slot]
		var r: float = agent.get_reward()
		agent.zero_reward()
		if _slot_fresh[slot]:
			_slot_fresh[slot] = false  # reward accrued across the reset boundary isn't this candidate's
		else:
			_slot_fitness[slot] += r
		if agent.get_done():
			agent.set_done_false()
			_fitness[cand] = _slot_fitness[slot]
			_assign_next(slot)
			continue
		var obs: Array = agent.get_obs()["obs"]
		var out: PackedFloat32Array = _runners[slot].run_inference(PackedFloat32Array(obs))
		if out.is_empty():
			push_error("ESTrainer: inference failed — aborting training.")
			set_physics_process(false)
			return
		agent.set_action(ActionDecode.decode_actions(out, _action_space))
	if not any_active and _next_candidate >= _candidates.size():
		_finish_generation()


func _finish_generation() -> void:
	var shaped: Array = EsMath.centered_ranks(_fitness)
	_theta = EsMath.es_update(_theta, _epsilons, shaped, sigma, alpha)
	var mean := 0.0
	var best := -INF
	for f in _fitness:
		mean += float(f)
		best = maxf(best, float(f))
	mean /= float(_fitness.size())
	print("[ESTrainer] generation %d/%d · mean fitness %.3f · best %.3f"
		% [_generation + 1, generations, mean, best])
	if mean > _best_mean:
		_best_mean = mean
		_save_checkpoint("%s_best" % checkpoint_stem)
	if checkpoint_every > 0 and (_generation + 1) % checkpoint_every == 0:
		_save_checkpoint("%s_gen%d" % [checkpoint_stem, _generation + 1])
	generation_finished.emit(_generation, mean, best)
	_generation += 1
	if _generation >= generations:
		_save_checkpoint("%s_final" % checkpoint_stem)
		_finished = true
		set_physics_process(false)
		print("[ESTrainer] done · best mean fitness %.3f · checkpoints in %s" % [_best_mean, out_dir])
		training_finished.emit(_best_mean)
		return
	_start_generation()


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
	pf.store_string(NcnnWeights.param_text(_spec))
	bf.store_buffer(NcnnWeights.bin_bytes(_spec, _theta))


# --- Introspection (tests, HUDs) ---

func current_theta() -> PackedFloat32Array:
	return _theta.duplicate()


func current_generation() -> int:
	return _generation


func is_finished() -> bool:
	return _finished

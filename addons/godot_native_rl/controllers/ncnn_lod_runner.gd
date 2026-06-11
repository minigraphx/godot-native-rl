extends Node
# Level-of-Detail policy switching (#21, novel-addons spec §3 B5): runs a cheap "reflex" ncnn net
# on most frames and an accurate "deliberative" ncnn net only every `interval` frames (or on a
# significant state change). Exactly ONE inference runs per frame — the reflex net carries the
# cheap frames, the deliberative net refreshes periodically — so the expensive net's cost is paid
# at ~1/interval the rate. Genuinely new in game RL, and only viable because we statically link the
# native inference layer (two resident nets, switched game-side, no per-call runtime cost).
#
# Both nets must share the obs and output contract (same input width, same action head); they
# differ only in capacity. Wire two NcnnRunners by model path, then call decide(obs) each frame.

const LodScheduler = preload("res://addons/godot_native_rl/controllers/lod_scheduler.gd")

@export var reflex_param_path: String = ""
@export var reflex_bin_path: String = ""
@export var deliberative_param_path: String = ""
@export var deliberative_bin_path: String = ""
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"
## The deliberative net runs every `deliberative_interval` frames (>= 1). 1 disables LOD (the
## deliberative net runs every frame). Changing it at runtime updates the live cadence. Clamped to
## >= 1 in the setter so the stored/inspector value matches the effective cadence (LodScheduler
## clamps too, but the export would otherwise show e.g. 0 while the scheduler runs at 1).
@export var deliberative_interval: int = 4:
	set(value):
		deliberative_interval = maxi(value, 1)
		if _scheduler != null:
			_scheduler.set_interval(deliberative_interval)

var _scheduler: LodScheduler
var _reflex
var _deliberative
var _last_deliberative_logits := PackedFloat32Array()

func _ready() -> void:
	# Idempotent: if already configured (e.g. setup_for_test injected runners + scheduler before
	# the node was added to the tree), don't clobber them — mirrors NcnnCrowdController's guard.
	if _scheduler != null:
		return
	_scheduler = LodScheduler.new(deliberative_interval)
	_reflex = _make_runner(reflex_param_path, reflex_bin_path)
	_deliberative = _make_runner(deliberative_param_path, deliberative_bin_path)

func _make_runner(param_path: String, bin_path: String):
	if param_path.is_empty() or bin_path.is_empty():
		push_error("NcnnLODRunner: reflex and deliberative model paths must be set.")
		return null
	var runner = NcnnRunner.new()
	runner.input_blob_name = input_blob_name
	runner.output_blob_name = output_blob_name
	add_child(runner)
	var param_bytes := FileAccess.get_file_as_bytes(param_path)
	var bin_bytes := FileAccess.get_file_as_bytes(bin_path)
	if param_bytes.is_empty() or bin_bytes.is_empty():
		push_error("NcnnLODRunner: cannot read model files '%s' / '%s'." % [param_path, bin_path])
		runner.queue_free()
		return null
	if not runner.load_model_from_buffers(param_bytes, bin_bytes):
		push_error("NcnnLODRunner: failed to load ncnn model '%s'." % param_path)
		runner.queue_free()
		return null
	return runner

## Test seam: inject pre-built runners + interval without file IO (mirrors the controllers'
## set_ncnn_runner_for_test).
func setup_for_test(reflex, deliberative, interval: int) -> void:
	_reflex = reflex
	_deliberative = deliberative
	_scheduler = LodScheduler.new(interval)

## Reset the LOD cadence (call on episode reset) so the next frame runs the deliberative net.
func reset() -> void:
	if _scheduler != null:
		_scheduler.reset()
	_last_deliberative_logits = PackedFloat32Array()

## Run inference for one frame. Returns a Dictionary:
##   { "logits": PackedFloat32Array, "tier": "reflex"|"deliberative", "ran_deliberative": bool }
## On a deliberative frame the deliberative net runs (and its logits are cached); otherwise the
## reflex net runs. `state_changed=true` forces the deliberative net this frame.
func decide(obs: PackedFloat32Array, state_changed: bool = false) -> Dictionary:
	if _scheduler == null:
		# decide() before _ready()/setup_for_test() — fail with a diagnostic, not a raw null crash
		# (mirrors the runner-null branch below; reset() guards the same way).
		push_error("NcnnLODRunner.decide: node not initialized (add it to the tree or call setup_for_test).")
		return {"logits": PackedFloat32Array(), "tier": "reflex", "ran_deliberative": false}
	var due: bool = _scheduler.tick(state_changed)
	var runner = _deliberative if due else _reflex
	if runner == null:
		push_error("NcnnLODRunner.decide: %s runner is not loaded." % ("deliberative" if due else "reflex"))
		return {"logits": PackedFloat32Array(), "tier": "deliberative" if due else "reflex", "ran_deliberative": due}
	var logits: PackedFloat32Array = runner.run_inference(obs)
	if due:
		_last_deliberative_logits = logits
	return {"logits": logits, "tier": "deliberative" if due else "reflex", "ran_deliberative": due}

## The most recent deliberative-net output (empty until the first deliberative frame). Useful when
## the reflex frames should be steered by the last accurate decision.
func last_deliberative_logits() -> PackedFloat32Array:
	return _last_deliberative_logits

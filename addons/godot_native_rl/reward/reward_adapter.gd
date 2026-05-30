class_name RewardAdapter
extends Node

var _pending := 0.0
var _reward_override = null   # set via bind_reward() for tests / non-child placement

# When added under a controller, trigger a re-scan so adapters created after the
# controller's _ready() (e.g. in a subclass _ready) are still collected for draining.
func _ready() -> void:
	var p := get_parent()
	if p != null and p.has_method("collect_reward_adapters"):
		p.collect_reward_adapters()

# Fire-and-forget: accumulate `delta` whenever `emitter` emits `signal_name`.
func on_signal(emitter: Object, signal_name: String, delta: float) -> void:
	_connect(emitter, signal_name, _make_scalar_handler(delta))

# Route a signal to a named event on the bound Reward (drives bonuses + progress rebasing).
func on_signal_event(emitter: Object, signal_name: String, event_name: String) -> void:
	_connect(emitter, signal_name, _make_event_handler(event_name))

func bind_reward(reward) -> void:
	_reward_override = reward

func drain() -> float:
	var r := _pending
	_pending = 0.0
	return r

# --- internals ---
func _resolve_reward():
	if _reward_override != null:
		return _reward_override
	var p := get_parent()
	if p != null and "reward_source" in p:
		return p.reward_source
	return null

func _make_scalar_handler(delta: float) -> Callable:
	return func() -> void:
		_pending += delta

func _make_event_handler(event_name: String) -> Callable:
	return func() -> void:
		var r = _resolve_reward()
		if r != null:
			r.trigger_event(event_name)

func _connect(emitter: Object, signal_name: String, handler: Callable) -> void:
	var argc := _signal_arg_count(emitter, signal_name)
	emitter.connect(signal_name, _trampoline(handler, argc))

func _signal_arg_count(emitter: Object, signal_name: String) -> int:
	for s in emitter.get_signal_list():
		if s["name"] == signal_name:
			return (s["args"] as Array).size()
	return 0

# Godot requires the callback arity to match the signal. Wrap the 0-arg handler in a
# correctly-sized shim for 0..4 emitted args (covers effectively all real signals).
func _trampoline(handler: Callable, argc: int) -> Callable:
	match argc:
		0: return func() -> void: handler.call()
		1: return func(_a) -> void: handler.call()
		2: return func(_a, _b) -> void: handler.call()
		3: return func(_a, _b, _c) -> void: handler.call()
		4: return func(_a, _b, _c, _d) -> void: handler.call()
		_:
			push_error("RewardAdapter: signals with >4 args are not supported (got %d)." % argc)
			return func() -> void: handler.call()

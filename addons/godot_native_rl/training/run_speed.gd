extends RefCounted
# Shared faster-than-real-time machinery for the training-loop nodes (NcnnSync, ESTrainer).
# One copy of the speedup rule so a fix reaches every trainer — the ES work (#131/#287) found
# that setting only ticks + time_scale leaves Engine.max_physics_steps_per_frame at its default
# of 8, silently capping large speedups to ~8 × fps ticks/s (a 13× throughput loss at 50×).


## Apply a speedup: scale physics ticks + time, and raise the per-frame physics-step cap so the
## configured rate is actually reachable. Idempotent and re-callable at runtime (live speed
## toggles): apply(1.0) restores the 60 Hz baseline (the step cap is only ever raised — harmless).
static func apply(speed_up: float) -> void:
	Engine.physics_ticks_per_second = int(speed_up * 60.0)
	Engine.time_scale = speed_up
	Engine.max_physics_steps_per_frame = maxi(Engine.max_physics_steps_per_frame, int(ceil(speed_up)) * 2)


## Parse `--key=value` (or `key=value`) cmdline tokens — the override convention every training
## script uses (speedup=, action_repeat=, port=, ...). Values are Strings; callers convert.
static func parse_cmdline_args() -> Dictionary:
	var arguments := {}
	for argument in OS.get_cmdline_args():
		if argument.find("=") > -1:
			var kv := argument.split("=")
			arguments[kv[0].lstrip("--")] = kv[1]
	return arguments


## Strict positive-int cmdline override (#290/#324): GDScript's `int("garbage")` is 0 and
## `int("1e3")` prefix-parses to 1, so an unvalidated override silently turns into a
## modulo-by-zero (SIGFPE in release) or truncates an overnight run to one generation. Returns
## `fallback` when the key is absent; pushes an error and returns -1 (callers fail loud) when
## the value is not a pure positive integer.
static func parse_positive_int(args: Dictionary, key: String, fallback: int) -> int:
	if not args.has(key):
		return fallback
	var raw := str(args[key])
	if not raw.is_valid_int() or int(raw) < 1:
		push_error("RunSpeed: cmdline override %s=%s is not a positive integer." % [key, raw])
		return -1
	return int(raw)


## Strict INT cmdline override, any sign (#332 — env_seed): rejects non-integers loudly.
## Returns `fallback` when absent; pushes an error and returns `fallback` on garbage (a seed has
## no sensible sentinel — corrupting comparability silently was the bug, so the error IS the fix).
static func parse_int_any(args: Dictionary, key: String, fallback: int) -> int:
	if not args.has(key):
		return fallback
	var raw := str(args[key])
	if not raw.is_valid_int():
		push_error("RunSpeed: cmdline override %s=%s is not an integer — using %d." % [key, raw, fallback])
		return fallback
	return int(raw)


## Strict positive-float cmdline override (#332): `to_float()` turns garbage into 0.0, which for
## speedups zeroes the tick rate and for socket timeouts means "wait forever" per their <= 0
## convention. Returns `fallback` when absent; pushes an error and returns -1.0 on a value that
## is not a positive number (callers decide: abort, or fall back).
static func parse_positive_float(args: Dictionary, key: String, fallback: float) -> float:
	if not args.has(key):
		return fallback
	var raw := str(args[key])
	if not raw.is_valid_float() or raw.to_float() <= 0.0:
		push_error("RunSpeed: cmdline override %s=%s is not a positive number." % [key, raw])
		return -1.0
	return raw.to_float()


## Strict FLOAT cmdline override, any sign (#332 — socket timeouts, whose documented convention
## is "<= 0 disables"): rejects non-numeric values loudly and returns `fallback` (an explicit
## numeric 0/negative passes through as the caller's disable semantics).
static func parse_float_any(args: Dictionary, key: String, fallback: float) -> float:
	if not args.has(key):
		return fallback
	var raw := str(args[key])
	if not raw.is_valid_float():
		push_error("RunSpeed: cmdline override %s=%s is not a number — using %s." % [key, raw, str(fallback)])
		return fallback
	return raw.to_float()

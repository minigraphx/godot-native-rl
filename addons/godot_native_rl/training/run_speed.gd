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

extends RefCounted

# Pure, node-free deadline helpers for bounding NcnnSync's socket poll loops.
# A timeout <= 0 means "no deadline" (infinite wait, opt-out), represented by -1.
# now_ms is a monotonic millisecond clock supplied by the caller (Time.get_ticks_msec()).

const INFINITE := -1

static func deadline_after(now_ms: int, timeout_ms: int) -> int:
	if timeout_ms <= 0:
		return INFINITE
	return now_ms + timeout_ms

static func is_expired(deadline_ms: int, now_ms: int) -> bool:
	return deadline_ms >= 0 and now_ms >= deadline_ms

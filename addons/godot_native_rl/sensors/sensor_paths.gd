extends RefCounted
# Shared NodePath → node resolver for sensors' `object_paths` exports (#329/#38).
#
# WHY object_paths exists at all: an exported typed NODE array (`Array[Node2D]`) in a
# hand-authored .tscn is stored as NodePaths that do NOT resolve at plain runtime instantiation
# ("Unable to convert array index 0 from NodePath to Object") — only single-Node exports get
# that treatment. NodePath arrays resolve fine, so sensors expose both and resolve here.
#
# Contract (#329): an unresolved entry returns null IN PLACE — the caller appends it so the slot
# is RESERVED and zero-fills, keeping the declared obs width. Silently shrinking the vector
# trains a policy on the wrong dim (the handshake sizes from the live obs) or crashes a shipped
# fixed-dim net with an opaque ncnn shape mismatch far from the root cause.


## Resolve `paths` relative to `sensor`. Returns one entry per path: the node when it exists and
## is a `required_class` (subclasses included), else null (with a loud error).
static func resolve(sensor: Node, paths: Array[NodePath], required_class: String) -> Array:
	var out: Array = []
	for p in paths:
		var n: Node = sensor.get_node_or_null(p)
		if n == null or not n.is_class(required_class):
			push_error("%s: object_paths entry '%s' does not resolve to a %s — slot reserved and ZERO-FILLED so the obs width stays as declared (fix the path)."
				% [sensor.name, str(p), required_class])
			out.append(null)
		else:
			out.append(n)
	return out


## The whole lazy-resolution body the four sensors share (#348): resolve `paths` and append the
## results (nulls included — they reserve zero-filling slots where width matters, and candidate
## unions skip them) into `dest`. Each sensor keeps only its ~3-line _paths_resolved guard.
static func append_resolved(sensor: Node, paths: Array[NodePath], required_class: String, dest) -> void:
	for n in resolve(sensor, paths, required_class):
		dest.append(n)

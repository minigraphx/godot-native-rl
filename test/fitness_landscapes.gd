extends RefCounted
# Shared fitness landscapes for the optimizer unit suites (test_es_math.gd, test_cma_math.gd) —
# the two files are the paired optimizer suites and must score candidates identically.


## Maximize -||x - target||^2 (optimum 0 at x == target).
static func sphere(x: PackedFloat32Array, target: Array) -> float:
	var total := 0.0
	for i in range(x.size()):
		var d := x[i] - float(target[i])
		total -= d * d
	return total

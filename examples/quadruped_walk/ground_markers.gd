extends Node3D
# Distance markers for the locomotion tracks (#229 visual polish): a RULER along the track edges
# so a viewer can SEE the creature move forward — on a flat green ground with no reference,
# forward motion is nearly invisible. Purely cosmetic: no physics, no collision, no effect on
# observations/rewards/training (a headless trainer just renders nothing).
#
# Reads as a ruler, not obstacles (#276): the original full-width stripes crossed the running
# lane and, from the low follow camera, looked like the hurdles demo's gates — viewers thought
# the walker was "running hurdles". Markers now stay OUT of the lane: paired short ticks at the
# track edges plus a billboarded `Label3D` distance readout ("5 m", "10 m", ...), with nothing
# drawn across the creature's path.

@export var count := 10          ## number of distance marks
@export var spacing := 5.0       ## metres between marks (Z); a "5 m" cadence reads clearly
@export var start_z := 0.0       ## first mark Z (creature starts at 0, runs +Z)
@export var width := 8.0         ## track X extent the ticks flank (lane between stays clear)
@export var tick_length := 1.2   ## X extent of each edge tick
@export var tick_depth := 0.18   ## Z thickness of each tick
@export var y := 0.02            ## just above the ground so it doesn't z-fight
@export var color := Color(0.85, 0.85, 0.9, 1.0)
@export var label_size := 0.55   ## world-space height of the distance text
@export var show_labels := true

func _ready() -> void:
	_build()

func _build() -> void:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	for i in range(maxi(count, 0)):
		var z := marker_z(i, start_z, spacing)
		for side in [-1.0, 1.0]:
			var tick := MeshInstance3D.new()
			var box := BoxMesh.new()
			box.size = Vector3(tick_length, 0.02, tick_depth)
			tick.mesh = box
			tick.material_override = mat
			tick.position = Vector3(side * (width - tick_length) * 0.5, y, z)
			add_child(tick)
		if show_labels:
			var label := Label3D.new()
			label.text = "%d m" % int(round(z))
			label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
			label.modulate = color
			label.pixel_size = label_size / 64.0  # ~label_size metres tall at default font size
			label.position = Vector3(-(width * 0.5 + 0.8), y + 0.35, z)
			add_child(label)

# Pure helper (testable): the Z position of marker `i`.
static func marker_z(i: int, p_start: float, p_spacing: float) -> float:
	return p_start + float(i) * p_spacing

# Back-compat alias for the original stripe naming (same math).
static func stripe_z(i: int, p_start: float, p_spacing: float) -> float:
	return marker_z(i, p_start, p_spacing)

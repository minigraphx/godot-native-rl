extends Node3D
# Distance markers for the locomotion tracks (#229 visual polish): code-built stripes laid across the
# track at a fixed Z spacing so a viewer can SEE the creature move forward — on a flat green ground
# with no reference, forward motion is nearly invisible. Purely cosmetic: no physics, no collision,
# no effect on observations/rewards/training (a headless trainer just renders nothing). Mirrors the
# HurdleTrack code-built pattern.

@export var count := 10          ## number of stripes
@export var spacing := 5.0       ## metres between stripes (Z); a "5 m" cadence reads clearly
@export var start_z := 0.0       ## first stripe Z (creature starts at 0, runs +Z)
@export var width := 8.0         ## stripe X extent (span the track)
@export var stripe_depth := 0.18 ## Z thickness of each stripe
@export var y := 0.02            ## just above the ground so it doesn't z-fight
@export var color := Color(0.85, 0.85, 0.9, 1.0)
@export var every_other_dim := true  ## dim alternate stripes so the cadence is easy to count

func _ready() -> void:
	_build()

func _build() -> void:
	var mat_bright := StandardMaterial3D.new()
	mat_bright.albedo_color = color
	var mat_dim := StandardMaterial3D.new()
	mat_dim.albedo_color = color.darkened(0.35)
	for i in range(maxi(count, 0)):
		var stripe := MeshInstance3D.new()
		var box := BoxMesh.new()
		box.size = Vector3(width, 0.02, stripe_depth)
		stripe.mesh = box
		stripe.material_override = mat_dim if (every_other_dim and i % 2 == 1) else mat_bright
		stripe.position = Vector3(0.0, y, start_z + float(i) * spacing)
		add_child(stripe)

# Pure helper (testable): the Z position of stripe `i`.
static func stripe_z(i: int, p_start: float, p_spacing: float) -> float:
	return p_start + float(i) * p_spacing

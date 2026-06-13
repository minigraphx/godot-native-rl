extends Camera3D
# Third-person follow camera for the FlyBy demo (#227). The previous static camera sat at a fixed
# transform and never tracked the plane, which flew into and out of frame (often filling the view
# with the plane mesh). This trails the plane from behind-and-above along its HORIZONTAL heading
# (roll/pitch are flattened so the horizon stays level), eases toward that pose, and looks at the
# plane — keeping it framed against the sky at a constant distance no matter where it flies.

@export var distance := 14.0   ## how far behind the plane (world units)
@export var height := 5.0      ## how far above the plane
@export var smooth := 5.0      ## position easing speed (higher = snappier follow)

var _game  # FlyByGame, duck-typed: provides get_plane_xform() -> Transform3D

func _ready() -> void:
	_game = get_parent()

# Pure helper (unit-testable): desired camera position for a given plane transform. Trails behind
# the plane's heading flattened to horizontal, lifted by `height`, so the camera never rolls/pitches.
func follow_position(plane_xform: Transform3D) -> Vector3:
	var fwd := -plane_xform.basis.z
	fwd.y = 0.0
	if fwd.length() < 0.001:
		fwd = Vector3.FORWARD
	fwd = fwd.normalized()
	return plane_xform.origin - fwd * distance + Vector3.UP * height

func _process(delta: float) -> void:
	if _game == null or not _game.has_method("get_plane_xform"):
		return
	var xform: Transform3D = _game.get_plane_xform()
	var target := follow_position(xform)
	global_position = global_position.lerp(target, clampf(smooth * delta, 0.0, 1.0))
	look_at(xform.origin, Vector3.UP)

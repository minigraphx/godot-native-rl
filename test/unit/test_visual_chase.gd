extends SceneTree
# Unit tests for the visual-chase example (#35): the pure rasterizer + the agent's image-only
# observation contract. No rendering, no physics — the whole point of code-rasterized obs.

const Harness = preload("res://test/harness.gd")
const VObs = preload("res://examples/visual_chase/visual_chase_obs.gd")
const Agent = preload("res://examples/visual_chase/visual_chase_agent.gd")
const ChaseGame = preload("res://examples/chase_the_target/chase_game.gd")

func _pixel(bytes: PackedByteArray, x: int, y: int, w: int) -> Array:
	var i := (y * w + x) * 3
	return [bytes[i], bytes[i + 1], bytes[i + 2]]

func _initialize() -> void:
	var h = Harness.new()
	var arena := Vector2(1000, 600)

	# to_pixel maps arena corners/edges into bounds.
	h.assert_eq(VObs.to_pixel(Vector2(0, 0), arena, 36, 36), Vector2i(0, 0), "origin -> 0,0")
	h.assert_eq(VObs.to_pixel(Vector2(1000, 600), arena, 36, 36), Vector2i(35, 35), "far corner clamped to 35,35")
	h.assert_eq(VObs.to_pixel(Vector2(500, 300), arena, 36, 36), Vector2i(18, 18), "center -> 18,18")

	# rasterize: correct buffer size, blocks at the right pixels, agent over target on overlap.
	var bytes := VObs.rasterize(Vector2(500, 300), Vector2(100, 100), arena, 36, 36)
	h.assert_eq(bytes.size(), 36 * 36 * 3, "buffer is 36x36x3")
	var agent_px := _pixel(bytes, 18, 18, 36)
	h.assert_eq(agent_px[2], 255, "agent pixel is blue-dominant")
	var target_px := _pixel(bytes, 3, 6, 36)
	h.assert_eq(target_px[0], 235, "target pixel is red-dominant")
	var bg_px := _pixel(bytes, 30, 30, 36)
	h.assert_eq(bg_px[0], 20, "background untouched")
	var overlap := VObs.rasterize(Vector2(500, 300), Vector2(500, 300), arena, 36, 36)
	h.assert_eq(_pixel(overlap, 18, 18, 36)[2], 255, "agent drawn over target on overlap")

	# Block at the corner clamps instead of crashing.
	var corner := VObs.rasterize(Vector2(0, 0), Vector2(1000, 600), arena, 36, 36)
	h.assert_eq(_pixel(corner, 0, 0, 36)[2], 255, "corner agent drawn, no OOB")

	# make_image round-trips the bytes.
	var img := VObs.make_image(bytes, 36, 36)
	h.assert_eq(img.get_width(), 36, "image width")
	h.assert_eq(img.get_format(), Image.FORMAT_RGB8, "image format RGB8")

	# --- Agent contract ---
	var game = ChaseGame.new()
	get_root().add_child(game)
	var body_a := Node2D.new()
	var body_t := Node2D.new()
	get_root().add_child(body_a)
	get_root().add_child(body_t)
	game._agent_body = body_a
	game._target = body_t
	var agent = Agent.new()
	get_root().add_child(agent)
	agent._game = game

	var space: Dictionary = agent.get_obs_space()
	h.assert_true("camera_2d" in space, "obs space declares camera_2d")
	h.assert_eq(space["camera_2d"]["size"], [36, 36, 3], "image space 36x36x3")
	var obs: Dictionary = agent.get_obs()
	h.assert_true("camera_2d" in obs, "obs carries camera_2d")
	h.assert_eq(String(obs["camera_2d"]).length(), 36 * 36 * 3 * 2, "hex length = 2 chars/byte")
	var inf_img: Image = agent.get_inference_image()
	h.assert_true(inf_img != null and inf_img.get_width() == 36, "inference image supplied (deploy route)")

	h.finish(self)

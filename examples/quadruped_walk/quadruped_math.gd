class_name QuadrupedMath
extends RefCounted
# Pure, scene-free locomotion math: action clamping/scaling, upright signal, progress
# shaping, observation composition. Unit-tested in test/unit/test_quadruped_math.gd.

static func clamp_action(v: float) -> float:
	return clampf(v, -1.0, 1.0)

# Map a policy action (post-clamp) to a hinge motor target velocity.
static func action_to_motor_velocity(action: float, max_speed: float) -> float:
	return clamp_action(action) * max_speed

# Cosine of the torso's tilt from vertical. body_up is the torso basis Y column in world space.
static func upright_dot(body_up: Vector3) -> float:
	return body_up.dot(Vector3.UP)

# Distance closed toward the goal since last step (positive = progress).
static func progress_delta(prev_dist: float, cur_dist: float) -> float:
	return prev_dist - cur_dist

# Concatenate the observation in the documented order:
# joint_angles + joint_velocities + body_up(3) + body_local_vel(3) + dir_to_finish(3) + foot_contacts(4)
static func compose_obs(joint_angles: Array, joint_velocities: Array, body_up: Vector3, body_local_vel: Vector3, dir_to_finish: Array, foot_contacts: Array) -> Array:
	var out: Array = []
	out.append_array(joint_angles)
	out.append_array(joint_velocities)
	out.append_array([body_up.x, body_up.y, body_up.z])
	out.append_array([body_local_vel.x, body_local_vel.y, body_local_vel.z])
	out.append_array(dir_to_finish)
	out.append_array(foot_contacts)
	return out

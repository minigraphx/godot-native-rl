extends SceneTree

# Guards the algorithm-agnostic deploy contract (issue #45): the deploy path keys ONLY off the
# policy output's *shape* + the action_space `action_type`, never the RL algorithm that produced
# the weights. PPO/A2C logits, DQN Q-values, and SAC/TD3/DDPG deterministic actors all flow through
# the SAME ActionDecode path and select the same action — so a non-PPO policy deploys unchanged.
# (This is a decode/runtime guard; it needs no training run. The full trained non-PPO regression —
# SB3 SAC/DQN end-to-end → ncnn → behavioral check — is the separate needs-training-run slice.)

const Harness = preload("res://test/harness.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")

func _initialize() -> void:
	var h := Harness.new()

	# --- Discrete head: PPO/A2C logits and DQN Q-values decode identically (argmax). ---
	# The decoder never normalizes or interprets the values, so unbounded Q-values (DQN) and
	# pre-softmax logits (PPO) hit the same argmax and pick the same action for the same ranking.
	var disc := {"move": {"size": 4, "action_type": "discrete"}}
	var ppo_logits := PackedFloat32Array([0.10, 0.90, 0.20, 0.05])      # PPO/A2C pre-softmax logits
	var dqn_qvalues := PackedFloat32Array([3.0, 12.0, 7.5, 1.0])        # DQN action-value estimates
	h.assert_eq(ActionDecode.decode_actions(ppo_logits, disc), {"move": 1},
		"PPO/A2C logits -> argmax action 1")
	h.assert_eq(ActionDecode.decode_actions(dqn_qvalues, disc), {"move": 1},
		"DQN Q-values -> SAME argmax action 1 (same code path)")

	# Equal ranking -> equal decoded action, regardless of the values' scale/meaning.
	var logits_b := PackedFloat32Array([2.0, -1.0, 0.0])
	var qvals_b := PackedFloat32Array([105.3, 4.2, 88.0])              # different scale, same argmax
	var space_b := {"move": {"size": 3, "action_type": "discrete"}}
	h.assert_eq(ActionDecode.decode_actions(logits_b, space_b),
		ActionDecode.decode_actions(qvals_b, space_b),
		"discrete decode depends on ranking, not algorithm/scale")

	# --- Continuous head: TD3/DDPG deterministic actor (raw mean, already in range). ---
	# Off-policy continuous actors output the action directly; deploy passes the mean through.
	var td3 := {"steer": {"size": 2, "action_type": "continuous"}}
	var r_td3 := ActionDecode.decode_actions(PackedFloat32Array([0.4, -0.7]), td3)
	h.assert_true(r_td3.has("steer") and r_td3["steer"].size() == 2,
		"TD3/DDPG continuous -> 2-vector")
	h.assert_true(absf(r_td3["steer"][0] - 0.4) < 1e-6 and absf(r_td3["steer"][1] - (-0.7)) < 1e-6,
		"TD3/DDPG deterministic actor -> raw mean passes through (== PPO continuous path)")

	# --- Continuous head: SAC squashed-Gaussian, deployed as tanh(mean) via per-key squash. ---
	# SAC's policy squashes the Gaussian mean with tanh into [-1, 1]; the squash flag reproduces the
	# exact deploy-time transform, sharing the same continuous code path as PPO/TD3.
	var sac := {"steer": {"size": 2, "action_type": "continuous", "squash": true}}
	var r_sac := ActionDecode.decode_actions(PackedFloat32Array([0.4, -0.7]), sac)
	h.assert_true(absf(r_sac["steer"][0] - tanh(0.4)) < 1e-6 and absf(r_sac["steer"][1] - tanh(-0.7)) < 1e-6,
		"SAC squashed actor -> tanh(mean) via shared continuous path")

	# --- Mixed multi-head (e.g. a hybrid actor): per-key decode is algorithm-blind end to end. ---
	# A discrete "fire" + continuous "steer" head decodes the same whether trained by PPO, a
	# DQN+continuous hybrid, or anything else — the contract is purely shape + action_type.
	var hybrid := {"fire": {"size": 2, "action_type": "discrete"},
		"steer": {"size": 2, "action_type": "continuous"}}
	var r_hybrid := ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.9, 0.3, -0.3]), hybrid)
	h.assert_eq(r_hybrid["fire"], 1, "hybrid: discrete head argmax")
	h.assert_true(absf(r_hybrid["steer"][0] - 0.3) < 1e-6 and absf(r_hybrid["steer"][1] - (-0.3)) < 1e-6,
		"hybrid: continuous head pass-through")

	h.finish(self)

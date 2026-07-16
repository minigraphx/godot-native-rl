"""godot_rl action-masking helpers + mask-aware envs for sb3-contrib MaskablePPO (#385).

Pure helpers (`flatten_mask_dict`, `discrete_action_layout`, `masks_from_response`) are
import-free at module scope so the flattening logic stays unit-testable without torch/SB3/
godot_rl. The env classes lazy-guard their heavy deps exactly like `godot_env_truncation.py`:

- `MaskAwareGodotEnv(TruncationAwareGodotEnv)` stashes the wire's additive per-agent
  `action_mask` field on every `step_recv`/`reset` as `self.last_action_masks` (flat 0/1 rows
  of uniform length, all-ones when the scene sends no masks), on top of the #12 real
  (terminated, truncated) split.
- `MaskableStableBaselinesGodotEnv(StableBaselinesGodotEnv)` is the sb3-contrib seam
  (`sb3_contrib/common/maskable/utils.py`): `get_action_masks` calls
  `np.stack(env.env_method("action_masks"))` and `is_masking_supported` calls
  `env.has_attr("action_masks")` on a VecEnv — so it overrides exactly those two, returning
  the cached masks env-major in the SAME order `step()` concatenates obs. Masking is
  discrete/multi-discrete only (Box keys are skipped).
"""

try:
    # Guarded like godot_env_truncation.py: godot_rl (and the truncation subclass built on it)
    # are needed to *use* MaskAwareGodotEnv, never to import the pure helpers.
    try:
        from scripts.godot_env_truncation import (
            TruncationAwareGodotEnv,
            split_done_truncated,
        )
    except ImportError:  # flat import when scripts/ itself is on sys.path (trainer style)
        from godot_env_truncation import TruncationAwareGodotEnv, split_done_truncated
    _HAVE_GODOT_RL = True
except Exception:  # noqa: BLE001  (keep the pure helpers importable without the dep)
    TruncationAwareGodotEnv = object
    _HAVE_GODOT_RL = False

try:
    # Separate guard: the SB3 wrapper pulls in stable_baselines3 (torch).
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    _HAVE_SB3_WRAPPER = True
except Exception:  # noqa: BLE001
    StableBaselinesGodotEnv = object
    _HAVE_SB3_WRAPPER = False


def flatten_mask_dict(mask_dict, action_keys, key_sizes):
    """Concatenate per-key 0/1 masks in `action_keys` order (the env_info action-space order the
    MultiDiscrete nvec was built from). A missing/None key (or a None mask_dict) defaults to
    all-ones (#385)."""
    out = []
    md = mask_dict or {}
    for key in action_keys:
        m = md.get(key)
        if m is None:
            out.extend([1] * key_sizes[key])
        else:
            out.extend(int(v) for v in m)
    return out


def discrete_action_layout(action_space_dict):
    """(action_keys, key_sizes) for the DISCRETE keys of one agent's gym Dict action space, in
    the space's own (OrderedDict) order — the order godot_rl's MultiDiscrete nvec was built from
    (#385). Continuous (Box) keys are skipped (masking is discrete-only)."""
    spaces = getattr(action_space_dict, "spaces", action_space_dict)
    action_keys = []
    key_sizes = {}
    for key, space in spaces.items():
        if type(space).__name__ == "Discrete":  # duck-typed: keeps the helper import-free
            action_keys.append(key)
            key_sizes[key] = int(space.n)
    return action_keys, key_sizes


def masks_from_response(response, action_keys, key_sizes, n_agents):
    """Per-agent flat 0/1 masks from a wire response dict (#385). `response.get("action_mask")`
    is a per-agent list of keyed dicts (or absent -> every agent all-ones). Each row has length
    `sum(key_sizes[k] for k in action_keys)` — uniform, so sb3-contrib's `np.stack` succeeds."""
    per_agent = response.get("action_mask")
    if per_agent is None:
        per_agent = [None] * n_agents
    return [
        flatten_mask_dict(per_agent[i] if i < len(per_agent) else None, action_keys, key_sizes)
        for i in range(n_agents)
    ]


if _HAVE_GODOT_RL:
    class MaskAwareGodotEnv(TruncationAwareGodotEnv):
        """TruncationAwareGodotEnv that also stashes the wire's per-agent `action_mask` (#385).

        `step_recv`/`reset` replicate the base bodies (the base consumes the response dict
        internally, so there is no hook) and additionally cache `self.last_action_masks` from
        the SAME response before returning."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._mask_action_keys, self._mask_key_sizes = discrete_action_layout(
                self.action_spaces[0])
            total_len = sum(self._mask_key_sizes[k] for k in self._mask_action_keys)
            self.last_action_masks = [[1] * total_len for _ in range(self.num_envs)]

        def step_recv(self):
            response = self._get_json_dict()
            response["obs"] = self._process_obs(response["obs"])
            self.last_action_masks = masks_from_response(
                response, self._mask_action_keys, self._mask_key_sizes, self.num_envs)
            # Kept for backward compatibility if the plugin doesn't send info (as upstream).
            default_info = [{}] * len(response["done"])
            terminated, truncated = split_done_truncated(
                response["done"], response.get("truncated"))
            return (
                response["obs"],
                response["reward"],
                terminated,
                truncated,
                response.get("info", default_info),
            )

        def reset(self, seed=None):
            message = {
                "type": "reset",
            }
            self._send_as_json(message)
            response = self._get_json_dict()
            response["obs"] = self._process_obs(response["obs"])
            assert response["type"] == "reset"
            self.last_action_masks = masks_from_response(
                response, self._mask_action_keys, self._mask_key_sizes, self.num_envs)
            return response["obs"], [{}] * self.num_envs


if _HAVE_GODOT_RL and _HAVE_SB3_WRAPPER:
    class MaskableStableBaselinesGodotEnv(StableBaselinesGodotEnv):
        """StableBaselinesGodotEnv built over MaskAwareGodotEnv, exposing the sb3-contrib
        masking seam: `env_method("action_masks")` -> one uniform-length flat mask per env
        slot, env-major (matching `step()`'s obs concatenation order), and
        `has_attr("action_masks")` -> True (#385)."""

        def __init__(self, env_path=None, n_parallel=1, seed=0, **kwargs):
            # Replicates the upstream __init__ with MaskAwareGodotEnv substituted for GodotEnv
            # (upstream hardcodes the class; there is no factory hook).
            if env_path is None and n_parallel > 1:
                raise ValueError(
                    "You must provide the path to a exported game executable if n_parallel > 1")

            from godot_rl.core.godot_env import GodotEnv

            port = kwargs.pop("port", GodotEnv.DEFAULT_PORT)
            self.envs = [
                MaskAwareGodotEnv(
                    env_path=env_path,
                    convert_action_space=True,
                    port=port + p,
                    seed=seed + p,
                    **kwargs,
                )
                for p in range(n_parallel)
            ]
            self.n_parallel = n_parallel
            self._check_valid_action_space()
            self.results = None

        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            if method_name == "action_masks":
                # Env-major, per-agent within each GodotEnv — the exact order step() extends
                # all_obs, so mask row i lines up with obs/reward row i.
                return [m for e in self.envs for m in e.last_action_masks]
            return super().env_method()  # base raises NotImplementedError

        def has_attr(self, attr_name):
            if attr_name == "action_masks":
                return True
            return super().has_attr(attr_name)

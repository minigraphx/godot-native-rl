"""godot_rl action-masking helpers (#385).

Pure, import-free at module scope so the flattening logic stays unit-testable without torch/SB3/
godot_rl. The env wrappers that consume this (added in a later task) lazy-import godot_rl inside.
"""


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

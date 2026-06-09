# FlyBy example — attribution

This example is ported from the **FlyBy** environment in
[`edbeeching/godot_rl_agents_examples`](https://github.com/edbeeching/godot_rl_agents_examples)
(`examples/FlyBy`), which is licensed **MIT**, © 2022 Edward Beeching.

Vendored assets (under that repo's MIT license):
- `cartoon_plane/` — the cartoon plane glTF model + materials + texture.
- `sky.hdr` — the HDR environment map (upstream `alps_field_2k.hdr`).

The environment scripts (`fly_by_game.gd`, `fly_by_agent.gd`) and scenes were
re-implemented against the `godot_native_rl` framework (NcnnSync / NcnnAIController3D);
only the visual assets above are vendored verbatim.

MIT License text: see https://github.com/edbeeching/godot_rl_agents_examples/blob/main/LICENSE

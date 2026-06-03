# RaycastSensor Multi-Class Detection (`class_sensor`) — Design

**Date:** 2026-06-03
**Backlog item:** 41 — *`RaycastSensor3D` multi-class detection mode* (roadmap Track A; upstream-plugin parity).
Scope extended in brainstorming to **2D as well** (shared pure helper makes it nearly free; see §7).
**Status:** design approved, pre-implementation

## Goal

Today each ray of `RaycastSensor2D`/`RaycastSensor3D` emits a single "closeness" float — *how near*
the nearest hit is, with no notion of *what* was hit. Upstream `godot_rl_agents` `RaycastSensor3D`
has a `class_sensor` mode that also encodes the *class* of the hit object (via its collision layer).

Add an opt-in `class_sensor` mode to both raycast sensors that, per ray, emits a **one-hot/multi-hot
class segment** (one slot per listed collision layer) plus an optional catch-all `other` slot and an
optional distance (closeness) slot. When `class_sensor` is off, behavior is **byte-for-byte
unchanged**.

## Key decisions (locked in brainstorming)

1. **One-hot class slots + a separate distance slot (not a fused/closeness-weighted one-hot).**
   Keeping the categorical fact ("what did I hit") and the continuous fact ("how far") in separate
   dedicated inputs is the cleanest RL signal, and it disambiguates a *miss* (all zeros) from a
   *far hit on a known class* (class slot set, closeness ≈ 0) — something the distance-only sensor
   can't express. Matches the backlog wording ("per-class one-hot segments, keeping the distance
   encoding as an optional additional slot") and upstream `class_sensor`, which keeps distance
   alongside the class encoding.

2. **Catch-all `other` slot (default on), and multi-hot across listed classes.** A hit on a layer
   not in `detection_classes` lights an explicit `other` slot, so "unknown obstacle" is a learnable
   category distinct from a miss even at closeness ≈ 0. A collider on several listed layers lights
   **every** matching slot (multi-hot) — the honest representation of Godot's bitmask collision
   layers, rather than silently dropping secondary classes. Both the `other` slot and the distance
   slot are behind flags, so a user chasing exact upstream parity (no catch-all) can turn `other`
   off.

3. **`detection_classes` are 1-based collision-layer numbers.** An `Array[int]` like `[2, 3, 5]`
   means layers 2/3/5 (Godot's layer numbering). A hit matches class *i* when the collider's
   `collision_layer` has that bit set: `(hit_layer & (1 << (L - 1))) != 0`. This is the literal
   backlog reading ("Array[int] of collision layers to distinguish") and the most intuitive editor
   experience.

4. **All encoding logic is pure, in shared `raycast_math.gd`.** A single dimension-agnostic
   `encode_ray_class(...)` (it only needs distance + hit layer) serves both 2D and 3D, stays fully
   headless-unit-testable with no physics, and keeps the node wrappers thin.

5. **The physics-cast test seam is extended additively.** The existing `_cast` returns only a
   `float` distance and `set_cast_fn_for_test` injects a `(origin, dir) -> float`. Class mode needs
   the hit collider's layer too, so we add a parallel class-mode cast (`{distance, layer}`) and a
   `set_class_cast_fn_for_test` seam. The existing float seam and the non-class path are untouched —
   no breakage to current tests.

## Public API (both sensors)

New exports added to **`RaycastSensor2D`** and **`RaycastSensor3D`** (identical names/semantics):

```gdscript
@export var class_sensor: bool = false           # off => current distance-only behavior, unchanged
@export var detection_classes: Array[int] = []   # 1-based collision-layer numbers to distinguish
@export var include_other: bool = true           # catch-all slot for hits on unlisted layers
@export var include_distance: bool = true        # appended closeness slot
```

When `class_sensor == false`, the existing distance-per-ray code path runs verbatim — no behavioral
change, no obs-layout change.

## Encoding — pure helper in `raycast_math.gd`

```gdscript
# Per-ray class/distance segment. Stateless, no physics. miss => hit_distance < 0.
# Returns, in order: [ class_0 .. class_{n-1}, (other), (closeness) ]
static func encode_ray_class(
        hit_distance: float, hit_layer: int, ray_length: float,
        detection_classes: Array, include_other: bool, include_distance: bool) -> Array
```

Semantics:
- **Class slots** (one per `detection_classes` entry, in order): `1.0` if the ray hit **and**
  `(hit_layer & (1 << (L - 1))) != 0`, else `0.0`. Multiple may be set (multi-hot).
- **`other` slot** (only when `include_other`): `1.0` if the ray hit but **no** listed class
  matched, else `0.0`.
- **`closeness` slot** (only when `include_distance`): `RaycastMath.closeness(hit_distance,
  ray_length)` (miss → `0.0`).
- **Miss** (`hit_distance < 0`): all slots `0.0`.

Per-ray segment length = `detection_classes.size() + (include_other ? 1 : 0) + (include_distance ? 1
: 0)`.

## Cast seam change

Each sensor keeps its existing `_cast(origin, dir) -> float` and `set_cast_fn_for_test` (used by the
non-class path and all current tests). Class mode adds:

- `_cast_class(origin, dir) -> Dictionary` returning `{ "distance": float, "layer": int }`. The
  real query reads `result.collider.collision_layer`; a miss returns `{ distance = -1.0, layer = 0 }`.
- `set_class_cast_fn_for_test(fn)` where `fn` is a `(origin, dir) -> Dictionary` — injected in
  class-mode unit tests so the full observation path is exercised headlessly.

In `get_observation()`, when `class_sensor` is true the sensor iterates ray directions, calls
`_cast_class`, and concatenates `RaycastMath.encode_ray_class(...)` per ray.

## `obs_size()`

```
if not class_sensor:
    n_rays                                  # unchanged (2D: n_rays; 3D: n_rays_width * n_rays_height)
else:
    n_rays * (detection_classes.size()
              + (include_other ? 1 : 0)
              + (include_distance ? 1 : 0))
```

**Degenerate guard:** `class_sensor` on with empty `detection_classes` and both `include_other` and
`include_distance` off → per-ray segment is empty; warn once (mirroring the existing
degenerate-ray-count warning) and return an empty observation.

## Testing (headless, pure-first)

**`raycast_math` unit tests for `encode_ray_class`:**
- miss → all zeros (every flag combination)
- single listed-class hit → that slot `1.0`, others `0.0`, distance = expected closeness
- collider on multiple listed layers → multi-hot (every matching slot `1.0`)
- hit on an unlisted layer → all class slots `0.0`, `other` = `1.0` (when on)
- `include_other` off / `include_distance` off / both off → correct shorter segments
- distance-slot value equals `RaycastMath.closeness(...)`

**`RaycastSensor2D` and `RaycastSensor3D` tests (via `set_class_cast_fn_for_test`):**
- `obs_size()` across flag combinations
- full `get_observation()` segment count and ordering for a mix of hits/misses/classes
- **regression:** `class_sensor = false` yields output byte-identical to the current sensor

All tests run under the existing dependency-free headless harness (`test/harness.gd`).

## Scope & docs

- **Both 2D and 3D** get the feature (3D is the backlog item; 2D added because the shared pure
  helper makes it a few extra lines and keeps the two sensors symmetric).
- `CameraSensor` and the other sensors are unaffected.
- Doc updates in the same change: `CLAUDE.md` (sensors blurb — note `class_sensor` mode),
  `docs/BACKLOG.md` (mark item 41 done; note 2D included), and `docs/DEVELOPMENT.md` if it
  documents per-sensor obs encodings.

## Non-goals (YAGNI)

- No closeness-weighted/fused one-hot variant (rejected in favor of separate slots).
- No first-match-priority one-hot mode (multi-hot is the chosen, honest default).
- No group-name or node-type based classing — collision-layer based only, per upstream and the
  backlog.
- No changes to the `NcnnControllerCore.collect_sensors()` discovery path: the class-mode sensor
  still exposes `get_observation() -> Array` + `obs_size() -> int`, so auto-discovery just works.

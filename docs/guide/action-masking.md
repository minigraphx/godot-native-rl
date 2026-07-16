# Discrete Action Masking

Some decisions are illegal in some states: a wall to the north, a card already played, a menu you
can't open right now. **Discrete action masking** lets an agent tell the policy which discrete
actions are invalid *this step*, so they are never selected. This is our parity with Unity
ML-Agents' `WriteDiscreteActionMask` — with the extra: masking runs at **deploy** through native
ncnn inference, not just during training.

## The agent contract

Implement one optional method on your agent:

```gdscript
func get_action_mask() -> Dictionary:
	# Keyed like your action space; each value is a 0/1 array over that head's actions.
	# 1 = valid, 0 = masked (invalid). Default {} = nothing masked (all actions allowed).
	return {"move": [1, 0, 1, 1, 1]}  # action index 1 ("up") is illegal this step
```

The keys and per-head lengths mirror `get_action_space()`. The method is **optional** — an agent
that doesn't implement it behaves exactly as before. Only discrete heads can be masked.

Unity contract, honored here: **at least one action per head must stay valid**. An all-masked head
is a bug — see the fallback below.

## The wire field (training)

`NcnnSync` adds an **additive `action_mask`** field to each step/reset message, emitted **only when
at least one training agent implements `get_action_mask()`** (cached at handshake). Scenes with no
masking send byte-identical messages to before — the field is fully backward-compatible, and
unknown fields are ignored by stock godot_rl tooling.

## Deploy behavior (the moat)

At inference the controller core reads `agent.get_action_mask()` and threads it into
`ActionDecode.decode_actions`, which calls `InferenceMath.apply_action_mask` **before** argmax /
sampling. Masked slots are set to `-6e4` — an fp16-safe sentinel (min fp16 ≈ −6.55e4): argmax can
never pick them and softmax gives them sampling probability exactly 0 (`exp` underflows).

This happens **natively in ncnn on the deploy target** — web/WASM, console, mobile, edge. No
ONNX/.NET runtime does per-step discrete masking without custom glue; here it is built into the
decode path.

**All-masked fallback:** if a mask has the wrong size or masks *every* action, `apply_action_mask`
`push_error`s and returns the logits **unmasked** (the caller degrades to ordinary argmax). Inference
never bricks — but fix the agent: Unity requires ≥1 valid action.

## Training with MaskablePPO

To train a policy that *learns* under the mask (rather than only masking at deploy), use
sb3-contrib's MaskablePPO:

```bash
MASKABLE=1 ./scripts/train_gridworld.sh
```

This swaps in `MaskableStableBaselinesGodotEnv` + MaskablePPO (`scripts/train_gridworld.py
--maskable`). sb3-contrib is an **opt-in dep** (the same package as RecurrentPPO —
`.venv-train/bin/pip install -r requirements-recurrent.txt`). The exported network head is **raw
logits**; the mask is re-applied game-side at deploy, so training and deployment agree.

The MaskablePPO train → ONNX → ncnn path is proven end-to-end by a guarded smoke in
`test/run_tests.sh` (skipped when sb3-contrib isn't installed).

## Worked example: GridWorld

`examples/gridworld` (`GridWorldAgent.get_action_mask()`) masks off-grid moves — you cannot step
into a wall. The deploy regression `test/integration/gridworld_masked_scene.tscn` runs the shipped
ncnn net and asserts it selects **0 masked actions over 233 decisions** while still reaching 43
goals — proving native masked inference is load-bearing on a real net.

## Shipped-net note

The existing `examples/gridworld/models/gridworld.ncnn.*` net is **kept as-is** — deploy masking is
decode-side and needs **no retrain** (GridWorld's off-grid moves already clamp to no-ops, so a
masked-trained net wouldn't measurably improve goal-reaching). Native masked *deploy* is proven on
that net by the 0-violations regression above; the MaskablePPO *training* path is proven by the
guarded smoke. (#385)

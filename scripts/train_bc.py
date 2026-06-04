"""Behavior cloning over expert demos -> TorchScript model + shape sidecar.

The model matches the deploy contract (logits for discrete, means for continuous), so
`export_to_ncnn.py models/<bc>.pt` consumes it unchanged. Run in .venv-train (torch via SB3).

Usage:
  .venv-train/bin/python scripts/train_bc.py --demos demos.json --out models/bc_policy.pt
  # legacy godot_rl files (no action_space metadata) need --action-type:
  .venv-train/bin/python scripts/train_bc.py --demos old.json --out m.pt --action-type discrete
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from load_expert_demos import load_demos, flatten_pairs  # noqa: E402


def resolve_branches(action_space, action_type, y_width):
    """Return [{"type", "size"}] action branches. Sizes: discrete = #classes (output logits),
    continuous = action dim. Legacy files (no action_space) use a single --action-type branch."""
    if action_space is not None:
        branches = []
        for spec in action_space.values():
            branches.append({"type": spec["action_type"], "size": int(spec["size"])})
        return branches
    if action_type is None:
        raise ValueError("legacy godot_rl demos have no action_space; pass --action-type")
    if action_type == "continuous":
        return [{"type": "continuous", "size": int(y_width)}]
    return [{"type": "discrete", "size": None}]  # size resolved from data in train()


def _build_model(obs_dim, out_dim, hidden):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


def _bc_loss(out, y, branches):
    import torch.nn.functional as F
    loss = None
    o_off = 0  # output-column offset
    y_off = 0  # target-column offset
    for b in branches:
        size = b["size"]
        if b["type"] == "discrete":
            term = F.cross_entropy(out[:, o_off:o_off + size], y[:, y_off].long())
            o_off += size
            y_off += 1
        else:
            term = F.mse_loss(out[:, o_off:o_off + size], y[:, y_off:y_off + size])
            o_off += size
            y_off += size
        loss = term if loss is None else loss + term
    return loss


def train(x, y, branches, epochs, lr, hidden, out_path):
    import torch
    # Resolve any data-derived discrete sizes (legacy single-branch path) WITHOUT
    # mutating the caller's dicts (immutable-update convention).
    branches = [
        {**b, "size": int(y[:, j].max()) + 1}
        if b["type"] == "discrete" and b["size"] is None else b
        for j, b in enumerate(branches)
    ]
    out_dim = sum(b["size"] for b in branches)
    obs_dim = x.shape[1]
    model = _build_model(obs_dim, out_dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)

    first_loss = None
    last_loss = None
    for _ in range(epochs):
        opt.zero_grad()
        loss = _bc_loss(model(xt), yt, branches)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if first_loss is None:
            first_loss = last_loss

    model.eval()
    scripted = torch.jit.trace(model, torch.zeros(1, obs_dim))
    scripted.save(out_path)
    Path(out_path + ".shape.json").write_text(json.dumps({"inputshape": f"[1,{obs_dim}]"}))
    return first_loss, last_loss


def main():
    ap = argparse.ArgumentParser(description="Behavior cloning over expert demos.")
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", default="models/bc_policy.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--action-type", choices=["discrete", "continuous"], default=None)
    args = ap.parse_args()

    ds = load_demos(args.demos)
    x, y = flatten_pairs(ds)
    branches = resolve_branches(ds.action_space, args.action_type, y_width=y.shape[1])
    first, last = train(x, y, branches, args.epochs, args.lr, args.hidden, args.out)
    print(f"BC done: loss {first:.4f} -> {last:.4f}; wrote {args.out} (+ .shape.json)")
    print(f"Next: .venv-train/bin/python scripts/export_to_ncnn.py {args.out}")


if __name__ == "__main__":
    main()

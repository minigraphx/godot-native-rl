#!/usr/bin/env python3
"""Self-play pool bookkeeping for the alternating-phase orchestrator (#29).

    selfplay_phase.py register-snapshot --pool-dir models/selfplay_pool/hider --name hider_gen1

Appends a member to the pool's ELO ledger (pool.json) at the current learner rating (league
convention: a frozen copy starts where the learner left off), creating the ledger if absent.
Stdlib only; the pure helper is unit-tested in test/python/test_selfplay_phase.py.
"""
import argparse
import json
import pathlib
import sys

DEFAULT_RATING = 1200.0


def register_snapshot(ledger: dict, name: str, rating=None) -> dict:
    """Pure: return a new ledger with `name` added at `rating` (default: learner_rating)."""
    members = dict(ledger.get("members", {}))
    if name in members:
        raise ValueError(f"snapshot '{name}' already registered")
    learner = float(ledger.get("learner_rating", DEFAULT_RATING))
    members[name] = {"rating": float(rating) if rating is not None else learner, "games": 0}
    return {"members": members, "learner_rating": learner}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="cmd", required=True)
    reg = sub.add_parser("register-snapshot")
    reg.add_argument("--pool-dir", required=True)
    reg.add_argument("--name", required=True)
    reg.add_argument("--rating", type=float, default=None)
    args = parser.parse_args(argv)

    pool_dir = pathlib.Path(args.pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = pool_dir / "pool.json"
    ledger = {"members": {}, "learner_rating": DEFAULT_RATING}
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
    ledger = register_snapshot(ledger, args.name, args.rating)
    ledger_path.write_text(json.dumps(ledger, indent=2))
    print(f"registered '{args.name}' at rating {ledger['members'][args.name]['rating']} in {ledger_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

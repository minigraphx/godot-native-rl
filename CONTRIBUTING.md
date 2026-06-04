# Contributing

Entry point for **repo contributors**. (Game developers: see
[README.md](README.md) → [docs/guide/](docs/guide/).)

## Build from source
[docs/dev/building.md](docs/dev/building.md) — godot-cpp, ncnn static lib, SCons, multi-arch.

## Architecture & internals
[docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md) — data flow, inference-backend boundary, the
algorithm-agnostic deploy contract.

## Gotchas
[docs/dev/gotchas.md](docs/dev/gotchas.md) — read before debugging headless/training/convert issues.

## Tests
Run `./test/run_tests.sh` — must be green before merge (headless GDScript unit tests + Python
protocol/helper tests + inference/golden regressions).

## Workflow & roadmap
Superpowers workflow (brainstorm → spec → plan → TDD) — see
[CLAUDE.md](CLAUDE.md) "Conventions". Open work: GitHub issues (`backlog` label) + `docs/BACKLOG.md`.

## Docs hygiene
Update README, CLAUDE.md, the relevant `docs/guide/` or `docs/dev/` page, and the gap analysis in
the **same** change. Stale paths/commands count as a bug.

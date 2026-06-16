# CI: Skip the Test Matrix on Docs-Only Changes — Design (#261)

**Status:** approved (2026-06-16)
**Issue:** #261 (CI: skip the full test matrix on docs-only changes)
**Scope:** `.github/workflows/ci.yml` trigger filter + a companion robustness fix to
`scripts/dev/pr_merge_when_green.sh`.

## Problem

A **docs-only** PR runs the **entire CI test matrix on both Godot versions** for no benefit. The
`build` job is already well-optimized (content-addressed binary cache #85/#99 + godot-cpp/ncnn
caches), so it's cheap on no-code PRs — but the `test` matrix still runs the full
`test/run_tests.sh` **twice** (Godot 4.5 + 4.6): Godot download, two Python venvs, every headless
integration scene, golden behavioral regressions, INT8 export+parity. None of that can be affected
by a Markdown change.

## Key fact that shapes the design

`main` is **NOT a protected branch** (verified 2026-06-16 via the REST branch-protection
endpoint → HTTP 404 "Branch not protected"). The issue's central worry — that `paths-ignore`
makes a *required* status check "expected but never reported" and wedges merges — **does not
apply**, because there are no required status checks today. This rules **in** the simplest option
(`paths-ignore`) and rules **out** the need for the `dorny/paths-filter` + `ci-gate` aggregation
machinery from the issue's Option B.

## Goal

Skip the `build` + `test` jobs when a push/PR changes **only** documentation, without breaking the
merge-on-green developer workflow.

## Approach

### 1. `paths-ignore` on the CI triggers

Add `paths-ignore` to both the `push` and `pull_request` triggers in `ci.yml`:

```yaml
on:
  push:
    branches: [main]
    paths-ignore:
      - '**.md'
      - 'docs/**'
  pull_request:
    branches: [main]
    paths-ignore:
      - '**.md'
      - 'docs/**'
  workflow_dispatch:
```

Semantics (GitHub Actions): a workflow run is skipped only when **every** changed file matches a
`paths-ignore` pattern. So:

- A PR touching only Markdown (`README.md`, `CLAUDE.md`, `docs/**/*.md`, specs, plans) → **skipped**.
- A PR touching only non-Markdown docs assets under `docs/` (diagrams, etc.) → **skipped**.
- A PR touching **any** code/build/config file (`src/**`, `**/*.gd`, `**/*.tscn`, `**/*.py`,
  `SConstruct`, `.github/**`, `requirements*.txt`, `project.godot`, …) → **runs the full matrix**,
  exactly as today. A mixed docs+code PR also runs (not all files match the ignore set).

Pattern choice: `'**.md'` (not `'**/*.md'`) so root-level Markdown like `README.md`/`CLAUDE.md`
matches too — `**` in GitHub filters matches zero or more path segments, and the no-slash form is
the documented "any `.md` anywhere" pattern. `'docs/**'` covers the docs tree wholesale.

`workflow_dispatch` keeps a manual escape hatch to run CI on any ref regardless of paths.

The two-engine (4.5 + 4.6) matrix is **unchanged** for real code changes — that genuine
correctness guarantee is preserved.

### 2. Companion fix: `pr_merge_when_green.sh` must not stall on a no-CI PR

When `paths-ignore` skips the workflow, **no check-runs are created** for the head commit. The
current helper's `ci_verdict()` returns `"pending"` for a commit with zero check-runs (it can't
tell "CI hasn't started yet" from "CI will never run"), and the main loop treats `pending` as
"keep waiting" until `TIMEOUT` (default 2400 s = 40 min), then exits 1. So merging a docs-only PR
with the helper would **stall 40 minutes and fail** — a regression in the primary merge workflow.

Fix: add an explicit opt-in flag — **`--no-ci-wait`** (alias `--docs`) — that skips the
check-runs wait entirely and merges as soon as GitHub reports the PR `mergeable`. Explicit and
unambiguous; no racy "empty checks ⇒ green" heuristic that could merge a code PR before its CI has
registered.

Behaviour with the flag:

- Skip the `INITIAL_WAIT` sleep and the entire `ci_verdict` check-runs polling loop.
- Still read `pr_snapshot()` and guard: bail if already `merged`, and refuse to merge if
  `mergeable == false` (conflicts).
- If `mergeable == null` (GitHub still computing), poll `pr_snapshot()` only (cheap REST), up to
  `TIMEOUT`, until it resolves to `true`/`false` — no check-runs calls.
- Then merge + branch-delete via the existing REST paths. `--dry-run`/`--keep-branch`/`--method`
  compose normally.

Usage: `scripts/dev/pr_merge_when_green.sh <PR> --no-ci-wait` for docs-only PRs.

## Tradeoff (accepted)

Docs-only PRs get **no** automated check. Acceptable on an unprotected `main`. The issue's optional
**Option D** (a fast Markdown link-lint to still catch broken links) is **deferred — YAGNI** for
now; it can be added later as a separate cheap job if broken doc links become a real problem.

If branch protection is ever added to `main`, this design must be revisited: `paths-ignore` on a
*required* check would wedge docs PRs, and the change-detection-gate + `ci-gate` aggregation
(issue Option B) would become the correct pattern. This is noted as a documented follow-up, not
built now.

## Testing / validation

- **Docs-only path** — open (or simulate) a PR that changes only a `.md` file and confirm the
  `build`/`test` jobs do not run (no CI checks appear on the PR).
- **Code path** — confirm a PR touching a `.gd`/`.py`/`src` file still runs the full two-engine
  matrix.
- **Helper flag** — `--no-ci-wait --dry-run` on a docs PR reports "would merge" immediately
  instead of waiting; without the flag, behaviour is unchanged (still waits for check-runs).
- These are workflow-level behaviours; there is no headless unit test. Validate on a real
  throwaway docs PR before relying on it.

## Files

- `.github/workflows/ci.yml` — add `paths-ignore` to `push` + `pull_request`.
- `scripts/dev/pr_merge_when_green.sh` — add `--no-ci-wait`/`--docs` flag + the no-CI merge path.
- `docs/dev/gotchas.md` — short note: docs-only PRs skip CI; use `--no-ci-wait` to merge them with
  the helper.
- `CLAUDE.md` — one line under the CI bullet noting the docs-only skip.

## Out of scope / YAGNI

- `dorny/paths-filter` + `ci-gate` aggregation (Option B) — only needed under branch protection.
- Finer per-language test splits (Option C) — GDScript/scenes/C++/Python are intertwined;
  over-filtering risks missing a real regression.
- Markdown link-lint (Option D) — deferred.

## Acceptance

- A docs-only push/PR no longer triggers the build + test matrix.
- A code push/PR still runs the full two-engine matrix unchanged.
- `pr_merge_when_green.sh --no-ci-wait` merges a docs PR without the 40-minute stall.

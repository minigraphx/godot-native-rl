# CI: Skip the Test Matrix on Docs-Only Changes — Implementation Plan (#261)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** A push/PR that changes only documentation no longer runs the two-engine test matrix, and
the merge-on-green helper still works on those check-less PRs.

**Architecture:** Add `paths-ignore` to the CI workflow's `push` + `pull_request` triggers so a
docs-only change is skipped. Add an explicit `--no-ci-wait` flag to `pr_merge_when_green.sh` so a
PR with zero check-runs merges as soon as it is `mergeable` instead of stalling 40 minutes.

**Tech Stack:** GitHub Actions YAML, Bash (`gh api`, REST-only).

**Spec:** `docs/superpowers/specs/2026-06-16-ci-docs-only-gate-design.md`

**Branch:** `feature/261-ci-docs-gate` (already created, spec committed).

**Key fact:** `main` is NOT a protected branch (verified 2026-06-16), so no required status check is
wedged by `paths-ignore`. This is what makes the simple approach safe.

---

## ⚠️ Local-edit note

The working tree carries local-only edits to `project.godot`, `export_presets.cfg`, and
`examples/fly_by/sky.hdr.import` (Godot-4.6 resave artifacts). This plan does NOT touch them — when
you `git add`, always name **exact paths**, never `git add -A` / `.` / `-u`.

---

## Task 1: Add `paths-ignore` to the CI triggers

**Files:**
- Modify: `.github/workflows/ci.yml:13-18` (the `on:` block)

- [ ] **Step 1: Edit the `on:` block**

Current (lines 13-18):

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
```

Replace with:

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

Notes baked into the choice (do not change without re-checking the spec):
- `'**.md'` (no slash) matches Markdown at any depth **including the repo root** (`README.md`,
  `CLAUDE.md`).
- `'docs/**'` covers the whole docs tree (specs, plans, diagrams).
- A run is skipped only when **every** changed file matches a pattern, so any `.gd`/`.tscn`/`.py`/
  `src/**`/`SConstruct`/`.github/**`/`requirements*`/`project.godot` change still runs the full
  matrix. `workflow_dispatch` stays as a manual run-anything escape hatch.

- [ ] **Step 2: Verify the workflow YAML still parses**

Run (uses the train venv's PyYAML; the authoritative check is GitHub accepting the workflow on
push):

```bash
.venv-train/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"
```

Expected: `yaml ok`. Then confirm both triggers carry the filter:

```bash
grep -c "paths-ignore" .github/workflows/ci.yml
```

Expected: `2`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: skip the test matrix on docs-only changes (#261)"
```

---

## Task 2: Add `--no-ci-wait` to `pr_merge_when_green.sh`

**Files:**
- Modify: `scripts/dev/pr_merge_when_green.sh` (usage text, arg parse, default, main loop)

Why: `paths-ignore` means a docs-only PR creates **no check-runs**; the helper's `ci_verdict()`
returns `pending` for a commit with zero check-runs, and the loop waits until `TIMEOUT` (40 min)
then fails. The flag skips the check-runs wait and merges once GitHub reports the PR `mergeable`.

- [ ] **Step 1: Extend the usage text**

In the header usage comment and the `usage:` echo (around lines 11 and 24), add the flag. Change
both occurrences of:

```
<PR_NUMBER> [--dry-run] [--keep-branch] [--method merge|squash|rebase]
```

to:

```
<PR_NUMBER> [--dry-run] [--keep-branch] [--no-ci-wait] [--method merge|squash|rebase]
```

- [ ] **Step 2: Add the default + parse the flag**

After `KEEP_BRANCH=0` (line 30) add:

```bash
NO_CI_WAIT=0
```

In the arg-parse `case` (lines 33-38), add a branch before the `*)` catch-all:

```bash
		--no-ci-wait|--docs) NO_CI_WAIT=1 ;;
```

- [ ] **Step 3: Branch the wait logic**

Replace the initial-wait line + the main `while` loop (current lines 94-133) with:

```bash
if [ "$NO_CI_WAIT" = "1" ]; then
	echo "--no-ci-wait: skipping CI checks (docs-only PR); merging once GitHub reports it mergeable."
else
	echo "waiting ${INITIAL_WAIT}s before first check…"
	sleep "$INITIAL_WAIT"
fi

deadline=$(( $(date +%s) + TIMEOUT ))
HEAD_REF=""
HEAD_REPO=""
while :; do
	rest_guard
	snap="$(pr_snapshot)"
	IFS='|' read -r merged mergeable mstate sha HEAD_REF HEAD_REPO <<< "$snap"

	if [ "$merged" = "true" ]; then
		echo "PR #$PR is already merged. Nothing to do."
		exit 0
	fi

	if [ "$NO_CI_WAIT" = "1" ]; then
		echo "[$(date +%H:%M:%S)] no-ci-wait mergeable=$mergeable state=$mstate"
		# true -> ready to merge; false -> conflicts (caught by the guard after the loop);
		# null/empty -> GitHub still computing mergeability, keep polling (cheap, no check-runs call).
		if [ "$mergeable" = "true" ] || [ "$mergeable" = "false" ]; then
			break
		fi
	else
		verdict="$(ci_verdict "$sha")"
		echo "[$(date +%H:%M:%S)] ci=$verdict mergeable=$mergeable state=$mstate"

		case "$verdict" in
			failed)
				echo "CI has a failing check — NOT merging:"
				print_checks "$sha"
				exit 1
				;;
			green)
				print_checks "$sha"
				break
				;;
			pending)
				: # keep waiting
				;;
		esac
	fi

	if [ "$(date +%s)" -ge "$deadline" ]; then
		echo "timeout: CI did not finish within ${TIMEOUT}s" >&2
		exit 1
	fi
	sleep "$POLL_INTERVAL"
done
```

The existing post-loop block (the `mergeable == "false"` conflict guard, the `--dry-run` report,
and the merge + branch-delete) is unchanged and serves both modes.

- [ ] **Step 4: Syntax-check the script**

```bash
bash -n scripts/dev/pr_merge_when_green.sh && echo "syntax ok"
```

Expected: `syntax ok`. If `shellcheck` is installed, also run
`shellcheck scripts/dev/pr_merge_when_green.sh` and confirm no new errors.

- [ ] **Step 5: Dry-run sanity (no merge)**

On any **open** PR you don't mind probing (e.g. this branch's own PR once opened — see Task 3),
confirm the flag short-circuits the wait:

```bash
scripts/dev/pr_merge_when_green.sh <SOME_OPEN_PR> --no-ci-wait --dry-run
```

Expected: it prints `--no-ci-wait: skipping CI checks …` and, once GitHub reports the PR mergeable,
`DRY RUN: PR #… is green and mergeable — would merge …` **without** the 40-minute wait. (Without the
flag, on a docs-only PR it would instead sit on `ci=pending`.)

- [ ] **Step 6: Commit**

```bash
git add scripts/dev/pr_merge_when_green.sh
git commit -m "feat(dev): --no-ci-wait flag so docs PRs don't stall the merge helper (#261)"
```

---

## Task 3: Docs, backlog, and live validation

**Files:**
- Modify: `docs/dev/gotchas.md` (extend the existing GraphQL/merge-helper section)
- Modify: `CLAUDE.md` (one line under the CI bullet)
- Modify: `docs/BACKLOG.md` (tick #261 if listed)

- [ ] **Step 1: Note the behaviour in `docs/dev/gotchas.md`**

Add a short paragraph near the `pr_merge_when_green.sh` / GraphQL section:

```markdown
- **Docs-only PRs skip CI.** `.github/workflows/ci.yml` has `paths-ignore: ['**.md', 'docs/**']`,
  so a change touching only docs creates no check-runs. `pr_merge_when_green.sh` would otherwise wait
  for checks that never appear (40-min timeout) — pass `--no-ci-wait` (alias `--docs`) to merge such
  a PR as soon as it is mergeable. Any code/build/workflow change still runs the full matrix.
```

- [ ] **Step 2: One line in `CLAUDE.md`**

Under the existing **CI** bullet (the `.github/workflows/ci.yml` description), append a sentence:

```
A docs-only push/PR (only `**.md` / `docs/**`) skips the build+test jobs (`paths-ignore`, #261);
merge such PRs with `scripts/dev/pr_merge_when_green.sh <PR> --no-ci-wait` since no checks run.
```

- [ ] **Step 3: Tick the backlog**

If #261 is listed in `docs/BACKLOG.md`, tick its checkbox.

- [ ] **Step 4: Commit docs**

```bash
git add docs/dev/gotchas.md CLAUDE.md docs/BACKLOG.md
git commit -m "docs: note docs-only CI skip + --no-ci-wait (#261)"
```

- [ ] **Step 5: Open the PR + live-validate both paths**

Push the branch and open a PR that `Closes #261`. This PR touches `.github/**` + `scripts/**` +
docs, so it is **not** docs-only — confirm the **full matrix runs** on it (proves the code path is
intact). After it merges, validate the skip with a throwaway docs-only PR:

```bash
git checkout main && git pull
git checkout -b chore/docs-skip-probe
printf '\n<!-- ci skip probe -->\n' >> README.md
git add README.md && git commit -m "docs: ci skip probe"
git push -u origin chore/docs-skip-probe
gh api repos/minigraphx/godot-native-rl/pulls -f title="docs: ci skip probe" -f head=chore/docs-skip-probe -f base=main >/dev/null
# Confirm NO build/test checks appear on the PR, then close it:
#   gh api repos/minigraphx/godot-native-rl/commits/$(git rev-parse HEAD)/check-runs --jq '.total_count'  # expect 0
```

Expected: the probe PR shows **0** check-runs (matrix skipped). Merge it with `--no-ci-wait` or just
close + delete the branch.

---

## Notes for the implementer

- `gh api` (REST) only — never `gh pr` / `gh issue` / `gh repo` (GraphQL, easy to exhaust). The
  helper already follows this; keep it that way.
- The two-engine (4.5 + 4.6) matrix must stay intact for real code changes — only the trigger
  filter changes, not the jobs.
- If branch protection is ever added to `main`, revisit: a `paths-ignore` on a *required* check
  would wedge docs PRs, and the change-detection-gate + `ci-gate` aggregation (issue Option B)
  becomes the correct pattern.

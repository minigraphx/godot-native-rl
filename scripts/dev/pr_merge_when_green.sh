#!/usr/bin/env bash
# pr_merge_when_green.sh — wait for a PR's CI to go green, then merge it, REST-ONLY.
#
# Why: `gh pr` / `gh issue` / `gh repo` subcommands use GitHub's GraphQL API, whose hourly quota is
# easy to exhaust with tight CI-watch loops (see docs/dev/gotchas.md). This helper touches ONLY the
# REST API via `gh api` (a separate, much-harder-to-exhaust 5000/hr bucket), guards on the free
# `rate_limit` endpoint, and polls at a sane cadence — turning a "merge on green" into ~2 REST calls
# per poll instead of dozens of GraphQL calls.
#
# Usage:
#   scripts/dev/pr_merge_when_green.sh <PR_NUMBER> [--dry-run] [--keep-branch] [--no-ci-wait] [--method merge|squash|rebase]
#
# Env overrides:
#   INITIAL_WAIT   seconds to wait before the first check (default 60; CI here takes ~18 min)
#   POLL_INTERVAL  seconds between checks (default 120)
#   TIMEOUT        max seconds to wait for CI overall (default 2400 = 40 min)
#   MERGE_METHOD   merge|squash|rebase (default merge; --method overrides)
#   --no-ci-wait (alias --docs): merge as soon as PR is mergeable, skipping check-runs wait
#     (for docs-only PRs that create no checks); INITIAL_WAIT is ignored when set; other env overrides still apply
#
# Exit codes: 0 merged (or dry-run reported green), 1 failure/timeout, 2 usage error.
set -euo pipefail

PR="${1:-}"
if [ -z "$PR" ]; then
	echo "usage: $0 <PR_NUMBER> [--dry-run] [--keep-branch] [--no-ci-wait] [--method merge|squash|rebase]" >&2
	exit 2
fi
shift || true

DRY_RUN=0
KEEP_BRANCH=0
NO_CI_WAIT=0
METHOD="${MERGE_METHOD:-merge}"
while [ $# -gt 0 ]; do
	case "$1" in
		--dry-run) DRY_RUN=1 ;;
		--keep-branch) KEEP_BRANCH=1 ;;
		--method) shift; METHOD="${1:?--method needs a value}" ;;
		--no-ci-wait|--docs) NO_CI_WAIT=1 ;;
		*) echo "unknown arg: $1" >&2; exit 2 ;;
	esac
	shift
done

INITIAL_WAIT="${INITIAL_WAIT:-60}"
POLL_INTERVAL="${POLL_INTERVAL:-120}"
TIMEOUT="${TIMEOUT:-2400}"

# Resolve owner/repo from the origin remote (git protocol, not the API — no GraphQL).
remote="$(git remote get-url origin 2>/dev/null || true)"
[ -n "$remote" ] || { echo "error: no 'origin' remote" >&2; exit 1; }
REPO="$(printf '%s' "$remote" | sed -E 's#^git@[^:]+:##; s#^https?://[^/]+/##; s#\.git$##')"
[ -n "$REPO" ] || { echo "error: could not parse owner/repo from '$remote'" >&2; exit 1; }
echo "repo=$REPO pr=#$PR method=$METHOD dry_run=$DRY_RUN"

# Free REST call (does not count against quota); pause until reset if the core bucket is nearly spent.
rest_guard() {
	local rem reset now wait
	rem="$(gh api rate_limit --jq '.resources.core.remaining' 2>/dev/null || echo 100)"
	if [ "${rem:-100}" -lt 10 ]; then
		reset="$(gh api rate_limit --jq '.resources.core.reset' 2>/dev/null || echo 0)"
		now="$(date +%s)"
		wait=$(( reset - now + 5 ))
		[ "$wait" -lt 5 ] && wait=5
		echo "REST budget low ($rem left); waiting ${wait}s for reset…"
		sleep "$wait"
	fi
}

# All PR fields we need in one REST call: merged | mergeable | mergeable_state | head_sha | head_ref | head_repo
pr_snapshot() {
	gh api "repos/$REPO/pulls/$PR" \
		--jq '"\(.merged)|\(.mergeable)|\(.mergeable_state)|\(.head.sha)|\(.head.ref)|\(.head.repo.full_name)"'
}

# Reduce a commit's check-runs to one verdict: green | pending | failed. One REST call.
# - failed: any run concluded failure/cancelled/timed_out/action_required/stale
# - green:  >=1 run AND all completed AND none failed
# - pending: no runs yet, or some not yet completed
ci_verdict() {
	local sha="$1"
	gh api "repos/$REPO/commits/$sha/check-runs" --jq '
		.check_runs as $r
		| if ($r | length) == 0 then "pending"
		  elif any($r[]; .conclusion == "failure" or .conclusion == "cancelled"
		                 or .conclusion == "timed_out" or .conclusion == "action_required"
		                 or .conclusion == "stale") then "failed"
		  elif all($r[]; .status == "completed") then "green"
		  else "pending" end'
}

print_checks() {
	local sha="$1"
	gh api "repos/$REPO/commits/$sha/check-runs" --jq '.check_runs[] | "  \(.name): \(.status)/\(.conclusion)"' 2>/dev/null || true
}

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

# CI is green. Confirm GitHub considers the PR mergeable (false = conflicts; null = still computing).
if [ "$mergeable" = "false" ]; then
	echo "CI green but GitHub reports the PR not mergeable (state=$mstate) — resolve conflicts first." >&2
	exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
	echo "DRY RUN: PR #$PR is green and mergeable — would merge (--method $METHOD) and delete branch '$HEAD_REF'."
	exit 0
fi

echo "merging PR #$PR (method=$METHOD)…"
rest_guard
gh api -X PUT "repos/$REPO/pulls/$PR/merge" -f merge_method="$METHOD" \
	--jq '"merged=\(.merged) sha=\(.sha[0:9]) msg=\(.message)"'

# Delete the source branch — only for a same-repo branch (never a fork), unless --keep-branch.
if [ "$KEEP_BRANCH" = "0" ] && [ "$HEAD_REPO" = "$REPO" ] && [ -n "$HEAD_REF" ]; then
	if gh api -X DELETE "repos/$REPO/git/refs/heads/$HEAD_REF" 2>/dev/null; then
		echo "deleted branch '$HEAD_REF'"
	else
		echo "note: could not delete branch '$HEAD_REF' (may already be gone)"
	fi
fi

echo "done."

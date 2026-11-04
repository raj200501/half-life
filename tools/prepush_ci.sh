#!/usr/bin/env bash
# Run `make ci` on a fresh clone of HEAD before pushing (DECISIONS.md D-022).
# Only committed work is checked, which is exactly what a push sends.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
sha=$(git -C "$repo" rev-parse HEAD)
tmp=$(mktemp -d -t halflife-prepush-XXXXXX)
trap 'rm -rf "$tmp"' EXIT
git clone --quiet --no-hardlinks "$repo" "$tmp/clone"
git -C "$tmp/clone" checkout --quiet --detach "$sha"
cd "$tmp/clone"
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
if make setup >"$tmp/setup.log" 2>&1 && make ci >"$tmp/ci.log" 2>&1; then
    echo "PREPUSH-CI: PASS $sha ($(grep -Eo '[0-9]+ passed' "$tmp/ci.log" | tail -1))"
else
    for log in "$tmp/setup.log" "$tmp/ci.log"; do
        if [ -f "$log" ]; then echo "== $log"; tail -40 "$log"; fi
    done
    echo "PREPUSH-CI: FAIL $sha"
    exit 1
fi

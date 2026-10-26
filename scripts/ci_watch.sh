#!/usr/bin/env bash
# Wait for the GitHub Actions CI run of the pushed HEAD and exit non-zero if any job fails.
# A run appears a few seconds after `git push`, so first poll until it is registered.
#
#   git push && make ci-watch
set -euo pipefail

workflow="ci.yml"
sha=$(git rev-parse HEAD)

if [[ "$sha" != "$(git rev-parse '@{u}' 2>/dev/null || true)" ]]; then
  echo "HEAD ${sha:0:7} is not pushed to its upstream branch; push first." >&2
  exit 1
fi

run_id=""
for _ in $(seq 1 30); do  # up to ~60 s for the run to be registered
  run_id=$(gh run list --commit "$sha" --workflow "$workflow" --limit 1 \
    --json databaseId --jq '.[0].databaseId // empty')
  [[ -n "$run_id" ]] && break
  sleep 2
done
if [[ -z "$run_id" ]]; then
  echo "No $workflow run found for ${sha:0:7} after 60 s." >&2
  exit 1
fi

echo "Watching $workflow run $run_id for ${sha:0:7} ..."
status=0
gh run watch "$run_id" --exit-status --interval 10 >/dev/null || status=$?
gh run view "$run_id" --json conclusion,url,jobs \
  --jq '"\(.conclusion): \(.url)", (.jobs[] | "  \(.name): \(.conclusion)")'
if [[ $status -ne 0 ]]; then
  echo "CI failed; inspect with: gh run view $run_id --log-failed" >&2
fi
exit $status

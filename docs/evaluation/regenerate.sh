#!/usr/bin/env bash
# Single scripted protocol that regenerates every evidence file under docs/evaluation/.
# Run from the repo root: bash docs/evaluation/regenerate.sh
#
# Every number cited in the paper's evaluation section must be traceable to a file this script
# produced, with the run count that produced it stated in the file itself -- no number should
# ever be typed from memory of terminal output that wasn't captured.
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT=docs/evaluation
RUNS=10

# Windows' CPU-name query pads its output with trailing spaces; trim it once here so every
# generated file that embeds it passes `git diff --check` instead of failing on trailing
# whitespace that has nothing to do with the content.
cpu_name() {
    powershell -NoProfile -Command '(Get-CimInstance Win32_Processor).Name' 2>/dev/null \
        | sed -E 's/[[:space:]]+$//' || echo unknown
}

echo "=== recording provenance ==="
# This file records the SOURCE revision this run evaluated. It never records the hash of the
# commit that will eventually contain this file, because that commit does not exist while this
# file is being written -- a generated file cannot contain the hash of the commit that contains
# that same generated file without a self-reference. See the archival sequence in
# docs/evaluation/README.md (or the paper's Workbench/Reproducibility section) for how the two
# commits that matter here -- the reviewed implementation, and the immutable evidence produced
# from a clean checkout of it -- relate, and where the SECOND commit's hash actually gets
# recorded (a git tag on that commit, or the paper's release/Zenodo metadata) -- never here.
{
    echo "# Evidence provenance -- regenerated every run, never hand-edited"
    echo "generated_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "invocation: bash docs/evaluation/regenerate.sh $*" | sed -E 's/[[:space:]]+$//'
    echo
    echo "## Source revision evaluated"
    echo "commit: $(git rev-parse HEAD)"
    echo "branch: $(git rev-parse --abbrev-ref HEAD)"
    DIRTY_FILES="$(git status --porcelain)"
    if [ -z "$DIRTY_FILES" ]; then
        echo "working_tree: clean -- this evidence corresponds exactly to the commit above."
        echo "  This is the condition required before the files in this directory may be"
        echo "  committed as the evidence-artifact commit -- see the archival sequence."
    else
        echo "working_tree: DIRTY -- this evidence does NOT correspond to the commit above by"
        echo "  itself, and MUST NOT be committed as final evidence in this state. It reflects"
        echo "  that commit plus the uncommitted changes listed below. Commit the reviewed"
        echo "  diff first, then re-run this script from that clean checkout before treating"
        echo "  any of these files as citable."
        echo "  uncommitted changes (git status --porcelain):"
        echo "$DIRTY_FILES" | sed 's/^/    /'
    fi
    echo
    echo "## Platform"
    echo "uname: $(uname -a 2>/dev/null || echo unknown)"
    echo "rustc: $(rustc --version)"
    echo "cargo: $(cargo --version)"
    echo "cpu: $(cpu_name)"
} > "$OUT/PROVENANCE.txt"
cat "$OUT/PROVENANCE.txt"

echo "=== full release test suite (must be green before any evidence is trusted) ==="
set +e
cargo test --workspace --release --locked > "$OUT/workspace_tests.txt" 2>&1
TEST_EXIT=$?
set -e
echo "cargo test exit code: ${TEST_EXIT}" >> "$OUT/workspace_tests.txt"
if [ "$TEST_EXIT" -ne 0 ]; then
    echo "!!! workspace tests failed (exit ${TEST_EXIT}) -- see $OUT/workspace_tests.txt -- stopping, no evidence regenerated" >&2
    exit 1
fi
echo "workspace tests: green"

echo "=== building release binaries ==="
cargo build --release \
    --example reachability_intersections -p mm-search \
    --example assess_trace_latency -p mm-search \
    --example method_breakdown -p mm-verifier \
    --example verify_step_latency -p mm-verifier
cargo build --release --example seeded_eval -p mm-solver

BIN_SEARCH=target/release/examples
BIN_VERIFIER=target/release/examples
BIN_SOLVER=target/release/examples

echo "=== reachability_intersections: ${RUNS} runs ==="
{
    echo "# ${RUNS} runs of reachability_intersections, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for i in $(seq 1 "$RUNS"); do
        echo "--- run $i/$RUNS ---"
        "./$BIN_SEARCH/reachability_intersections.exe"
    done
} > "$OUT/reachability_intersections_runs.txt"

echo "=== method_breakdown: ${RUNS} runs ==="
{
    echo "# ${RUNS} runs of method_breakdown, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for i in $(seq 1 "$RUNS"); do
        echo "--- run $i/$RUNS ---"
        "./$BIN_VERIFIER/method_breakdown.exe"
    done
} > "$OUT/method_breakdown_runs.txt"

echo "=== verify_step_latency: 1 run (protocol is internally repeated) ==="
"./$BIN_VERIFIER/verify_step_latency.exe" > "$OUT/verify_step_bench.txt"
{
    echo ""
    echo "cpu: $(cpu_name)"
} >> "$OUT/verify_step_bench.txt"

echo "=== assess_trace_latency: 1 run (protocol is internally repeated) ==="
"./$BIN_SEARCH/assess_trace_latency.exe" > "$OUT/assess_trace_bench.txt"
{
    echo ""
    echo "cpu: $(cpu_name)"
} >> "$OUT/assess_trace_bench.txt"

echo "=== seeded_eval: pinned seed 42 ==="
set +e
"./$BIN_SOLVER/seeded_eval.exe" 42 > "$OUT/seeded_eval_manifest_seed42.txt" 2>&1
SEED_EXIT=$?
set -e
echo "exit code: $SEED_EXIT" >> "$OUT/seeded_eval_manifest_seed42.txt"
if [ "$SEED_EXIT" -ne 0 ]; then
    echo "seeded evaluation failed" >&2
    exit "$SEED_EXIT"
fi

echo "=== determinism check: rerun seed 42, diff modulo timing ==="
"./$BIN_SOLVER/seeded_eval.exe" 42 > "$OUT/.seeded_eval_rerun.tmp" 2>&1
if diff <(sed -E 's/[0-9]+\.[0-9]+m?s//g' "$OUT/seeded_eval_manifest_seed42.txt" | sed '/^exit code:/d') \
        <(sed -E 's/[0-9]+\.[0-9]+m?s//g' "$OUT/.seeded_eval_rerun.tmp") > "$OUT/seeded_eval_determinism_check.txt"; then
    echo "IDENTICAL modulo timing -- deterministic on seed 42" >> "$OUT/seeded_eval_determinism_check.txt"
    rm -f "$OUT/.seeded_eval_rerun.tmp"
else
    echo "DIFFERED -- NOT deterministic, investigate before citing seed 42" >> "$OUT/seeded_eval_determinism_check.txt"
    rm -f "$OUT/.seeded_eval_rerun.tmp"
    echo "!!! seeded_eval is not deterministic on seed 42 -- see $OUT/seeded_eval_determinism_check.txt -- stopping" >&2
    exit 1
fi

echo "=== deriving summary from the files just written ==="
python3 "$OUT/summarize.py" > "$OUT/SUMMARY.txt"
cat "$OUT/SUMMARY.txt"

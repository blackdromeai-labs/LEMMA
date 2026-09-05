#!/usr/bin/env python3
"""Derive a machine-checked summary from the evidence files regenerate.sh just wrote.

Every number below is parsed out of a captured file in this directory -- nothing here is
typed from memory of terminal output. Run counts (denominators) are counted from the actual
number of "--- run" blocks found, not assumed.
"""
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def read(name: str) -> str:
    return (HERE / name).read_text()


def split_runs(text: str) -> list[str]:
    """Split a multi-run capture file into one chunk per run block."""
    blocks = re.split(r"--- run \d+/\d+ ---\n", text)
    return [b for b in blocks[1:]]  # blocks[0] is the header before the first run


def parse_field(block: str, label: str) -> int:
    m = re.search(re.escape(label) + r"\s*:\s*(-?\d+)", block)
    if not m:
        raise ValueError(f"field {label!r} not found in block")
    return int(m.group(1))


def summarize_intersections() -> str:
    text = read("reachability_intersections_runs.txt")
    runs = split_runs(text)
    n = len(runs)

    accepted = [parse_field(b, "accepted") for b in runs]
    acc_not_reach = [parse_field(b, "accepted but NOT reachable") for b in runs]
    reach_not_acc = [parse_field(b, "reachable but NOT accepted") for b in runs]
    exec_not_reach = [parse_field(b, "executable but NOT reachable") for b in runs]
    reachable = [parse_field(b, "reachable                 ") for b in runs]
    executable = [parse_field(b, "executable                ") for b in runs]
    registered = [parse_field(b, "registered                ") for b in runs]

    lines = []
    lines.append(f"reachability_intersections.rs -- {n} runs (source: reachability_intersections_runs.txt)")
    lines.append(f"  registered: {sorted(set(registered))}  (constant across all {n} runs: {len(set(registered)) == 1})")
    lines.append(f"  executable: {sorted(set(executable))}  (constant across all {n} runs: {len(set(executable)) == 1})")
    lines.append(f"  reachable:  {sorted(set(reachable))}  (constant across all {n} runs: {len(set(reachable)) == 1})")
    acc_counts = Counter(accepted)
    lines.append(f"  accepted, exact distribution over {n} runs: " +
                 ", ".join(f"{v} ({c}/{n})" for v, c in sorted(acc_counts.items())))
    lines.append(f"  accepted but NOT reachable, exact distribution over {n} runs: " +
                 ", ".join(f"{v} ({c}/{n})" for v, c in sorted(Counter(acc_not_reach).items())))
    if set(acc_not_reach) == {0}:
        lines.append(
            f"    -> 0 in all {n}/{n} runs: on this 228-witness corpus, every generated "
            f"application of the guardrail-hidden rules was rejected in all {n} runs. This "
            f"does not establish unconditional rejection for every possible expression -- only "
            f"that no counterexample appeared in this corpus across this many runs."
        )
    reach_not_acc_min, reach_not_acc_max = min(reach_not_acc), max(reach_not_acc)
    lines.append(f"  reachable minus accepted, per-run range over {n} runs: {reach_not_acc_min}-{reach_not_acc_max} "
                 f"(= reachable[{reachable[0]}] - accepted, tracks the accepted distribution above exactly, not an independent figure)")
    lines.append(f"  executable but NOT reachable, exact distribution over {n} runs: " +
                 ", ".join(f"{v} ({c}/{n})" for v, c in sorted(Counter(exec_not_reach).items())))
    return "\n".join(lines)


def summarize_method_breakdown() -> str:
    text = read("method_breakdown_runs.txt")
    runs = split_runs(text)
    n = len(runs)

    def field(block, section_after, label):
        # find the label occurrence that comes after `section_after` marker text
        idx = block.index(section_after)
        rest = block[idx:]
        return parse_field(rest, label)

    applications_sym = [field(b, "APPLICATIONS", "symbolic equivalence") for b in runs]
    applications_num = [field(b, "APPLICATIONS", "numeric sampling") for b in runs]
    applications_rep = [field(b, "APPLICATIONS", "rule replay only") for b in runs]

    pairs_sym = [field(b, "PAIRS", "symbolic equivalence") for b in runs]
    pairs_num = [field(b, "PAIRS", "numeric sampling") for b in runs]
    pairs_rep = [field(b, "PAIRS", "rule replay only") for b in runs]

    rules_sym = [field(b, "not disjoint", "symbolic equivalence") for b in runs]
    rules_num = [field(b, "not disjoint", "numeric sampling") for b in runs]
    rules_rep = [field(b, "not disjoint", "rule replay only") for b in runs]
    rules_union = [parse_field(b, "union (any method)") for b in runs]

    def dist(vals):
        c = Counter(vals)
        return ", ".join(f"{v} ({cnt}/{n})" for v, cnt in sorted(c.items()))

    lines = []
    lines.append(f"method_breakdown.rs -- {n} runs (source: method_breakdown_runs.txt)")
    lines.append(f"  applications (symbolic/numeric/replay-only), exact distributions over {n} runs:")
    lines.append(f"    symbolic : {dist(applications_sym)}")
    lines.append(f"    numeric  : {dist(applications_num)}")
    lines.append(f"    replay   : {dist(applications_rep)}")
    lines.append(f"  distinct (rule, witness) pairs, exact distributions over {n} runs:")
    lines.append(f"    symbolic : {dist(pairs_sym)}")
    lines.append(f"    numeric  : {dist(pairs_num)}")
    lines.append(f"    replay   : {dist(pairs_rep)}")
    lines.append(f"  distinct rules (not disjoint across methods), exact distributions over {n} runs:")
    lines.append(f"    symbolic : {dist(rules_sym)}")
    lines.append(f"    numeric  : {dist(rules_num)}")
    lines.append(f"    replay   : {dist(rules_rep)}")
    lines.append(f"    union    : {dist(rules_union)}")
    lines.append("  NOTE: this file alone does not establish accepted-minus-reachable=0 -- only")
    lines.append("  reachability_intersections.rs measures reachability, so only its own runs support that claim.")
    return "\n".join(lines)


def summarize_seeded_eval() -> str:
    text = read("seeded_eval_manifest_seed42.txt")
    seed_m = re.search(r"seed = (\d+)", text)
    overall_m = re.search(
        r"Overall: (\d+)/(\d+) passed strictly \(([\d.]+)%\); (\d+) more independently confirmed "
        r"via polynomial coefficients.*?; (\d+) more sampling-equivalent only.*?; (\d+) genuinely "
        r"wrong or unreplayed",
        text,
    )
    exit_m = re.search(r"exit code: (\d+)", text)
    determinism = read("seeded_eval_determinism_check.txt").strip().splitlines()[-1]

    lines = []
    lines.append("seeded_eval.rs -- pinned seed (source: seeded_eval_manifest_seed42.txt)")
    if seed_m:
        lines.append(f"  seed: {seed_m.group(1)}")
    if overall_m:
        strict, total, pct, poly, sample, wrong = overall_m.groups()
        lines.append(f"  strict PASS:            {strict}/{total} ({pct}%)")
        lines.append(f"  POLY (independently confirmed via coefficient extraction): {poly}")
        lines.append(f"  SAMPLE (sampling-equivalent only, NOT independently confirmed): {sample}")
        lines.append(f"  genuinely wrong or unreplayed: {wrong}")
    else:
        lines.append("  WARNING: could not parse the Overall line from the manifest")
    if exit_m:
        lines.append(
            f"  process exit code: {exit_m.group(1)} "
            f"(0 = no genuinely-wrong/unreplayed cases AND no SAMPLE-tier-only cases)"
        )
    lines.append(f"  determinism check (rerun same seed, diff modulo timing): {determinism}")
    return "\n".join(lines)


STAT_KEYS = ("min", "mean", "p50", "p90", "p99", "max")


def parse_stats_block(block: str) -> dict[str, str]:
    stats = {}
    for stat in STAT_KEYS:
        m = re.search(rf"{stat}:\s*([\d.]+)", block)
        if m:
            stats[stat] = m.group(1)
    return stats


def summarize_benches() -> str:
    lines = []

    # verify_step: one unlabeled block -- the whole corpus is timed as a single population.
    text = read("verify_step_bench.txt")
    lines.append("verify_step benchmark (source: verify_step_bench.txt):")
    for key in ("rustc", "profile", "cpu", "warm-up passes", "sample passes"):
        m = re.search(re.escape(key) + r"\s*:?\s*(.+)", text)
        if m:
            lines.append(f"  {key}: {m.group(1).strip()}")
    for stat, value in parse_stats_block(text).items():
        lines.append(f"  {stat}: {value} ns/call")
    lines.append("")

    # assess_trace: split into its labeled trace-length groups so the per-group stats are not
    # printed as two unlabeled sets of min/mean/p50/... -- that ambiguity is exactly what made
    # the previous SUMMARY.txt misleading despite being "machine-derived".
    text = read("assess_trace_bench.txt")
    lines.append("assess_trace benchmark (source: assess_trace_bench.txt):")
    for key in ("rustc", "profile", "cpu", "warm-up passes", "sample passes", "inner reps/pass"):
        m = re.search(re.escape(key) + r"\s*:?\s*(.+)", text)
        if m:
            lines.append(f"  {key}: {m.group(1).strip()}")

    group_pattern = re.compile(
        r"--- trace length = (\d+) step\(s\), (\d+) reference trace\(s\) in group ---\n(.*?)(?=\n---|\Z)",
        re.DOTALL,
    )
    groups = group_pattern.findall(text)
    if not groups:
        lines.append("  WARNING: no labeled trace-length groups found in assess_trace_bench.txt")
    for length, trace_count, block in groups:
        lines.append(f"  -- length={length} step(s), {trace_count} reference trace(s) --")
        for stat, value in parse_stats_block(block).items():
            lines.append(f"    {stat}: {value} ns/call")
    lines.append("  NOTE: only 1-step and 2-step groups exist in this reference set -- do not")
    lines.append("  infer a general scaling trend from two points.")
    lines.append("")
    return "\n".join(lines).rstrip()


def summarize_workspace_tests() -> str:
    text = read("workspace_tests.txt")
    exit_m = re.search(r"cargo test exit code:\s*(\d+)", text)

    result_lines = re.findall(
        r"test result:\s*(ok|FAILED)\.\s*(\d+) passed;\s*(\d+) failed;\s*(\d+) ignored", text
    )
    total_passed = sum(int(p) for _, p, _, _ in result_lines)
    total_failed = sum(int(f) for _, _, f, _ in result_lines)
    total_ignored = sum(int(i) for _, _, _, i in result_lines)
    suites_with_failures = sum(1 for status, _, f, _ in result_lines if status == "FAILED" or int(f) > 0)

    lines = []
    lines.append("full workspace release test suite (source: workspace_tests.txt):")
    lines.append(f"  test binaries reporting a result line: {len(result_lines)}")
    lines.append(f"  total: {total_passed} passed, {total_failed} failed, {total_ignored} ignored")
    lines.append(f"  suites with at least one failure: {suites_with_failures}")
    if exit_m:
        code = exit_m.group(1)
        lines.append(f"  cargo test exit code: {code} ({'GREEN' if code == '0' else 'RED -- evidence below is NOT trustworthy'})")
    else:
        lines.append("  WARNING: could not find 'cargo test exit code' line -- run did not finish normally")
    return "\n".join(lines)


def main() -> int:
    sections = [
        read("PROVENANCE.txt").rstrip(),
        "",
        summarize_workspace_tests(),
        "",
        summarize_intersections(),
        "",
        summarize_method_breakdown(),
        "",
        summarize_seeded_eval(),
        "",
        summarize_benches(),
    ]
    print("\n".join(sections))
    return 0


if __name__ == "__main__":
    sys.exit(main())

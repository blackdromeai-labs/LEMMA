//! Benchmark protocol for the paper: `verify_step` latency, isolated from everything that is
//! not `verify_step` -- specifically `Rule::can_apply`, `Rule::apply` (pattern matching,
//! allocation, `RuleApplication` construction), and iteration overhead. Those all happened in
//! the timed region of an earlier version of this benchmark, which meant the reported number
//! was never actually `verify_step`'s own cost.
//!
//! Also isolated from `assess_trace` (that one lives in
//! `mm-search/examples/assess_trace_latency.rs`, since the two answer different questions --
//! checking one proposed step vs replaying an already-computed trace).
//!
//! Protocol: every `(before, after, rule)` triple the witness corpus produces is computed once,
//! up front, and stored -- this is the untimed precompute phase. Then `WARMUP_PASSES` untimed
//! passes over the stored triples let branch predictors/caches settle, followed by
//! `SAMPLE_PASSES` timed passes that call *only* `verify_step` on each stored triple, with
//! `std::hint::black_box` around both the inputs and the return value so the optimizer cannot
//! see that the result is unused and elide the call.
//!
//! A single timed pass is still only a few hundred calls in well under a millisecond, close to
//! `Instant`'s practical per-call resolution, so each *pass* is one sample and percentiles are
//! taken over many passes -- the same reason microbenchmark harnesses batch instead of timing
//! one fast call directly.
//!
//! Run with `--release`; a debug build's numbers are not representative and are not what the
//! paper cites.
//!
//! Usage: cargo run --release --example verify_step_latency -p mm-verifier

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use mm_core::Expr;
use mm_rules::{corpus, rule::Rule, standard_rules, RuleContext, WitnessSymbols};
use mm_verifier::Verifier;

const WARMUP_PASSES: usize = 20;
const SAMPLE_PASSES: usize = 200;

fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Every `(before, after, rule)` triple the corpus produces, computed once and stored. Building
/// this is deliberately outside every timed region.
fn precompute_triples<'a>(
    rules: &'a mm_rules::rule::RuleSet,
    witnesses: &[Expr],
    ctx: &RuleContext,
) -> Vec<(Expr, Expr, &'a Rule)> {
    let mut triples = Vec::new();
    for rule in rules.all() {
        for witness in witnesses {
            if !rule.can_apply(witness, ctx) {
                continue;
            }
            for app in rule.apply(witness, ctx) {
                if app.result == *witness {
                    continue;
                }
                triples.push((witness.clone(), app.result, rule));
            }
        }
    }
    triples
}

/// One timed pass: call `verify_step` on every precomputed triple, nothing else.
fn one_pass(
    verifier: &Verifier,
    triples: &[(Expr, Expr, &Rule)],
    ctx: &RuleContext,
) -> std::time::Duration {
    let start = Instant::now();
    for (before, after, rule) in triples {
        let result = verifier.verify_step(
            black_box(before),
            black_box(after),
            black_box(*rule),
            black_box(ctx),
        );
        black_box(result);
    }
    start.elapsed()
}

fn percentile(sorted_ns_per_call: &[f64], p: f64) -> f64 {
    let idx = ((sorted_ns_per_call.len() - 1) as f64 * p).round() as usize;
    sorted_ns_per_call[idx]
}

fn main() {
    let rules = standard_rules();
    let verifier = Verifier::new();
    let symbols = WitnessSymbols::new();
    let witnesses = corpus(&symbols);
    let ctx = RuleContext::default();

    let triples = precompute_triples(&rules, &witnesses, &ctx);

    println!("=== verify_step benchmark ===");
    println!("rustc:          {}", rustc_version());
    println!(
        "profile:        {}",
        if cfg!(debug_assertions) {
            "debug (NOT representative -- rerun with --release)"
        } else {
            "release"
        }
    );
    println!("target arch:    {}", std::env::consts::ARCH);
    println!("target os:      {}", std::env::consts::OS);
    println!(
        "logical cores:  {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
    println!("witness corpus: {} expressions", witnesses.len());
    println!("registered rules: {}", rules.len());
    println!(
        "precomputed (before, after, rule) triples: {}",
        triples.len()
    );
    println!("warm-up passes: {WARMUP_PASSES}");
    println!("sample passes:  {SAMPLE_PASSES}");

    // Warm-up: run and discard. Not timed.
    for _ in 0..WARMUP_PASSES {
        let _ = one_pass(&verifier, &triples, &ctx);
    }

    // Timed samples: one mean-ns-per-call figure per pass, calling only verify_step.
    let mut ns_per_call_samples: Vec<f64> = Vec::with_capacity(SAMPLE_PASSES);
    for _ in 0..SAMPLE_PASSES {
        let elapsed = one_pass(&verifier, &triples, &ctx);
        ns_per_call_samples.push(elapsed.as_nanos() as f64 / triples.len() as f64);
    }

    ns_per_call_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ns_per_call_samples.len();
    let sum: f64 = ns_per_call_samples.iter().sum();
    let mean = sum / n as f64;
    let min = ns_per_call_samples[0];
    let max = ns_per_call_samples[n - 1];
    let p50 = percentile(&ns_per_call_samples, 0.50);
    let p90 = percentile(&ns_per_call_samples, 0.90);
    let p99 = percentile(&ns_per_call_samples, 0.99);

    println!("\ncalls per pass: {}", triples.len());
    println!("\nper-call verify_step latency across {n} sample passes (ns/call):");
    println!("  min:    {min:.1}");
    println!("  mean:   {mean:.1}");
    println!("  p50:    {p50:.1}");
    println!("  p90:    {p90:.1}");
    println!("  p99:    {p99:.1}");
    println!("  max:    {max:.1}");
}

//! Benchmark protocol for the paper: `assess_trace` latency, isolated from search cost and
//! from `verify_step` (that one lives in `mm-verifier/examples/verify_step_latency.rs`).
//! `assess_trace` replays an already-computed trace -- it does no rule search and no
//! `Verifier::verify_step` calls of its own, only expression-equality checks between
//! consecutive steps and an evidence-status combination -- so it answers a different cost
//! question than either of those, and this benchmark's numbers are not directly comparable to
//! `verify_step`'s until both are read as "cost of the specific operation each name refers to,"
//! not as a general system-overhead ratio.
//!
//! A fixed set of representative solved traces is generated once with `NeuralMCTS::simplify`
//! (untimed -- search cost must not leak into this measurement), then grouped by trace length
//! in steps, since `assess_trace`'s cost is expected to scale with steps and reporting one
//! pooled number across different lengths would hide that. Each sample for a given length
//! group replays every trace in that group `INNER_REPS` times (not once) so the sample's total
//! duration is comfortably above `Instant`'s practical resolution floor rather than being one
//! or a handful of near-instantaneous calls; `std::hint::black_box` wraps both the inputs and
//! the return value so the optimizer cannot see the result is discarded and elide the call.
//!
//! Protocol: `WARMUP_PASSES` untimed passes per length group, then `SAMPLE_PASSES` timed passes
//! per group. Run with `--release`; a debug build's numbers are not representative and are not
//! what the paper cites.
//!
//! Usage: cargo run --release --example assess_trace_latency -p mm-search

use std::collections::BTreeMap;
use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use mm_core::{Expr, SymbolTable};
use mm_rules::standard_rules;
use mm_search::{assess_trace, MCTSConfig, NeuralMCTS, Step};
use mm_verifier::Verifier;

const WARMUP_PASSES: usize = 10;
const SAMPLE_PASSES: usize = 200;
const INNER_REPS: usize = 2000;

fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// A representative set of (problem, result, steps) traces, computed once by an actual solver
/// run so the recorded evidence is real, not hand-constructed. Solver cost happens here, before
/// timing starts.
fn generate_reference_traces() -> Vec<(Expr, Expr, Vec<Step>)> {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let y = symbols.intern("y");

    let problems = vec![
        // Identity noise: short trace.
        Expr::Add(
            Box::new(Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::Var(y)))),
            Box::new(Expr::int(0)),
        ),
        // Power product: short trace.
        Expr::Mul(
            Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(3)))),
            Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(2)))),
        ),
        // Pythagorean identity.
        Expr::Add(
            Box::new(Expr::Pow(
                Box::new(Expr::Sin(Box::new(Expr::Var(x)))),
                Box::new(Expr::int(2)),
            )),
            Box::new(Expr::Pow(
                Box::new(Expr::Cos(Box::new(Expr::Var(x)))),
                Box::new(Expr::int(2)),
            )),
        ),
        // Derivative power rule.
        Expr::Derivative {
            expr: Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(5)))),
            var: x,
        },
        // Binomial square expansion: exercises heuristic/partial evidence.
        Expr::Pow(
            Box::new(Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(3)))),
            Box::new(Expr::int(2)),
        ),
        // Linear equation solving: equation-aware rewriting.
        Expr::Equation {
            lhs: Box::new(Expr::Add(
                Box::new(Expr::Mul(Box::new(Expr::int(4)), Box::new(Expr::Var(x)))),
                Box::new(Expr::int(-3)),
            )),
            rhs: Box::new(Expr::int(9)),
        },
        // GCD.
        Expr::GCD(Box::new(Expr::int(462)), Box::new(Expr::int(315))),
    ];

    let rules = standard_rules();
    let verifier = Verifier::new();
    let config = MCTSConfig {
        simulations: 150,
        ..Default::default()
    };
    let mcts = NeuralMCTS::with_config(rules, verifier, config);

    problems
        .into_iter()
        .map(|p| {
            let solution = mcts.simplify(p);
            (solution.problem, solution.result, solution.steps)
        })
        .collect()
}

/// One timed pass over one length group: replay every trace in the group `INNER_REPS` times.
fn one_pass(group: &[(Expr, Expr, Vec<Step>)]) -> std::time::Duration {
    let start = Instant::now();
    for _ in 0..INNER_REPS {
        for (problem, result, steps) in group {
            let status = assess_trace(black_box(problem), black_box(result), black_box(steps));
            black_box(status);
        }
    }
    start.elapsed()
}

fn percentile(sorted_ns_per_call: &[f64], p: f64) -> f64 {
    let idx = ((sorted_ns_per_call.len() - 1) as f64 * p).round() as usize;
    sorted_ns_per_call[idx]
}

fn main() {
    let traces = generate_reference_traces();

    println!("=== assess_trace benchmark ===");
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
    println!("reference traces: {}", traces.len());
    println!("warm-up passes:  {WARMUP_PASSES} (per length group)");
    println!("sample passes:   {SAMPLE_PASSES} (per length group)");
    println!("inner reps/pass: {INNER_REPS}");

    // Group traces by step length -- assess_trace's cost is expected to scale with steps, so
    // pooling lengths together would hide that.
    let mut by_length: BTreeMap<usize, Vec<(Expr, Expr, Vec<Step>)>> = BTreeMap::new();
    for trace in traces {
        by_length.entry(trace.2.len()).or_default().push(trace);
    }

    for (length, group) in &by_length {
        for _ in 0..WARMUP_PASSES {
            let _ = one_pass(group);
        }

        let calls_per_pass = (INNER_REPS * group.len()) as f64;
        let mut ns_per_call_samples: Vec<f64> = Vec::with_capacity(SAMPLE_PASSES);
        for _ in 0..SAMPLE_PASSES {
            let elapsed = one_pass(group);
            ns_per_call_samples.push(elapsed.as_nanos() as f64 / calls_per_pass);
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

        println!(
            "\n--- trace length = {length} step(s), {} reference trace(s) in group ---",
            group.len()
        );
        println!("per-call assess_trace latency across {n} sample passes (ns/call):");
        println!("  min:    {min:.1}");
        println!("  mean:   {mean:.1}");
        println!("  p50:    {p50:.1}");
        println!("  p90:    {p90:.1}");
        println!("  p99:    {p99:.1}");
        println!("  max:    {max:.1}");
    }
}

//! Verify that a trained policy actually loads and carries learned signal.
//!
//! This is the gate before any trained-vs-uniform experiment. Three things have to hold, and
//! each has failed at least once during this project:
//!
//! 1. The weights load against the current rule set at all (the manifest records the action
//!    vocabulary digest, and loading refuses a mismatch rather than silently proceeding).
//! 2. Provenance flips `Untrained -> Loaded`. `NeuralMCTS` asks the policy whether it is
//!    trained before consulting it; a model that loads but still reports `Untrained` would be
//!    silently ignored and every prior would stay uniform.
//! 3. The priors are actually non-uniform. A network can load cleanly and still emit a flat
//!    distribution, which is uniform search wearing a trained model's name.
//!
//! Usage: check_policy [path-to-weights]

use mm_brain::{ModelProvenance, PolicyNetwork};
use mm_core::{Expr, SymbolTable};
use mm_rules::ActionVocabulary;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "experiments/models/policy.safetensors".to_string());

    let vocabulary = ActionVocabulary::standard();
    println!("weights   : {path}");
    println!(
        "vocabulary: {} actions, digest {:#018x}",
        vocabulary.len(),
        vocabulary.digest()
    );

    let policy = match PolicyNetwork::load(
        std::path::Path::new(&path),
        vocabulary.clone(),
        candle_core::Device::Cpu,
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL: could not load weights: {e}");
            std::process::exit(1);
        }
    };

    match policy.provenance() {
        ModelProvenance::Loaded {
            vocabulary_digest,
            weights_digest,
            ..
        } => {
            println!(
                "provenance: Loaded (vocab {vocabulary_digest:#018x}, weights {weights_digest:#018x})"
            );
        }
        ModelProvenance::Untrained { .. } => {
            eprintln!("FAIL: provenance is Untrained after loading; search would ignore it");
            std::process::exit(2);
        }
    }
    if !policy.provenance().is_trained() {
        eprintln!("FAIL: provenance.is_trained() is false; NeuralMCTS would use uniform priors");
        std::process::exit(3);
    }
    println!("is_trained: true");

    // Probe a few expressions the training set covers, and check the distribution is not flat.
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let probes: Vec<(&str, Expr)> = vec![
        (
            "x * 0",
            Expr::Mul(Box::new(Expr::Var(x)), Box::new(Expr::int(0))),
        ),
        (
            "x + 0",
            Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(0))),
        ),
        (
            "d/dx(x^3)",
            Expr::Derivative {
                expr: Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(3)))),
                var: x,
            },
        ),
    ];

    let uniform = 1.0f32 / (vocabulary.len() as f32 + 1.0);
    println!("\nuniform prior would be {uniform:.6} for every action\n");

    let mut all_flat = true;
    for (label, expr) in &probes {
        match policy.rule_priors(expr) {
            Ok(priors) => {
                let max = priors.iter().cloned().fold(f32::MIN, f32::max);
                let sum: f32 = priors.iter().sum();
                let best = priors
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let name = vocabulary
                    .key_at(best)
                    .map(|k| k.to_string())
                    .unwrap_or_else(|_| "<terminal>".to_string());
                println!(
                    "{label:<12} max prior {max:.4} on {name}  (sum {sum:.3}, {:.0}x uniform)",
                    max / uniform
                );
                if max < uniform * 5.0 {
                    println!("             ^ WARNING: close to uniform, little learned signal");
                } else {
                    all_flat = false;
                }
            }
            Err(e) => {
                eprintln!("FAIL: rule_priors failed for {label}: {e}");
                std::process::exit(4);
            }
        }
    }

    if all_flat {
        eprintln!("\nFAIL: every probe produced a near-uniform distribution");
        std::process::exit(5);
    }
    println!("\nOK: weights load, provenance is Loaded, priors carry learned signal.");
}

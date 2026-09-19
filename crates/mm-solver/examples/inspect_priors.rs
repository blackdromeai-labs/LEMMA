//! Diagnose why the trained policy hurts search: print the raw and renormalized priors the
//! policy actually assigns to every verified, legal child at the root of a specific problem.
//!
//! Usage: inspect_priors "<expr>" [model.safetensors]

use candle_core::Device;
use mm_brain::PolicyNetwork;
use mm_core::parse::Parser;
use mm_core::SymbolTable;
use mm_rules::{standard_rules, ActionVocabulary, RuleContext};
use mm_verifier::Verifier;

fn main() {
    let input = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "1 * (1 * (3 * x + 6 * x) + 0)".to_string());
    let model_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "experiments/models/policy.safetensors".to_string());

    let mut symbols = SymbolTable::new();
    let mut parser = Parser::new(&mut symbols);
    let expr = parser.parse(&input).expect("input must parse");

    let vocabulary = ActionVocabulary::standard();
    let policy = PolicyNetwork::load(
        std::path::Path::new(&model_path),
        vocabulary.clone(),
        Device::Cpu,
    )
    .expect("model must load");

    println!("input: {input}");
    println!("provenance: {:?}\n", policy.provenance());

    let raw_priors = policy
        .rule_priors(&expr)
        .expect("forward pass must succeed");
    let raw_max = raw_priors.iter().cloned().fold(f32::MIN, f32::max);
    let raw_sum: f32 = raw_priors.iter().sum();
    println!(
        "raw policy output over ALL {} actions: max={raw_max:.6} sum={raw_sum:.6}\n",
        raw_priors.len()
    );

    // Reproduce exactly what NeuralMCTS::expand does: filter by guardrail, can_apply, apply,
    // dedupe self-loops, verify, then collect (rule_name, raw_prior).
    let rules = standard_rules();
    let ctx = RuleContext::default();
    let profile = mm_boink::analyze(&expr);
    let valid_rules = mm_boink::filter_rules(rules.all(), &profile);
    let verifier = Verifier::new();

    let mut legal: Vec<(String, f32, String)> = Vec::new();
    for rule in valid_rules {
        if !rule.can_apply(&expr, &ctx) {
            continue;
        }
        for app in rule.apply(&expr, &ctx) {
            if app.result == expr {
                continue;
            }
            let verify = verifier.verify_step(&expr, &app.result, rule, &ctx);
            if !verify.is_valid() {
                continue;
            }
            let prior = vocabulary
                .prior_for_rule(&raw_priors, rule.id)
                .unwrap_or(f32::NAN);
            legal.push((
                rule.name.to_string(),
                prior,
                mm_core::format_expr(&app.result, &symbols),
            ));
        }
    }

    let total: f32 = legal.iter().map(|(_, p, _)| p).sum();
    legal.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!(
        "{} legal (verified) children at the root; raw-prior sum over just these = {total:.8}\n",
        legal.len()
    );
    println!(
        "{:<28} {:>12} {:>12} {:>10}  -> result",
        "rule", "raw_prior", "renorm", "share"
    );
    for (name, raw, result) in &legal {
        let renorm = if total > 1e-12 { raw / total } else { f32::NAN };
        println!(
            "{name:<28} {raw:>12.8} {renorm:>12.6} {:>9.1}%  -> {result}",
            renorm * 100.0
        );
    }
}

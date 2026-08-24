//! Which rules the verifier will actually let the search use.
//!
//! `mm-rules`' census (`rule_census.rs`) answers "does this rule change an expression".
//! That is necessary but not sufficient: [`crate::Verifier::verify_step`] is called on every
//! expansion, and a transformation it rejects never becomes a search node. A rule can
//! therefore be perfectly functional and still be dead.
//!
//! Two kinds of rejection matter and this test tells them apart:
//!
//! - **Rejected everywhere** - every transformation the rule produced was refused. Either the
//!   rule is unsound, or it produces something the verifier cannot check. Either way the
//!   search cannot use it, and the census's "transforms" count overstates what is available.
//! - **Accepted by replay only** - accepted, but only because the expression contains a
//!   derivative or integral that the evaluator cannot sample. These contribute steps that can
//!   never raise a result above `Heuristic`.

use std::collections::BTreeMap;

use mm_rules::rule::RuleKey;
use mm_rules::{corpus, standard_rules, RuleContext, WitnessSymbols};
use mm_verifier::{VerificationMethod, Verifier};

struct Acceptance {
    key: RuleKey,
    /// Transformations the rule produced across the corpus.
    produced: usize,
    /// Transformations the verifier accepted.
    accepted: usize,
    /// Whether any acceptance rested on more than re-running the rule.
    accepted_independently: bool,
}

/// Pinned. Update deliberately, with the run that produced the number.
const EXPECTED_TRANSFORMING: usize = 146;
const EXPECTED_ACCEPTED_SOMEWHERE: usize = 117;
const EXPECTED_REJECTED_EVERYWHERE: usize = 29;

fn acceptance() -> Vec<Acceptance> {
    let rules = standard_rules();
    let verifier = Verifier::new();
    let symbols = WitnessSymbols::new();
    let witnesses = corpus(&symbols);
    let ctx = RuleContext::default();

    let mut out = Vec::new();

    for (rule, key) in rules.all().iter().zip(rules.keys()) {
        let mut produced = 0;
        let mut accepted = 0;
        let mut accepted_independently = false;

        for witness in &witnesses {
            if !rule.can_apply(witness, &ctx) {
                continue;
            }
            for app in rule.apply(witness, &ctx) {
                if app.result == *witness {
                    continue;
                }
                produced += 1;

                let result = verifier.verify_step(witness, &app.result, rule, &ctx);
                if result.is_valid() {
                    accepted += 1;
                    if result
                        .method()
                        .is_some_and(|m| m != VerificationMethod::RuleReplayOnly)
                    {
                        accepted_independently = true;
                    }
                }
            }
        }

        if produced > 0 {
            out.push(Acceptance {
                key: *key,
                produced,
                accepted,
                accepted_independently,
            });
        }
    }

    out
}

#[test]
fn verifier_acceptance_matches_the_pinned_counts() {
    let reports = acceptance();
    let accepted_somewhere = reports.iter().filter(|r| r.accepted > 0).count();
    let rejected_everywhere = reports.iter().filter(|r| r.accepted == 0).count();

    println!("\nverifier acceptance over the witness corpus:");
    println!("  rules that transform something : {}", reports.len());
    println!("  accepted on at least one witness: {accepted_somewhere}");
    println!("  rejected on every witness       : {rejected_everywhere}");
    println!(
        "  accepted by rule replay only    : {}",
        reports
            .iter()
            .filter(|r| r.accepted > 0 && !r.accepted_independently)
            .count()
    );

    let mut per_module: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for report in &reports {
        let slot = per_module.entry(report.key.module).or_insert((0, 0));
        slot.0 += 1;
        if report.accepted > 0 {
            slot.1 += 1;
        }
    }
    println!("\n  module          transforms  accepted");
    for (module, (t, a)) in &per_module {
        println!("  {module:<16}{t:>10}{a:>10}");
    }

    assert_eq!(reports.len(), EXPECTED_TRANSFORMING);
    assert_eq!(accepted_somewhere, EXPECTED_ACCEPTED_SOMEWHERE);
    assert_eq!(rejected_everywhere, EXPECTED_REJECTED_EVERYWHERE);
}

#[test]
fn rules_the_verifier_always_rejects_are_listed() {
    // These transform an expression but the search can never use them. Not a failure: some
    // are genuinely unsound and being rejected is correct. The list is what someone auditing
    // rule soundness should start from.
    let mut rejected: Vec<String> = acceptance()
        .iter()
        .filter(|r| r.accepted == 0)
        .map(|r| format!("{} ({} transformations, all refused)", r.key, r.produced))
        .collect();
    rejected.sort();

    println!(
        "\n{} rules transform an expression but the verifier refuses every result:",
        rejected.len()
    );
    for name in &rejected {
        println!("  {name}");
    }

    assert_eq!(rejected.len(), EXPECTED_REJECTED_EVERYWHERE);
}

#[test]
fn acceptance_counts_are_internally_consistent() {
    for report in acceptance() {
        assert!(
            report.produced > 0,
            "{} recorded with no output",
            report.key
        );
        assert!(
            report.accepted <= report.produced,
            "{} accepted more than it produced",
            report.key
        );
        if report.accepted == 0 {
            assert!(!report.accepted_independently);
        }
    }
}

//! Measured census of the rule corpus.
//!
//! Module comments in this crate state things like "28 working, 56 need implementation" and
//! the crate documentation totals them into "162 working rules". Those numbers were written
//! by hand and nothing checked them. This test replaces them with a measurement: run every
//! registered rule against every expression in [`mm_rules::witness`] and record what happened.
//!
//! Three verdicts:
//!
//! - **Transforms** - some witness makes the rule applicable and it returns something other
//!   than its input. This is the only verdict that means the rule does anything.
//! - **NoOp** - applicable to at least one witness, but every application returns the input
//!   unchanged (or returns nothing at all). These are stubs: they are offered to the search
//!   as legal moves and waste expansions without changing the expression.
//! - **NoWitness** - no expression in the corpus makes `is_applicable` true. This is *not*
//!   the same as "unreachable"; it means this corpus does not reach it. Widening the corpus
//!   can only move rules out of this bucket, never into it.
//!
//! A fourth axis is recorded separately: whether the BOINK guardrail would ever let the rule
//! through for a witness it transforms. A rule that transforms but is always filtered out is
//! invisible to search regardless of what it can do.
//!
//! The counts are pinned. A change in either direction fails, so improving a stub is a test
//! failure that tells you to update the number, and silently breaking a working rule is a
//! test failure too.

use std::collections::{BTreeMap, HashSet};

use mm_rules::rule::RuleKey;
use mm_rules::{analyze, corpus, filter_rules, standard_rules, RuleContext, WitnessSymbols};

/// What running a rule against the whole corpus showed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Verdict {
    /// Applicable somewhere, and changes the expression.
    Transforms,
    /// Applicable somewhere, but never changes the expression.
    NoOp,
    /// Never applicable to any witness in this corpus.
    NoWitness,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Verdict::Transforms => "transforms",
            Verdict::NoOp => "no-op",
            Verdict::NoWitness => "no-witness",
        }
    }
}

struct RuleReport {
    key: RuleKey,
    verdict: Verdict,
    /// Number of corpus expressions the rule declared itself applicable to.
    applicable_to: usize,
    /// Number of corpus expressions where it produced a different expression.
    transforms_on: usize,
    /// Whether the guardrail would offer this rule for at least one witness it transforms.
    reachable_through_guardrail: bool,
}

/// Pinned counts. Update deliberately, with the run that produced them.
///
/// Last moved by turning 7 rules honest: `pigeonhole`, `pigeonhole_principle`,
/// `binomial_alternating_sum`, `bertrand_postulate`, `sum_of_divisors`, `number_of_divisors`
/// and `primitive_root_find` all matched a broad expression shape and replaced it with a
/// value unrelated to the match (a discrete fact, or a different number-theoretic property of
/// the input, standing in for a rewrite of it). They now leave the expression unchanged
/// instead, which moves them from "transforms" to "no-op".
const EXPECTED_TOTAL: usize = 572;
const EXPECTED_TRANSFORMS: usize = 138;
const EXPECTED_NO_OP: usize = 244;
const EXPECTED_NO_WITNESS: usize = 190;

fn census() -> Vec<RuleReport> {
    let rules = standard_rules();
    let symbols = WitnessSymbols::new();
    let witnesses = corpus(&symbols);
    let ctx = RuleContext::default();

    // Which rules the guardrail would offer for each witness.
    let visible: Vec<HashSet<u32>> = witnesses
        .iter()
        .map(|w| {
            let profile = analyze(w);
            filter_rules(rules.all(), &profile)
                .into_iter()
                .map(|r| r.id.0)
                .collect()
        })
        .collect();

    rules
        .all()
        .iter()
        .zip(rules.keys())
        .map(|(rule, key)| {
            let mut applicable_to = 0;
            let mut transforms_on = 0;
            let mut reachable_through_guardrail = false;

            for (index, witness) in witnesses.iter().enumerate() {
                if !rule.can_apply(witness, &ctx) {
                    continue;
                }
                applicable_to += 1;

                let changed = rule
                    .apply(witness, &ctx)
                    .iter()
                    .any(|app| app.result != *witness);
                if changed {
                    transforms_on += 1;
                    if visible[index].contains(&rule.id.0) {
                        reachable_through_guardrail = true;
                    }
                }
            }

            let verdict = if applicable_to == 0 {
                Verdict::NoWitness
            } else if transforms_on == 0 {
                Verdict::NoOp
            } else {
                Verdict::Transforms
            };

            RuleReport {
                key: *key,
                verdict,
                applicable_to,
                transforms_on,
                reachable_through_guardrail,
            }
        })
        .collect()
}

fn counts(reports: &[RuleReport]) -> BTreeMap<Verdict, usize> {
    let mut out = BTreeMap::new();
    for report in reports {
        *out.entry(report.verdict).or_insert(0) += 1;
    }
    out
}

#[test]
fn rule_census_matches_the_pinned_counts() {
    let reports = census();
    let by_verdict = counts(&reports);

    let get = |v: Verdict| by_verdict.get(&v).copied().unwrap_or(0);

    println!("\nrule census over {} witnesses:", {
        let symbols = WitnessSymbols::new();
        corpus(&symbols).len()
    });
    println!("  total registered : {}", reports.len());
    println!("  transforms       : {}", get(Verdict::Transforms));
    println!("  no-op (stub)     : {}", get(Verdict::NoOp));
    println!("  no witness here  : {}", get(Verdict::NoWitness));

    // Per module, so the hand-written module comments can be checked against reality.
    let mut per_module: BTreeMap<&str, [usize; 3]> = BTreeMap::new();
    for report in &reports {
        let slot = per_module.entry(report.key.module).or_insert([0, 0, 0]);
        match report.verdict {
            Verdict::Transforms => slot[0] += 1,
            Verdict::NoOp => slot[1] += 1,
            Verdict::NoWitness => slot[2] += 1,
        }
    }
    println!("\n  module          transforms  no-op  no-witness");
    for (module, [t, n, w]) in &per_module {
        println!("  {module:<16}{t:>10}{n:>7}{w:>12}");
    }

    assert_eq!(reports.len(), EXPECTED_TOTAL, "registry size changed");
    assert_eq!(
        get(Verdict::Transforms),
        EXPECTED_TRANSFORMS,
        "the number of rules that actually transform something changed; update the pin \
         together with the run that produced it"
    );
    assert_eq!(get(Verdict::NoOp), EXPECTED_NO_OP, "stub count changed");
    assert_eq!(
        get(Verdict::NoWitness),
        EXPECTED_NO_WITNESS,
        "the number of rules this corpus does not reach changed"
    );
}

#[test]
fn transforming_rules_are_reachable_through_the_guardrail() {
    // A rule that does something but is always filtered out before the search sees it is
    // dead in practice. If this list grows, the guardrail is hiding working rules.
    let reports = census();
    let hidden: Vec<String> = reports
        .iter()
        .filter(|r| r.verdict == Verdict::Transforms && !r.reachable_through_guardrail)
        .map(|r| r.key.to_string())
        .collect();

    assert!(
        hidden.is_empty(),
        "{} rules transform an expression but the guardrail never offers them:\n  {}",
        hidden.len(),
        hidden.join("\n  ")
    );
}

#[test]
fn no_op_rules_are_listed_so_they_can_be_found() {
    // Not a failure - stubs are known incomplete work. Printing them is the point: the list
    // is what someone implementing rules should work from, and it is measured rather than
    // copied from a comment.
    let reports = census();
    let mut stubs: Vec<String> = reports
        .iter()
        .filter(|r| r.verdict == Verdict::NoOp)
        .map(|r| format!("{} (applicable to {} witnesses)", r.key, r.applicable_to))
        .collect();
    stubs.sort();

    println!(
        "\n{} rules are applicable but never change anything:",
        stubs.len()
    );
    for stub in &stubs {
        println!("  {stub}");
    }

    assert_eq!(stubs.len(), EXPECTED_NO_OP);
}

#[test]
fn no_witness_rules_are_listed_so_the_corpus_can_be_widened() {
    // Also not a failure. A rule here is one this corpus never makes applicable, which is a
    // statement about the corpus as much as about the rule. The list is the input to
    // widening it.
    let reports = census();
    let mut unreached: Vec<String> = reports
        .iter()
        .filter(|r| r.verdict == Verdict::NoWitness)
        .map(|r| r.key.to_string())
        .collect();
    unreached.sort();

    println!(
        "
{} rules are not reached by this corpus:",
        unreached.len()
    );
    for name in &unreached {
        println!("  {name}");
    }

    assert_eq!(unreached.len(), EXPECTED_NO_WITNESS);
}

#[test]
fn a_rule_that_transforms_does_so_on_at_least_one_witness() {
    // Guards the census logic itself: the two counters must not disagree.
    for report in census() {
        match report.verdict {
            Verdict::Transforms => {
                assert!(report.transforms_on > 0);
                assert!(report.applicable_to >= report.transforms_on);
            }
            Verdict::NoOp => {
                assert!(report.applicable_to > 0);
                assert_eq!(report.transforms_on, 0);
            }
            Verdict::NoWitness => {
                assert_eq!(report.applicable_to, 0);
                assert_eq!(report.transforms_on, 0);
            }
        }
    }
}

#[test]
fn every_rule_receives_exactly_one_verdict() {
    let reports = census();
    let total: usize = counts(&reports).values().sum();
    assert_eq!(total, reports.len());
    assert_eq!(total, standard_rules().len());
}

#[test]
fn the_verdict_labels_are_distinct() {
    let labels: HashSet<&str> = [Verdict::Transforms, Verdict::NoOp, Verdict::NoWitness]
        .iter()
        .map(|v| v.label())
        .collect();
    assert_eq!(labels.len(), 3);
}

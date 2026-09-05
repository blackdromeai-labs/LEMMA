//! Paper measurement: how accepted transformations were established, reported at three
//! granularities that answer different questions:
//!
//! - **applications**: one count per accepted `RuleApplication` returned by `rule.apply`. A
//!   single (rule, witness) can yield more than one `RuleApplication` (e.g. the rule matches
//!   more than one subterm, or produces more than one candidate rewrite), so this is an upper
//!   bound on how many distinct (rule, witness) pairs were involved, not equal to it.
//! - **distinct (rule, witness) pairs**: the applications above deduplicated by
//!   `(RuleKey, witness index)`. This is what "how much of the corpus is checked by the weaker
//!   method" should actually be measured against -- it does not double-count a rule that
//!   produced two accepted rewrites of the same witness.
//! - **distinct rules**: rule keys with at least one acceptance by that method, further
//!   deduplicated across witnesses. A rule can appear under more than one method's set (e.g.
//!   accepted symbolically on one witness, numerically on another), so the three rule-level
//!   sets are not disjoint.
//!
//! "Accepted" means `Verifier::verify_step` returned a `VerificationMethod` for that
//! application on this corpus -- i.e. **verifier-accepted on at least one witness**, not
//! proven sound: numeric-sampling and rule-replay-only evidence are finite-sample checks, and
//! this script does not claim more for them than the corpus it ran on shows.
//!
//! `numeric sampling` counts (at every granularity) are not fully deterministic: the RNG in
//! `numerical::verify_equivalent` is unseeded. Run this multiple times before citing an exact
//! figure -- see `docs/evaluation/` for the measured run-to-run distribution, computed by
//! parsing captured output files, not by eye.

use std::collections::BTreeSet;

use mm_rules::{corpus, rule::RuleKey, standard_rules, RuleContext, WitnessSymbols};
use mm_verifier::{VerificationMethod, Verifier};

fn main() {
    let rules = standard_rules();
    let verifier = Verifier::new();
    let symbols = WitnessSymbols::new();
    let witnesses = corpus(&symbols);
    let ctx = RuleContext::default();

    let mut symbolic_applications = 0u32;
    let mut numeric_applications = 0u32;
    let mut replay_only_applications = 0u32;

    let mut symbolic_pairs: BTreeSet<(RuleKey, usize)> = BTreeSet::new();
    let mut numeric_pairs: BTreeSet<(RuleKey, usize)> = BTreeSet::new();
    let mut replay_only_pairs: BTreeSet<(RuleKey, usize)> = BTreeSet::new();

    let mut symbolic_rules: BTreeSet<RuleKey> = BTreeSet::new();
    let mut numeric_rules: BTreeSet<RuleKey> = BTreeSet::new();
    let mut replay_only_rules: BTreeSet<RuleKey> = BTreeSet::new();

    for (rule, key) in rules.all().iter().zip(rules.keys()) {
        for (witness_index, witness) in witnesses.iter().enumerate() {
            if !rule.can_apply(witness, &ctx) {
                continue;
            }
            for app in rule.apply(witness, &ctx) {
                if app.result == *witness {
                    continue;
                }
                let result = verifier.verify_step(witness, &app.result, rule, &ctx);
                if let Some(method) = result.method() {
                    let pair = (*key, witness_index);
                    match method {
                        VerificationMethod::SymbolicEquivalence => {
                            symbolic_applications += 1;
                            symbolic_pairs.insert(pair);
                            symbolic_rules.insert(*key);
                        }
                        VerificationMethod::NumericSampling => {
                            numeric_applications += 1;
                            numeric_pairs.insert(pair);
                            numeric_rules.insert(*key);
                        }
                        VerificationMethod::RuleReplayOnly => {
                            replay_only_applications += 1;
                            replay_only_pairs.insert(pair);
                            replay_only_rules.insert(*key);
                        }
                    }
                }
            }
        }
    }

    println!("accepted transformation APPLICATIONS by method (one count per accepted RuleApplication -- can exceed the number of distinct (rule, witness) pairs below if a rule matched a witness more than once):");
    println!("  symbolic equivalence : {symbolic_applications}");
    println!("  numeric sampling     : {numeric_applications}");
    println!("  rule replay only     : {replay_only_applications}");
    println!(
        "  total applications   : {}",
        symbolic_applications + numeric_applications + replay_only_applications
    );

    println!("\ndistinct (rule, witness) PAIRS accepted by method (deduplicated; a method's three counts here can still overlap with each other since the same pair can be accepted via more than one method on different applications):");
    println!("  symbolic equivalence : {}", symbolic_pairs.len());
    println!("  numeric sampling     : {}", numeric_pairs.len());
    println!("  rule replay only     : {}", replay_only_pairs.len());

    println!("\ndistinct RULES with at least one acceptance by method (not disjoint):");
    println!("  symbolic equivalence : {}", symbolic_rules.len());
    println!("  numeric sampling     : {}", numeric_rules.len());
    println!("  rule replay only     : {}", replay_only_rules.len());
    println!(
        "  union (any method)   : {}",
        symbolic_rules
            .union(&numeric_rules)
            .cloned()
            .collect::<BTreeSet<_>>()
            .union(&replay_only_rules)
            .count()
    );
}

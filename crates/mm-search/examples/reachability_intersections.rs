//! Paper measurement: the four rule-level predicates (executable, reachable, accepted) as
//! actual set intersections over the same witness corpus and rule set in one pass, rather than
//! three separately-run point counts that happen to share a corpus. Each existing pinned test
//! (`rule_census`, `guardrail_reachability`, `rule_acceptance`) measures one predicate
//! correctly on its own; none of them reports how the three relate to each other, which is
//! exactly what a funnel presentation would silently assume without checking.
//!
//! Definitions, applied per rule over the full witness corpus:
//! - **executable**: `rule.apply(witness)` produces a result different from `witness`, for at
//!   least one witness (this is `rule_census.rs`'s "Transforms" verdict).
//! - **reachable**: `mm_boink::filter_rules` offers the rule for at least one witness where it
//!   is executable (this is what `guardrail_reachability.rs` calls NOT hidden).
//! - **accepted**: `Verifier::verify_step` accepts at least one of the rule's outputs, on any
//!   witness, independent of whether the guardrail would ever offer that witness to search
//!   (this is `rule_acceptance.rs`'s "accepted somewhere").
//!
//! Note on `accepted`: this predicate is not fully deterministic. Numeric sampling in
//! `numerical::verify_equivalent` is unseeded, so a small number of borderline rules can flip
//! between runs. Run this multiple times if citing the accepted-set boundary precisely; see
//! the paper's evaluation section for the measured run-to-run distribution.

use std::collections::BTreeSet;

use mm_boink::{analyze, filter_rules};
use mm_rules::{corpus, rule::RuleKey, standard_rules, RuleContext, WitnessSymbols};
use mm_verifier::Verifier;

fn main() {
    // Optional seed argument. With a seed, every numeric-sampling draw this measurement makes
    // is reproducible (`mm_core::sampling`), so the accepted set becomes a deterministic
    // function of the seed instead of varying run to run. Without one, behaviour is unchanged:
    // the sampler seeds itself from entropy, as it always did.
    match std::env::args().nth(1).and_then(|s| s.parse::<u64>().ok()) {
        Some(seed) => {
            mm_core::sampling::seed_sampling_rng(seed);
            println!("sampling seed           : {seed}");
        }
        None => println!("sampling seed           : none (entropy; results will vary per run)"),
    }

    let rules = standard_rules();
    let verifier = Verifier::new();
    let symbols = WitnessSymbols::new();
    let witnesses = corpus(&symbols);
    let ctx = RuleContext::default();

    let mut executable: BTreeSet<RuleKey> = BTreeSet::new();
    let mut reachable: BTreeSet<RuleKey> = BTreeSet::new();
    let mut accepted: BTreeSet<RuleKey> = BTreeSet::new();

    for (rule, key) in rules.all().iter().zip(rules.keys()) {
        for witness in &witnesses {
            if !rule.can_apply(witness, &ctx) {
                continue;
            }
            for app in rule.apply(witness, &ctx) {
                if app.result == *witness {
                    continue;
                }
                executable.insert(*key);

                let profile = analyze(witness);
                let offered = filter_rules(rules.all(), &profile)
                    .iter()
                    .any(|r| r.id == rule.id);
                if offered {
                    reachable.insert(*key);
                }

                if verifier
                    .verify_step(witness, &app.result, rule, &ctx)
                    .is_valid()
                {
                    accepted.insert(*key);
                }
            }
        }
    }

    let registered: usize = rules.len();
    let reachable_and_accepted: BTreeSet<&RuleKey> = reachable.intersection(&accepted).collect();
    let hidden_but_accepted: BTreeSet<&RuleKey> = accepted.difference(&reachable).collect();
    let reachable_but_rejected: BTreeSet<&RuleKey> = reachable.difference(&accepted).collect();
    let executable_not_reachable: BTreeSet<&RuleKey> = executable.difference(&reachable).collect();

    println!("registered                : {registered}");
    println!("executable                : {}", executable.len());
    println!("reachable                 : {}", reachable.len());
    println!("accepted                  : {}", accepted.len());
    println!(
        "reachable AND accepted    : {}",
        reachable_and_accepted.len()
    );
    println!("accepted but NOT reachable: {}", hidden_but_accepted.len());
    println!(
        "reachable but NOT accepted: {}",
        reachable_but_rejected.len()
    );
    println!(
        "executable but NOT reachable: {}",
        executable_not_reachable.len()
    );

    println!("\naccepted-but-not-reachable rules (verifier-accepted on at least one witness, guardrail-hidden):");
    for key in &hidden_but_accepted {
        println!("  {key}");
    }

    println!("\nreachable-but-not-accepted rules (guardrail exposes them, verifier refuses):");
    for key in &reachable_but_rejected {
        println!("  {key}");
    }

    // Sanity: reachable rules are a subset of executable ones by construction (the guardrail
    // is only ever consulted for a witness where the rule already produced a change), and
    // accepted rules must also be executable (verify_step is only ever called on a produced,
    // changed result). If either of these fails, the three measurements are not actually
    // comparable and the intersection numbers above are not meaningful.
    assert!(
        reachable.is_subset(&executable),
        "reachable rules must be a subset of executable rules"
    );
    assert!(
        accepted.is_subset(&executable),
        "accepted rules must be a subset of executable rules"
    );
}

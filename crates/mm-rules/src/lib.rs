// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! # mm-rules
//!
//! Mathematical transformation rules for the LEMMA system.
//!
//! ## How many rules actually work
//!
//! 572 rules are registered. That is a count of constructors. Measured against the witness
//! corpus in [`witness`] (228 expressions), by `tests/rule_census.rs`:
//!
//! | Verdict | Count | Meaning |
//! |---|---:|---|
//! | transforms | 146 | applicable to some witness and returns a different expression |
//! | no-op | 237 | applicable, but always returns its input; a stub |
//! | not reached | 189 | no witness in the corpus makes it applicable |
//!
//! Of the 146 that transform, `mm-verifier`'s `tests/rule_acceptance.rs` finds 117 produce at
//! least one transformation the verifier accepts; the other 29 are refused on every witness
//! and so cannot be used by search.
//!
//! "Not reached" is a statement about the corpus, not proof that a rule is unreachable.
//! Widening the corpus can only move rules out of that bucket.
//!
//! These numbers come from a test run and are pinned there. Do not restate them from memory;
//! run the census.

pub mod action;
pub mod algebra;
pub mod backward;
pub mod board_exam;
pub mod calculus;
pub mod case_analysis;
pub mod combinatorics;
pub mod equations;
pub mod geometry;
pub mod guardrail;
pub mod induction;
pub mod inequalities;
pub mod inequality_chain;
pub mod integration;
pub mod number_theory;
pub mod patterns;
pub mod polynomial;
pub mod polynomials;
pub mod quantifier;
pub mod rule;
pub mod trig;
pub mod witness;

pub use action::{
    standard_action_vocabulary, ActionEntry, ActionError, ActionVocabulary,
    ACTION_VOCABULARY_VERSION,
};
pub use guardrail::{
    analyze, decompose_additive, filter_rules, is_rule_applicable, solvability_score,
    ProblemProfile,
};
pub use patterns::match_integral_pattern;
pub use rule::{
    standard_rules, try_standard_rules, Domain, Feature, RegistryError, Rule, RuleApplication,
    RuleCategory, RuleContext, RuleId, RuleKey, RuleSet,
};
pub use witness::{corpus, WitnessSymbols};

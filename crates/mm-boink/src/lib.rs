// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! # mm-boink
//!
//! Domain-based rule filtering for `mm-search`'s `NeuralMCTS`.
//!
//! [`analyze`] walks an expression into a [`ProblemProfile`] (which domains it touches:
//! trigonometry, calculus, number theory, ...), and [`filter_rules`] offers a rule only when
//! its declared `domains` overlap the profile (or it declares none, meaning universal). This
//! is a coarse pre-filter, not a correctness check: a rule's own `is_applicable` still gates
//! every application. See `mm-search/tests/guardrail_reachability.rs` for what this measures
//! and for a record of rules a bad domain tag made unreachable.
//!
//! This crate previously also carried a credit-tracking layer (`Bank`, `Budget`,
//! `BoinkSupervisor`) and a fast-path integral-pattern matcher (`patterns`), wired together by
//! `mm-search`'s `BoinkMCTS`. None of it was reachable from `mm-solver` or `mm-tui` — the
//! actual product only ever called `mm_boink::{analyze, filter_rules}` directly from
//! `NeuralMCTS` — and none of it had a test anywhere in this crate. It was removed rather than
//! wired in or documented as a caveat, since neither this crate nor `BoinkMCTS` made it
//! visible that solving a problem never touched the budget or credit numbers they reported.

pub mod guardrail;

pub use guardrail::{analyze, filter_rules, is_rule_applicable, ProblemProfile};

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! # mm-tui
//!
//! A terminal symbolic workbench for LEMMA.
//!
//! Three operations, all backed by the real `mm-solver` APIs: simplify, differentiate, and
//! check a candidate value against an equation. There is no natural-language mode and no
//! equation solving, because `IMOSolver::solve_text` and `LemmaSolver::solve_for` report
//! themselves as unsupported and unimplemented; offering them would be a lie in the menu.
//!
//! The interface exists to make verification status legible. `mm-verifier` distinguishes a
//! replayed, independently checked trace from one resting on numeric sampling or on rule
//! replay alone, and this UI keeps those apart in text, in colour, and in the per-step
//! evidence it shows. Only `CHECKED` is described as verified.
//!
//! ## Shape
//!
//! - [`app`] holds all state and the key-to-action reducer.
//! - [`presentation`] holds owned, render-ready values and the domain-to-badge mapping.
//! - [`worker`] owns the solver on its own thread and formats results while its symbol table
//!   is alive.
//! - [`ui`] draws a frame from [`app::App`] and nothing else.
//! - [`terminal`] owns raw mode and the alternate screen, and restores them on every exit
//!   path including a panic.

pub mod app;
pub mod presentation;
pub mod terminal;
pub mod ui;
pub mod worker;

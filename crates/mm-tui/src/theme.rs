// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The workbench's colour palette: ocean blue, teal, sea blue, and one warm accent (coral).
//!
//! Named eight-colour terminal colours (`Color::Cyan`, `Color::Red`, ...) are what the rest
//! of this codebase used to reach for, and a terminal without truecolor support still falls
//! back to something reasonable when it sees an RGB value close to one of its named colours.
//! These constants exist so the same handful of deliberately chosen colours are used
//! everywhere instead of whichever named colour happened to be nearby when a line was
//! written.
//!
//! Colour is still never the only carrier of meaning here — see `ui.rs`'s module doc. Every
//! one of these is paint on top of a text label and an ASCII marker that already say the same
//! thing.

use ratatui::style::Color;

/// Primary accent: focus, headers, the selected mode, primary navigation. The colour a reader
/// should associate with "this workbench" before any status-specific colour applies.
pub const OCEAN: Color = Color::Rgb(56, 158, 255);

/// Secondary accent: independently-checked evidence, the strongest verification outcomes.
pub const TEAL: Color = Color::Rgb(45, 212, 191);

/// Deep sea blue: used sparingly, for chrome that should read as "part of the frame" rather
/// than draw attention — the splash screen's resting state, mid-strength emphasis.
pub const SEA: Color = Color::Rgb(20, 70, 110);

/// Warm accent: errors, invalid input, unverified results. The one deliberate departure from
/// blue, so a failure state does not have to compete with the same hue as everything else.
pub const CORAL: Color = Color::Rgb(255, 111, 97);

/// Unfocused chrome and secondary text: a blue-grey rather than a neutral grey, so dimmed
/// elements still read as part of the same palette instead of drained of colour.
pub const DIM: Color = Color::Rgb(100, 116, 139);

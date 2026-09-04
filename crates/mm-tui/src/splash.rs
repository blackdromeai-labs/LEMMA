// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The boot screen shown once, before the workbench's own event loop starts.
//!
//! Purely cosmetic: it carries no state the rest of the application reads, and skipping it
//! entirely (any key press, or running headless under [`ratatui::backend::TestBackend`])
//! changes nothing about what the workbench can do. It cannot appear any earlier than this —
//! nothing running before `cargo` finishes compiling and starts this binary can draw to the
//! terminal, since compilation is `cargo`'s process, not this one — but it is the first thing
//! this program draws once it starts.
//!
//! [`frame_lines`] is the pure part — content for a given tick, with no timing or I/O — so it
//! can be checked without a real clock or terminal; [`run`] is the thin driver that actually
//! waits between frames and reads input.

use std::io;
use std::time::Duration;

use crossterm::event::{self, Event};
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::terminal::Guard;
use crate::theme;

/// The wordmark, one row per line, block characters only. Generated from an 8-row bitmap per
/// letter (`L`, `E`, `M`, `A`) joined with a one-column gap; every row is the same length by
/// construction, not by eyeballing it — see the width assertion in the test module.
const BANNER: [&str; 8] = [
    "██      ███████ █       █ █       █   ███  ",
    "██      ██      ██     ██ ██     ██  █   █ ",
    "██      ██      █ █   █ █ █ █   █ █ █     █",
    "██      ██      █  █ █  █ █  █ █  █ █     █",
    "██      █████   █   █   █ █   █   █ ███████",
    "██      ██      █       █ █       █ █     █",
    "██      ██      █       █ █       █ █     █",
    "███████ ███████ █       █ █       █ █     █",
];
const BANNER_WIDTH: usize = 43;

/// Ticks to sweep the colour wipe across the full banner width.
const REVEAL_TICKS: u32 = 80;
/// Ticks the screen holds, fully revealed, before [`run`] returns on its own.
///
/// Deliberately long: an animation that lands in under a second reads as a flicker, not a
/// boot screen. Held long enough to actually be seen, at the cost of a real, requested wait
/// on every launch — that trade was asked for explicitly, not a default this module chose.
const HOLD_TICKS: u32 = 142;
/// Last tick [`run`] will draw. Total wall-clock time is roughly this times [`FRAME_INTERVAL`]
/// — currently around ten seconds.
pub const TOTAL_FRAMES: u32 = REVEAL_TICKS + HOLD_TICKS;
/// Time between ticks. The terminal poll this drives also doubles as the "skip on any key"
/// wait, so it is short enough that a key press still feels immediate.
pub const FRAME_INTERVAL: Duration = Duration::from_millis(45);

/// Show the boot screen, ticking until [`TOTAL_FRAMES`] or until an actual key press arrives,
/// whichever is first.
///
/// Two things that are not a key press must not end it early, or it never runs at all:
///
/// - Entering the alternate screen / enabling raw mode can itself queue a `Resize` event, and
///   on some terminals a `FocusGained`. Either lands in the very first poll below, before a
///   human could possibly have reacted, and would otherwise kill the animation on frame one
///   every single time — indistinguishable from "no splash screen appearing" from the other
///   side of the terminal.
/// - Whatever was already queued before this ran — most commonly the Enter key that launched
///   `cargo run` itself, still sitting in the console's input buffer when raw mode turns on.
///   Drained up front so it cannot be mistaken for a real "skip the animation" key press.
pub fn run(guard: &mut Guard) -> io::Result<()> {
    while event::poll(Duration::ZERO)? {
        let _ = event::read();
    }

    for tick in 0..=TOTAL_FRAMES {
        guard.terminal().draw(|frame| draw(frame, tick))?;
        if event::poll(FRAME_INTERVAL)? {
            match event::read()? {
                Event::Key(_) => return Ok(()),
                // Not a key press: draw the next tick and keep going instead of treating
                // this as the signal to stop.
                _ => continue,
            }
        }
    }
    Ok(())
}

fn draw(frame: &mut Frame, tick: u32) {
    let area = frame.area();
    let lines = frame_lines(tick);
    let height = lines.len() as u16;
    let popup = Rect {
        x: area.x,
        y: area.y + area.height.saturating_sub(height) / 2,
        width: area.width,
        height: height.min(area.height),
    };
    frame.render_widget(Paragraph::new(lines).alignment(Alignment::Center), popup);
}

/// Content for animation tick `tick`. Pure: no timing, no I/O, safe to call directly in a
/// test with any tick value, including ones past [`TOTAL_FRAMES`] (the reveal just stays
/// complete).
fn frame_lines(tick: u32) -> Vec<Line<'static>> {
    let revealed_cols = ((tick as usize) * BANNER_WIDTH / REVEAL_TICKS as usize).min(BANNER_WIDTH);

    let mut lines: Vec<Line<'static>> = BANNER
        .iter()
        .map(|row| banner_row(row, revealed_cols))
        .collect();

    let fully_revealed = revealed_cols == BANNER_WIDTH;
    let after_reveal = tick.saturating_sub(REVEAL_TICKS);

    lines.push(Line::from(""));
    lines.push(subtitle_line("SYMBOLIC WORKBENCH", fully_revealed));
    lines.push(Line::from(""));
    lines.push(subtitle_line(
        "by BlackdromeAI Labs",
        fully_revealed && after_reveal >= 30,
    ));

    if fully_revealed && after_reveal >= 60 {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "press any key",
            Style::default().fg(theme::DIM),
        )));
    }

    lines
}

/// One row of the banner: the wordmark's full silhouette is visible from the first frame in
/// [`theme::DIM`], and a colour wipe paints over it left to right as `revealed_cols` grows —
/// so the shape reads as "LEMMA" immediately and the animation is the colour arriving, not
/// the letters.
fn banner_row(row: &'static str, revealed_cols: usize) -> Line<'static> {
    // `.chars().enumerate()` deliberately, not `.char_indices()`: the block character `█` is
    // three bytes in UTF-8, so a byte offset would run far ahead of `revealed_cols` (a column
    // count) and leave most of the banner permanently un-swept. Caught by
    // `the_sweep_fully_colours_the_banner_once_reveal_ticks_pass` below.
    let spans: Vec<Span<'static>> = row
        .chars()
        .enumerate()
        .map(|(col, ch)| {
            if ch == ' ' {
                return Span::raw(" ");
            }
            let style = if col < revealed_cols {
                Style::default()
                    .fg(sweep_color(col))
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::DIM)
            };
            Span::styled(ch.to_string(), style)
        })
        .collect();
    Line::from(spans)
}

/// Colour for column `col` of the sweep: a gradient from ocean to teal across the width, with
/// the last few columns forced to coral — a small accent on the landing rather than a
/// gradient that ends on an arbitrary in-between colour.
fn sweep_color(col: usize) -> Color {
    if col + 3 >= BANNER_WIDTH {
        return theme::CORAL;
    }
    let t = col as f32 / BANNER_WIDTH as f32;
    lerp(theme::OCEAN, theme::TEAL, t)
}

fn lerp(from: Color, to: Color, t: f32) -> Color {
    let (Color::Rgb(r0, g0, b0), Color::Rgb(r1, g1, b1)) = (from, to) else {
        return from;
    };
    let mix = |a: u8, b: u8| -> u8 { (a as f32 + (b as f32 - a as f32) * t).round() as u8 };
    Color::Rgb(mix(r0, r1), mix(g0, g1), mix(b0, b1))
}

/// A secondary line that is present in the layout from the first frame (so nothing shifts
/// vertically as it appears) but only styled visibly once `visible` is true.
fn subtitle_line(text: &'static str, visible: bool) -> Line<'static> {
    let style = if visible {
        Style::default().fg(theme::TEAL)
    } else {
        Style::default().fg(Color::Reset)
    };
    Line::from(Span::styled(text, style))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn every_banner_row_is_the_same_width() {
        for row in BANNER {
            assert_eq!(
                row.chars().count(),
                BANNER_WIDTH,
                "row {row:?} is not {BANNER_WIDTH} columns wide"
            );
        }
    }

    #[test]
    fn nothing_is_swept_at_tick_zero() {
        let lines = frame_lines(0);
        // Every banner row's spans should be either a raw space or styled with `theme::DIM`
        // (the un-swept silhouette) -- none should carry a sweep colour yet.
        for line in &lines[..BANNER.len()] {
            for span in &line.spans {
                let is_space = span.content.trim().is_empty();
                let is_dim = span.style.fg == Some(theme::DIM);
                assert!(is_space || is_dim, "unexpected styling at tick 0: {span:?}");
            }
        }
    }

    #[test]
    fn the_sweep_fully_colours_the_banner_once_reveal_ticks_pass() {
        let lines = frame_lines(REVEAL_TICKS);
        for line in &lines[..BANNER.len()] {
            for span in &line.spans {
                let is_space = span.content.trim().is_empty();
                let is_dim = span.style.fg == Some(theme::DIM);
                assert!(
                    is_space || !is_dim,
                    "expected full sweep by tick {REVEAL_TICKS}, found unswept: {span:?}"
                );
            }
        }
    }

    #[test]
    fn the_hint_only_appears_after_the_sweep_and_both_subtitles_have_landed() {
        let just_revealed = REVEAL_TICKS;
        let early = frame_lines(just_revealed);
        assert!(
            !early
                .iter()
                .any(|l| l.to_string().contains("press any key")),
            "hint appeared before its delay"
        );

        let late = frame_lines(TOTAL_FRAMES);
        assert!(
            late.iter().any(|l| l.to_string().contains("press any key")),
            "hint never appeared by the last frame"
        );
    }

    #[test]
    fn renders_without_panicking_at_every_tick_on_a_realistic_backend() {
        let backend = TestBackend::new(90, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        for tick in 0..=TOTAL_FRAMES {
            terminal.draw(|frame| draw(frame, tick)).unwrap();
        }
    }

    #[test]
    fn renders_without_panicking_on_a_minimum_size_backend() {
        // The workbench's own floor is smaller than this, but the splash only needs to not
        // panic -- it does not claim to be legible below the workbench's stated minimum.
        let backend = TestBackend::new(40, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, TOTAL_FRAMES)).unwrap();
    }

    #[test]
    fn fits_within_the_workbenchs_stated_minimum_width() {
        assert!(
            BANNER_WIDTH <= crate::app::MIN_WIDTH as usize,
            "banner ({BANNER_WIDTH} cols) is wider than the workbench's own MIN_WIDTH"
        );
    }
}

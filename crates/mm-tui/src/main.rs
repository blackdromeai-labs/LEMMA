// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Entry point and event loop for the LEMMA symbolic workbench.

use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event};

use mm_tui::app::{action_for_key, App, Overlay};
use mm_tui::terminal::Guard;
use mm_tui::ui;
use mm_tui::worker::Worker;

/// How long to wait for input before drawing again.
///
/// The loop is not a busy spinner: it blocks on input for this long, so an idle workbench
/// costs nothing. It is short enough that the running indicator still animates and a worker
/// reply is picked up promptly.
const TICK: Duration = Duration::from_millis(120);

fn main() -> io::Result<()> {
    let mut guard = Guard::new()?;
    let outcome = run(&mut guard);

    // Restore before reporting: `Guard`'s `Drop` runs at the end of this function, and an
    // error printed while the alternate screen is still up would be discarded with it.
    drop(guard);

    if let Err(error) = &outcome {
        eprintln!("mm-tui: {error}");
    }
    outcome
}

fn run(guard: &mut Guard) -> io::Result<()> {
    let worker = Worker::spawn();
    let mut app = App::new();

    // Displayed in the header. Counting the registry here rather than asking the worker keeps
    // the first frame from waiting on the solver.
    app.rule_count = mm_rules_count();

    let mut last_tick = Instant::now();

    while !app.should_quit {
        guard.terminal().draw(|frame| ui::draw(frame, &app))?;

        let timeout = TICK.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(key) => {
                    // History navigation is handled here because it drives the overlay list
                    // rather than the trace, and only while that overlay is open.
                    if app.overlay == Some(Overlay::History) {
                        match key.code {
                            crossterm::event::KeyCode::Up => {
                                app.move_history_cursor(-1);
                                continue;
                            }
                            crossterm::event::KeyCode::Down => {
                                app.move_history_cursor(1);
                                continue;
                            }
                            _ => {}
                        }
                    }

                    if let Some(action) = action_for_key(&app, key) {
                        if let Some(request) = app.apply(action) {
                            if !worker.submit(request) {
                                // The solver thread is gone; there is no way to make progress.
                                app.should_quit = true;
                            }
                        }
                    }
                }
                // Ratatui reads the size each frame, so a resize only needs to wake the loop.
                Event::Resize(_, _) => {}
                _ => {}
            }
        }

        for response in worker.drain() {
            app.accept(response);
        }

        if last_tick.elapsed() >= TICK {
            app.ticks = app.ticks.wrapping_add(1);
            last_tick = Instant::now();
        }
    }

    Ok(())
}

/// Size of the standard rule registry, for the header.
fn mm_rules_count() -> usize {
    mm_solver::LemmaSolver::new().num_rules()
}

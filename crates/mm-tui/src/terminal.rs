// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Terminal setup and, more importantly, teardown.
//!
//! Raw mode and the alternate screen are process-global state. Leaving either behind gives
//! the user a shell with no echo and no visible cursor, which reads as a hung machine. Three
//! paths have to restore it: normal exit, an error return, and a panic. The first two are
//! covered by [`Guard`]'s `Drop`; the third needs a panic hook, because a panic while the
//! alternate screen is active would otherwise print its message onto a screen that is about
//! to be discarded.

use std::io::{self, Stdout, Write};
use std::panic;

use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen, SetTitle,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

/// Window/tab title while the workbench is running.
///
/// Set once, on entry, and deliberately not restored on exit: `SetTitle` has no matching
/// "read the current title" so there is nothing to put back, and every terminal emulator this
/// was tried against already replaces it with the shell's own title once this process exits.
const TITLE: &str = "LEMMA — Symbolic Workbench (BlackdromeAI Labs)";

/// Terminal handle whose `Drop` restores the console.
pub struct Guard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl Guard {
    /// Enter raw mode and the alternate screen, and install the panic hook.
    pub fn new() -> io::Result<Self> {
        install_panic_hook();

        enable_raw_mode()?;
        let mut stdout = io::stdout();
        // Mouse capture is enabled only so the terminal does not pass drag-selection through
        // as escape sequences; no mouse handling is implemented.
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableMouseCapture,
            SetTitle(TITLE)
        )?;

        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;
        Ok(Self { terminal })
    }

    /// The Ratatui terminal.
    pub fn terminal(&mut self) -> &mut Terminal<CrosstermBackend<Stdout>> {
        &mut self.terminal
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        // Best effort: if restoration fails there is nowhere useful left to report it, and
        // returning early would skip the remaining steps.
        let _ = restore();
        let _ = self.terminal.show_cursor();
    }
}

/// Undo everything [`Guard::new`] did.
///
/// Safe to call more than once, which matters because the panic hook and `Drop` can both run.
pub fn restore() -> io::Result<()> {
    let mut stdout = io::stdout();
    let _ = execute!(stdout, DisableMouseCapture, LeaveAlternateScreen);
    let _ = disable_raw_mode();
    let _ = crossterm::execute!(stdout, crossterm::cursor::Show);
    stdout.flush()
}

/// Restore the terminal before the default panic handler prints anything.
///
/// Without this, a panic message is written to the alternate screen and vanishes with it, and
/// the shell is left in raw mode.
fn install_panic_hook() {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = restore();
        previous(info);
    }));
}

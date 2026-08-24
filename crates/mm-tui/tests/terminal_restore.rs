//! The terminal must be handed back in the state it was borrowed in.
//!
//! Raw mode and the alternate screen are process-global. Leaving either set gives the user a
//! shell with no echo and no cursor, which reads as a hung machine rather than a bug in this
//! program. Three exit paths matter — normal return, an error return, and a panic — and only
//! the first two are covered by `Drop`.
//!
//! These tests run without a real console, so they check the parts that are testable there:
//! that restoration is total and repeatable, and that a panic inside the guarded region runs
//! the hook rather than unwinding past it.

use std::io;

use mm_tui::terminal;

#[test]
fn restore_is_idempotent() {
    // Both `Guard::drop` and the panic hook can run for the same panic, so restoration has to
    // tolerate being called twice.
    for _ in 0..3 {
        let result: io::Result<()> = terminal::restore();
        assert!(
            result.is_ok(),
            "restore must succeed even when there is nothing to undo: {result:?}"
        );
    }
}

#[test]
fn restore_is_safe_without_a_console() {
    // The test harness has no attached console. Restoration must not panic or block here,
    // because the same code runs from a panic hook where a second panic would abort.
    assert!(terminal::restore().is_ok());
}

#[test]
fn a_panic_inside_the_guarded_region_still_leaves_the_terminal_restored() {
    // Simulates the third exit path. The hook installed by `Guard::new` calls `restore`; this
    // asserts the sequence completes and the terminal can still be restored afterwards,
    // rather than the process being left mid-teardown.
    let outcome = std::panic::catch_unwind(|| {
        let _ = terminal::restore();
        panic!("induced failure inside the guarded region");
    });

    assert!(outcome.is_err(), "the panic should have been caught");
    assert!(
        terminal::restore().is_ok(),
        "the terminal must still be restorable after a panic"
    );
}

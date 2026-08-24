// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Rendering. A pure function of [`App`]; it holds no state of its own.
//!
//! Three layouts, chosen by terminal size: side-by-side at 110 columns or more, stacked from
//! 80, and a notice below that. The notice exists because a layout squeezed into 40x10
//! produces overlapping borders and zero-height panes rather than an error, which looks like
//! a crash without being one.
//!
//! Colour is never the only carrier of meaning. Every status shows its text label and a short
//! ASCII marker, so an 8-colour terminal, a monochrome one, and a screen reader all still
//! convey the verification state.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap,
};
use ratatui::Frame;

use mm_core::truncate_chars;

use crate::app::{App, Focus, Job, Overlay, MIN_HEIGHT, MIN_WIDTH, WIDE_WIDTH};
use crate::presentation::{ErrorField, Mode, StatusBadge, UiResult, UiStep};

/// Accent used for focus and navigation. Exactly one, on purpose.
const ACCENT: Color = Color::Cyan;
/// Border colour for unfocused panes.
const DIM: Color = Color::DarkGray;

/// Draw the whole application.
pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();

    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        draw_too_small(frame, area);
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header
            Constraint::Length(1), // operation selector
            Constraint::Length(form_height(app)),
            Constraint::Min(6),    // result / trace / detail
            Constraint::Length(1), // footer
        ])
        .split(area);

    draw_header(frame, chunks[0], app);
    draw_modes(frame, chunks[1], app);
    draw_form(frame, chunks[2], app);
    draw_body(frame, chunks[3], app);
    draw_footer(frame, chunks[4], app);

    match app.overlay {
        Some(Overlay::Help) => draw_help(frame, area),
        Some(Overlay::History) => draw_history(frame, area, app),
        None => {}
    }
}

/// Height needed by the form: one row per visible field, plus the hint line.
fn form_height(app: &App) -> u16 {
    let mut rows = 1; // expression
    if app.mode.uses_variable() {
        rows += 1;
    }
    if app.mode.uses_candidate() {
        rows += 1;
    }
    rows + 3 // borders plus the hint row
}

fn draw_too_small(frame: &mut Frame, area: Rect) {
    let text = vec![
        Line::from(Span::styled(
            "Terminal too small",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(format!(
            "Have {}x{}, need at least {}x{}.",
            area.width, area.height, MIN_WIDTH, MIN_HEIGHT
        )),
        Line::from("Resize the window, or press Ctrl+Q to quit."),
    ];
    // No border: at this size a border would consume most of the available rows.
    frame.render_widget(Paragraph::new(text).wrap(Wrap { trim: true }), area);
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App) {
    let left = Span::styled(
        " LEMMA · SYMBOLIC WORKBENCH ",
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    );
    let right = format!("{} rules · formal input ", app.rule_count);

    let used = left.content.chars().count() + right.chars().count();
    let gap = (area.width as usize).saturating_sub(used);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            left,
            Span::raw(" ".repeat(gap)),
            Span::styled(right, Style::default().fg(DIM)),
        ])),
        area,
    );
}

fn draw_modes(frame: &mut Frame, area: Rect, app: &App) {
    let mut spans = vec![Span::raw(" ")];
    for (index, mode) in Mode::ALL.iter().enumerate() {
        let selected = *mode == app.mode;
        let style = if selected {
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        // The digit is the shortcut, so it is part of the label rather than a legend.
        spans.push(Span::styled(
            format!(" {} {} ", index + 1, mode.label()),
            style,
        ));
        spans.push(Span::raw(" "));
    }
    spans.push(Span::styled("[? Help]", Style::default().fg(DIM)));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_form(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus.is_editor();
    frame.render_widget(block("Input", focused), area);

    let inner = area.inner(ratatui::layout::Margin::new(1, 1));
    if inner.height == 0 {
        return;
    }

    let mut constraints = vec![Constraint::Length(1)];
    if app.mode.uses_variable() {
        constraints.push(Constraint::Length(1));
    }
    if app.mode.uses_candidate() {
        constraints.push(Constraint::Length(1));
    }
    constraints.push(Constraint::Length(1)); // hint

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(inner);

    let mut row = 0;
    draw_field(frame, rows[row], app, Focus::Expression, "expression");
    row += 1;

    if app.mode.uses_variable() {
        draw_field(frame, rows[row], app, Focus::Variable, "variable");
        row += 1;
    }
    if app.mode.uses_candidate() {
        draw_field(frame, rows[row], app, Focus::Candidate, "candidate");
        row += 1;
    }

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            app.mode.hint(),
            Style::default().fg(DIM),
        ))),
        rows[row],
    );
}

/// One labelled editor row, plus its inline error if the last run blamed this field.
fn draw_field(frame: &mut Frame, area: Rect, app: &App, focus: Focus, label: &str) {
    let label_width = 11u16;
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(label_width), Constraint::Min(1)])
        .split(area);

    let is_focused = app.focus == focus;
    let field_error = app.error_field() == Some(error_field_for(focus));

    let label_style = if is_focused {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("{label:<10}"),
            label_style,
        ))),
        columns[0],
    );

    let editor = match focus {
        Focus::Expression => &app.expression,
        Focus::Variable => &app.variable,
        Focus::Candidate => &app.candidate,
        Focus::Trace => return,
    };

    // The error message shares the row with the editor rather than adding a line, so the form
    // does not change height when a run fails.
    if field_error {
        if let Job::Failed(error) = &app.job {
            let split = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(columns[1]);
            frame.render_widget(editor, split[0]);
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    truncate_chars(&error.message, split[1].width.max(1) as usize),
                    Style::default().fg(Color::Red),
                ))),
                split[1],
            );
            return;
        }
    }

    frame.render_widget(editor, columns[1]);
}

fn error_field_for(focus: Focus) -> ErrorField {
    match focus {
        Focus::Expression => ErrorField::Expression,
        Focus::Variable => ErrorField::Variable,
        Focus::Candidate => ErrorField::Candidate,
        Focus::Trace => ErrorField::Request,
    }
}

/// Result, trace and detail. Side by side when there is room, stacked otherwise.
fn draw_body(frame: &mut Frame, area: Rect, app: &App) {
    if area.width >= WIDE_WIDTH {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(area);

        let top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[0]);

        draw_result(frame, top[0], app);
        draw_trace(frame, top[1], app);
        draw_detail(frame, rows[1], app);
    } else {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(34),
                Constraint::Percentage(33),
                Constraint::Percentage(33),
            ])
            .split(area);

        draw_result(frame, rows[0], app);
        draw_trace(frame, rows[1], app);
        draw_detail(frame, rows[2], app);
    }
}

fn draw_result(frame: &mut Frame, area: Rect, app: &App) {
    frame.render_widget(block("Result", false), area);
    let inner = area.inner(ratatui::layout::Margin::new(1, 1));
    if inner.height == 0 {
        return;
    }

    let lines: Vec<Line> = match &app.job {
        Job::Idle => vec![
            Line::from(Span::styled(
                "Enter an expression and press Ctrl+Enter.",
                Style::default().fg(Color::Gray),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "This workbench takes formal LEMMA syntax, not prose.",
                Style::default().fg(DIM),
            )),
        ],
        Job::Running { .. } => vec![
            Line::from(vec![
                Span::styled(spinner(app.ticks), Style::default().fg(ACCENT)),
                Span::raw(" "),
                Span::styled(
                    "RUNNING",
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Solving. The interface stays responsive; the run itself cannot be cancelled.",
                Style::default().fg(DIM),
            )),
        ],
        Job::Failed(error) => vec![
            badge_line(StatusBadge::InputError),
            Line::from(""),
            Line::from(Span::styled(
                format!("{}: {}", error.field.label(), error.message),
                Style::default().fg(Color::Red),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Nothing was solved: the input could not be read.",
                Style::default().fg(DIM),
            )),
        ],
        Job::Complete(result) => result_lines(result),
    };

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn result_lines(result: &UiResult) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(Span::styled(
            result.output.clone(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        badge_line(result.badge),
        Line::from(Span::styled(
            result.reason.clone(),
            Style::default().fg(Color::Gray),
        )),
        Line::from(""),
    ];

    lines.push(Line::from(Span::styled(
        format!(
            "{} {} · {} ms",
            result.steps.len(),
            plural(result.steps.len(), "step", "steps"),
            result.elapsed.as_millis()
        ),
        Style::default().fg(DIM),
    )));

    // The distinction the hardening work introduced, said out loud where a reader will see it.
    // Only for trace-verification states: a candidate check is a substitution test, and the
    // disclaimer would understate a symbolic one rather than qualify it.
    if matches!(
        result.badge,
        StatusBadge::Partial
            | StatusBadge::Heuristic
            | StatusBadge::Unverified
            | StatusBadge::NotFound
    ) {
        lines.push(Line::from(Span::styled(
            "Not a proof of correctness.",
            Style::default().fg(DIM),
        )));
    }

    lines
}

/// Pick a singular or plural noun for `count`.
fn plural(count: usize, one: &'static str, many: &'static str) -> &'static str {
    if count == 1 {
        one
    } else {
        many
    }
}

fn badge_line(badge: StatusBadge) -> Line<'static> {
    Line::from(vec![Span::styled(
        format!(" {} {} ", badge.marker(), badge.label()),
        Style::default()
            .fg(Color::Black)
            .bg(badge_color(badge))
            .add_modifier(Modifier::BOLD),
    )])
}

fn badge_color(badge: StatusBadge) -> Color {
    match badge {
        StatusBadge::Checked | StatusBadge::CandidateValid => Color::Green,
        StatusBadge::Partial => Color::Yellow,
        StatusBadge::Heuristic => Color::Cyan,
        StatusBadge::Unverified | StatusBadge::CandidateInvalid | StatusBadge::InputError => {
            Color::Red
        }
        StatusBadge::Unsupported => Color::Gray,
        StatusBadge::NotFound => Color::Yellow,
    }
}

fn spinner(ticks: u64) -> &'static str {
    const FRAMES: [&str; 4] = ["-", "\\", "|", "/"];
    FRAMES[(ticks / 2) as usize % FRAMES.len()]
}

fn draw_trace(frame: &mut Frame, area: Rect, app: &App) {
    let count = app.step_count();
    let title = format!("Trace · {count} {}", plural(count, "step", "steps"));
    frame.render_widget(block(&title, app.focus == Focus::Trace), area);

    let inner = area.inner(ratatui::layout::Margin::new(1, 1));
    if inner.height == 0 {
        return;
    }

    let Some(result) = app.result() else {
        let message = if app.is_running() {
            "Waiting for the solver."
        } else {
            "No trace yet."
        };
        frame.render_widget(
            Paragraph::new(Span::styled(message, Style::default().fg(DIM))),
            inner,
        );
        return;
    };

    if result.steps.is_empty() {
        let message = match result.mode {
            // A candidate check substitutes and compares; it has no transformations to show,
            // and an empty list is more honest than a fabricated one.
            Mode::VerifyCandidate => "A candidate check records no transformations.",
            _ => "No transformations were recorded.",
        };
        frame.render_widget(
            Paragraph::new(Span::styled(message, Style::default().fg(DIM)))
                .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    }

    let width = inner.width as usize;
    let items: Vec<ListItem> = result
        .steps
        .iter()
        .map(|step| {
            let style = if step.evidence.checked {
                if step.evidence.independent {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Cyan)
                }
            } else {
                Style::default().fg(Color::Red)
            };
            ListItem::new(Line::from(Span::styled(
                truncate_chars(&step.summary(), width.saturating_sub(2).max(1)),
                style,
            )))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected_step.min(result.steps.len() - 1)));

    frame.render_stateful_widget(
        List::new(items).highlight_symbol("> ").highlight_style(
            Style::default()
                .bg(ACCENT)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        inner,
        &mut state,
    );
}

fn draw_detail(frame: &mut Frame, area: Rect, app: &App) {
    frame.render_widget(block("Selected step", false), area);
    let inner = area.inner(ratatui::layout::Margin::new(1, 1));
    if inner.height == 0 {
        return;
    }

    let step = app
        .result()
        .and_then(|result| result.steps.get(app.selected_step));

    let Some(step) = step else {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "Select a step to see its before/after, rule and evidence.",
                Style::default().fg(DIM),
            ))
            .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    };

    frame.render_widget(
        Paragraph::new(detail_lines(step)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn detail_lines(step: &UiStep) -> Vec<Line<'static>> {
    let rule = match step.rule_id {
        Some(id) => format!("{} · id {}", step.rule, id),
        // Normalisation steps have no registry identifier, and inventing one would give them
        // an identity they do not have.
        None => format!("{} · not a registry rule", step.rule),
    };

    let evidence_style = if step.evidence.checked {
        if step.evidence.independent {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::Cyan)
        }
    } else {
        Style::default().fg(Color::Red)
    };

    vec![
        labelled("Rule", rule, Style::default().fg(Color::White)),
        labelled(
            "Before",
            step.before.clone(),
            Style::default().fg(Color::Gray),
        ),
        labelled(
            "After",
            step.after.clone(),
            Style::default().fg(Color::White),
        ),
        labelled("Evidence", step.evidence.label.clone(), evidence_style),
        labelled(
            "Why",
            step.justification.clone(),
            Style::default().fg(Color::Gray),
        ),
    ]
}

fn labelled(label: &str, value: String, style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<9}"), Style::default().fg(DIM)),
        Span::styled(value, style),
    ])
}

fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let keys = if app.focus.is_editor() {
        "Tab focus · Ctrl+Enter run · Ctrl+L clear · h history · ? help · Ctrl+Q quit"
    } else {
        "1/2/3 mode · Tab focus · ↑↓ trace · Ctrl+Enter run · h history · ? help · q quit"
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            truncate_chars(keys, area.width.max(1) as usize),
            Style::default().fg(DIM),
        ))),
        area,
    );
}

fn draw_help(frame: &mut Frame, area: Rect) {
    // Sized to fit the content at the minimum supported terminal height; `centred` clamps it
    // if the terminal is shorter, and the paragraph then scrolls out rather than overflowing.
    let popup = centred(area, 68, 24);
    frame.render_widget(Clear, popup);
    frame.render_widget(block("Help", true), popup);

    let inner = popup.inner(ratatui::layout::Margin::new(1, 1));
    let lines = vec![
        section("Operations"),
        entry("1", "Simplify an expression"),
        entry("2", "Differentiate with respect to a variable"),
        entry("3", "Check a candidate value against an equation"),
        Line::from(""),
        section("Keys"),
        entry("Tab / Shift+Tab", "Move focus"),
        entry("Ctrl+Enter", "Run"),
        entry("Ctrl+L", "Clear the form"),
        entry("Up / Down", "Move through the trace"),
        entry("PgUp / PgDn", "Jump through the trace"),
        entry("h", "Recent requests"),
        entry("?", "This help"),
        entry("Esc", "Close an overlay"),
        entry("q / Ctrl+Q", "Quit"),
        Line::from(""),
        section("Input"),
        // Each line is kept inside the popup's inner width so wrapping cannot split a
        // statement the user needs to read whole.
        Line::from(Span::styled(
            "Formal syntax only: x^2, sin(x), diff(x^3, x), x + 3 = 7.",
            Style::default().fg(Color::Gray),
        )),
        Line::from(Span::styled(
            "Prose problems and equation solving are not supported.",
            Style::default().fg(Color::Gray),
        )),
        Line::from(Span::styled(
            "Only CHECKED means verified.",
            Style::default().fg(Color::Gray),
        )),
        Line::from(Span::styled(
            "Other states are evidence, not proof.",
            Style::default().fg(Color::Gray),
        )),
    ];
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn section(title: &str) -> Line<'static> {
    Line::from(Span::styled(
        title.to_string(),
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    ))
}

fn entry(key: &str, description: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key:<16}"), Style::default().fg(Color::White)),
        Span::styled(description.to_string(), Style::default().fg(Color::Gray)),
    ])
}

fn draw_history(frame: &mut Frame, area: Rect, app: &App) {
    let popup = centred(area, 76, 18);
    frame.render_widget(Clear, popup);
    frame.render_widget(
        block(
            &format!(
                "Recent · {} of {}",
                app.history.len(),
                crate::app::HISTORY_LIMIT
            ),
            true,
        ),
        popup,
    );

    let inner = popup.inner(ratatui::layout::Margin::new(1, 1));
    if inner.height == 0 {
        return;
    }

    if app.history.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "Nothing yet. Completed requests appear here, in memory only.",
                Style::default().fg(DIM),
            ))
            .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    }

    let width = inner.width.saturating_sub(2).max(1) as usize;
    let items: Vec<ListItem> = app
        .history
        .iter()
        .map(|entry| {
            ListItem::new(Line::from(Span::raw(truncate_chars(
                &entry.result.history_summary(),
                width,
            ))))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.history_cursor.min(app.history.len() - 1)));

    frame.render_stateful_widget(
        List::new(items).highlight_symbol("> ").highlight_style(
            Style::default()
                .bg(ACCENT)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        inner,
        &mut state,
    );
}

/// A bordered pane. Focused panes take the accent, everything else stays dim.
fn block(title: &str, focused: bool) -> Block<'_> {
    let border_style = if focused {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(DIM)
    };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(border_style)
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(if focused { ACCENT } else { Color::Gray }),
        ))
}

/// Centre a popup, clamped so it always fits.
fn centred(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

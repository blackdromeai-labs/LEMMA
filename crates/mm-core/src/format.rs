// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Readable, precedence-aware rendering of [`Expr`].
//!
//! `Debug` output is a Rust AST dump: `Add(Mul(Const(2), Var(SymbolU32 { value: 1 })), Const(3))`
//! for `2 * x + 3`. It is unreadable as mathematics and it leaks symbol internals, so it must
//! not be what a user sees.
//!
//! The output here is the same surface syntax [`crate::parse`] accepts, so a rendered
//! expression can be pasted back in. Parentheses are inserted only where removing them would
//! change the tree: `(a + b) * c` keeps them, `a + b * c` does not.
//!
//! A [`SymbolTable`] resolves variable names. Symbols are indices into a particular table, so
//! rendering must happen while the table that created them is still available; a symbol from
//! another table renders as a placeholder rather than panicking.

use std::fmt::Write as _;

use crate::{Expr, Rational, Symbol, SymbolTable};

/// Binding strength, loosest first. Used to decide when a child needs parentheses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Prec {
    /// `forall`, `exists`
    Quantifier,
    /// `=>`
    Implies,
    /// `or`
    Or,
    /// `and`
    And,
    /// `not`
    Not,
    /// `=`, `<`, `<=`, `>`, `>=`
    Relation,
    /// `+`, `-`
    Sum,
    /// `*`, `/`, `mod`
    Product,
    /// unary `-`
    Neg,
    /// `^`
    Power,
    /// `!`
    Factorial,
    /// atoms and `f(...)` calls
    Atom,
}

/// How a symbol renders when the table does not know it.
///
/// Keeps rendering total: an expression built against a different `SymbolTable` still produces
/// output a human can read and report, instead of panicking inside the UI.
const UNKNOWN_SYMBOL: &str = "?";

/// Render an expression as readable formal syntax.
///
/// ```
/// use mm_core::{format_expr, Expr, SymbolTable};
///
/// let mut symbols = SymbolTable::new();
/// let x = symbols.intern("x");
/// let expr = Expr::Mul(
///     Box::new(Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(1)))),
///     Box::new(Expr::int(2)),
/// );
/// assert_eq!(format_expr(&expr, &symbols), "(x + 1) * 2");
/// ```
pub fn format_expr(expr: &Expr, symbols: &SymbolTable) -> String {
    let mut out = String::new();
    write_expr(&mut out, expr, symbols, Prec::Quantifier);
    out
}

/// Render an expression, truncating to `max_chars` on a character boundary.
///
/// Truncation counts characters, not bytes, so multi-byte input cannot be split mid-scalar.
pub fn format_expr_truncated(expr: &Expr, symbols: &SymbolTable, max_chars: usize) -> String {
    let full = format_expr(expr, symbols);
    truncate_chars(&full, max_chars)
}

/// Truncate to `max_chars` characters, appending an ellipsis when anything was removed.
pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    if max_chars <= 1 {
        return text.chars().take(max_chars).collect();
    }
    let mut out: String = text.chars().take(max_chars - 1).collect();
    out.push('…');
    out
}

/// Resolve a symbol's name, or [`UNKNOWN_SYMBOL`] if this table does not contain it.
pub fn symbol_name(symbol: Symbol, symbols: &SymbolTable) -> &str {
    symbols.resolve(symbol).unwrap_or(UNKNOWN_SYMBOL)
}

/// Render a constant. Integers render bare, other rationals as `n/d`.
fn write_rational(out: &mut String, value: &Rational) {
    if value.is_integer() {
        let _ = write!(out, "{}", value.numer());
    } else {
        let _ = write!(out, "{}/{}", value.numer(), value.denom());
    }
}

/// Precedence of the operator at the root of `expr`.
fn precedence(expr: &Expr) -> Prec {
    match expr {
        Expr::Const(_) | Expr::Var(_) | Expr::Pi | Expr::E => Prec::Atom,

        // Rendered as `f(...)`, so they bind as tightly as an atom.
        Expr::Sqrt(_)
        | Expr::Sin(_)
        | Expr::Cos(_)
        | Expr::Tan(_)
        | Expr::Arcsin(_)
        | Expr::Arccos(_)
        | Expr::Arctan(_)
        | Expr::Ln(_)
        | Expr::Exp(_)
        | Expr::Abs(_)
        | Expr::Floor(_)
        | Expr::Ceiling(_)
        | Expr::GCD(_, _)
        | Expr::LCM(_, _)
        | Expr::Binomial(_, _)
        | Expr::Derivative { .. }
        | Expr::Integral { .. }
        | Expr::Summation { .. }
        | Expr::BigProduct { .. } => Prec::Atom,

        Expr::Factorial(_) => Prec::Factorial,
        Expr::Pow(_, _) => Prec::Power,
        Expr::Neg(_) => Prec::Neg,
        Expr::Mul(_, _) | Expr::Div(_, _) | Expr::Mod(_, _) | Expr::Product(_) => Prec::Product,
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Sum(_) => Prec::Sum,
        Expr::Equation { .. }
        | Expr::Gte(_, _)
        | Expr::Gt(_, _)
        | Expr::Lte(_, _)
        | Expr::Lt(_, _) => Prec::Relation,
        Expr::Not(_) => Prec::Not,
        Expr::And(_, _) => Prec::And,
        Expr::Or(_, _) => Prec::Or,
        Expr::Implies(_, _) => Prec::Implies,
        Expr::ForAll { .. } | Expr::Exists { .. } => Prec::Quantifier,
    }
}

/// Write `expr`, parenthesising it if it binds more loosely than `context` requires.
fn write_expr(out: &mut String, expr: &Expr, symbols: &SymbolTable, context: Prec) {
    let needs_parens = precedence(expr) < context;
    if needs_parens {
        out.push('(');
    }
    write_bare(out, expr, symbols);
    if needs_parens {
        out.push(')');
    }
}

/// Write a `f(a, b, ...)` call. Arguments are at the loosest precedence: commas delimit them.
fn write_call(out: &mut String, name: &str, args: &[&Expr], symbols: &SymbolTable) {
    out.push_str(name);
    out.push('(');
    for (index, arg) in args.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        write_expr(out, arg, symbols, Prec::Quantifier);
    }
    out.push(')');
}

/// Write a left-associative binary operator.
fn write_binary(
    out: &mut String,
    lhs: &Expr,
    op: &str,
    rhs: &Expr,
    symbols: &SymbolTable,
    prec: Prec,
) {
    write_expr(out, lhs, symbols, prec);
    let _ = write!(out, " {op} ");
    // The right operand is written one level tighter so `a - (b - c)` keeps its parentheses
    // and `a - b - c` does not gain any.
    write_expr(out, rhs, symbols, next_tighter(prec));
}

/// The precedence one step above `prec`, used for the right operand of a left-associative
/// operator.
fn next_tighter(prec: Prec) -> Prec {
    match prec {
        Prec::Quantifier => Prec::Implies,
        Prec::Implies => Prec::Or,
        Prec::Or => Prec::And,
        Prec::And => Prec::Not,
        Prec::Not => Prec::Relation,
        Prec::Relation => Prec::Sum,
        Prec::Sum => Prec::Product,
        Prec::Product => Prec::Neg,
        Prec::Neg => Prec::Power,
        Prec::Power => Prec::Factorial,
        Prec::Factorial | Prec::Atom => Prec::Atom,
    }
}

fn write_bare(out: &mut String, expr: &Expr, symbols: &SymbolTable) {
    match expr {
        // ---- atoms ----
        Expr::Const(value) => write_rational(out, value),
        Expr::Var(symbol) => out.push_str(symbol_name(*symbol, symbols)),
        Expr::Pi => out.push_str("pi"),
        Expr::E => out.push('e'),

        // ---- function-call forms ----
        Expr::Sqrt(inner) => write_call(out, "sqrt", &[inner], symbols),
        Expr::Sin(inner) => write_call(out, "sin", &[inner], symbols),
        Expr::Cos(inner) => write_call(out, "cos", &[inner], symbols),
        Expr::Tan(inner) => write_call(out, "tan", &[inner], symbols),
        Expr::Arcsin(inner) => write_call(out, "arcsin", &[inner], symbols),
        Expr::Arccos(inner) => write_call(out, "arccos", &[inner], symbols),
        Expr::Arctan(inner) => write_call(out, "arctan", &[inner], symbols),
        Expr::Ln(inner) => write_call(out, "ln", &[inner], symbols),
        Expr::Exp(inner) => write_call(out, "exp", &[inner], symbols),
        Expr::Floor(inner) => write_call(out, "floor", &[inner], symbols),
        Expr::Ceiling(inner) => write_call(out, "ceil", &[inner], symbols),
        Expr::GCD(a, b) => write_call(out, "gcd", &[a, b], symbols),
        Expr::LCM(a, b) => write_call(out, "lcm", &[a, b], symbols),
        Expr::Binomial(n, k) => write_call(out, "binomial", &[n, k], symbols),

        // `|a|` is unambiguous and shorter than `abs(a)`, and the parser accepts `abs` back.
        Expr::Abs(inner) => {
            out.push('|');
            write_expr(out, inner, symbols, Prec::Quantifier);
            out.push('|');
        }

        // ---- arithmetic ----
        Expr::Add(a, b) => write_binary(out, a, "+", b, symbols, Prec::Sum),
        Expr::Sub(a, b) => write_binary(out, a, "-", b, symbols, Prec::Sum),
        Expr::Mul(a, b) => write_binary(out, a, "*", b, symbols, Prec::Product),
        Expr::Div(a, b) => write_binary(out, a, "/", b, symbols, Prec::Product),
        Expr::Mod(a, b) => write_binary(out, a, "%", b, symbols, Prec::Product),

        Expr::Neg(inner) => {
            out.push('-');
            // `-(a + b)` must keep its parentheses; `-x^2` must not gain any.
            write_expr(out, inner, symbols, Prec::Neg);
        }

        Expr::Pow(base, exponent) => {
            // Right-associative: `a^b^c` is `a^(b^c)`, so the base binds tighter than the
            // exponent.
            write_expr(out, base, symbols, Prec::Factorial);
            out.push('^');
            write_expr(out, exponent, symbols, Prec::Power);
        }

        Expr::Factorial(inner) => {
            write_expr(out, inner, symbols, Prec::Factorial);
            out.push('!');
        }

        // ---- canonical n-ary forms ----
        Expr::Sum(terms) => {
            if terms.is_empty() {
                out.push('0');
                return;
            }
            for (index, term) in terms.iter().enumerate() {
                if index > 0 {
                    out.push_str(" + ");
                }
                if term.coeff.is_integer() && term.coeff.numer() == 1 {
                    write_expr(out, &term.expr, symbols, Prec::Product);
                } else {
                    write_rational(out, &term.coeff);
                    out.push_str(" * ");
                    write_expr(out, &term.expr, symbols, Prec::Neg);
                }
            }
        }
        Expr::Product(factors) => {
            if factors.is_empty() {
                out.push('1');
                return;
            }
            for (index, factor) in factors.iter().enumerate() {
                if index > 0 {
                    out.push_str(" * ");
                }
                let unit_power = matches!(&factor.power, Expr::Const(c)
                    if c.is_integer() && c.numer() == 1);
                if unit_power {
                    write_expr(out, &factor.base, symbols, Prec::Neg);
                } else {
                    write_expr(out, &factor.base, symbols, Prec::Factorial);
                    out.push('^');
                    write_expr(out, &factor.power, symbols, Prec::Power);
                }
            }
        }

        // ---- calculus ----
        Expr::Derivative { expr: inner, var } => {
            out.push_str("diff(");
            write_expr(out, inner, symbols, Prec::Quantifier);
            let _ = write!(out, ", {})", symbol_name(*var, symbols));
        }
        Expr::Integral { expr: inner, var } => {
            out.push_str("int(");
            write_expr(out, inner, symbols, Prec::Quantifier);
            let _ = write!(out, ", {})", symbol_name(*var, symbols));
        }

        // ---- relations ----
        Expr::Equation { lhs, rhs } => write_binary(out, lhs, "=", rhs, symbols, Prec::Relation),
        Expr::Gte(a, b) => write_binary(out, a, ">=", b, symbols, Prec::Relation),
        Expr::Gt(a, b) => write_binary(out, a, ">", b, symbols, Prec::Relation),
        Expr::Lte(a, b) => write_binary(out, a, "<=", b, symbols, Prec::Relation),
        Expr::Lt(a, b) => write_binary(out, a, "<", b, symbols, Prec::Relation),

        // ---- summation and product notation ----
        Expr::Summation {
            var,
            from,
            to,
            body,
        } => write_indexed(out, "sum", *var, from, to, body, symbols),
        Expr::BigProduct {
            var,
            from,
            to,
            body,
        } => write_indexed(out, "prod", *var, from, to, body, symbols),

        // ---- logic ----
        Expr::Not(inner) => {
            out.push_str("not ");
            write_expr(out, inner, symbols, Prec::Not);
        }
        Expr::And(a, b) => write_binary(out, a, "and", b, symbols, Prec::And),
        Expr::Or(a, b) => write_binary(out, a, "or", b, symbols, Prec::Or),
        Expr::Implies(a, b) => write_binary(out, a, "=>", b, symbols, Prec::Implies),

        Expr::ForAll { var, domain, body } => {
            write_quantifier(out, "forall", *var, domain.as_deref(), body, symbols)
        }
        Expr::Exists { var, domain, body } => {
            write_quantifier(out, "exists", *var, domain.as_deref(), body, symbols)
        }
    }
}

/// `sum(k, from, to, body)` / `prod(k, from, to, body)`, matching the parser.
fn write_indexed(
    out: &mut String,
    name: &str,
    var: Symbol,
    from: &Expr,
    to: &Expr,
    body: &Expr,
    symbols: &SymbolTable,
) {
    let _ = write!(out, "{}({}, ", name, symbol_name(var, symbols));
    write_expr(out, from, symbols, Prec::Quantifier);
    out.push_str(", ");
    write_expr(out, to, symbols, Prec::Quantifier);
    out.push_str(", ");
    write_expr(out, body, symbols, Prec::Quantifier);
    out.push(')');
}

/// `forall x. body` or `forall x in D. body`.
fn write_quantifier(
    out: &mut String,
    name: &str,
    var: Symbol,
    domain: Option<&Expr>,
    body: &Expr,
    symbols: &SymbolTable,
) {
    let _ = write!(out, "{} {}", name, symbol_name(var, symbols));
    if let Some(domain) = domain {
        out.push_str(" in ");
        write_expr(out, domain, symbols, Prec::Relation);
    }
    out.push_str(". ");
    write_expr(out, body, symbols, Prec::Quantifier);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Factor, Term};

    struct Ctx {
        symbols: SymbolTable,
        x: Symbol,
        y: Symbol,
        k: Symbol,
    }

    fn ctx() -> Ctx {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");
        let y = symbols.intern("y");
        let k = symbols.intern("k");
        Ctx { symbols, x, y, k }
    }

    fn int(v: i64) -> Expr {
        Expr::int(v)
    }
    fn b(e: Expr) -> Box<Expr> {
        Box::new(e)
    }

    fn render(expr: &Expr, c: &Ctx) -> String {
        format_expr(expr, &c.symbols)
    }

    #[test]
    fn atoms_render_plainly() {
        let c = ctx();
        assert_eq!(render(&int(3), &c), "3");
        assert_eq!(render(&int(-7), &c), "-7");
        assert_eq!(render(&Expr::Const(Rational::new(1, 2)), &c), "1/2");
        assert_eq!(render(&Expr::Var(c.x), &c), "x");
        assert_eq!(render(&Expr::Pi, &c), "pi");
        assert_eq!(render(&Expr::E, &c), "e");
    }

    #[test]
    fn arithmetic_keeps_only_the_parentheses_it_needs() {
        let c = ctx();
        let x = || Expr::Var(c.x);
        let y = || Expr::Var(c.y);

        // Multiplication binds tighter, so no parentheses here.
        assert_eq!(
            render(&Expr::Add(b(x()), b(Expr::Mul(b(y()), b(int(2))))), &c),
            "x + y * 2"
        );
        // ... and they are required here.
        assert_eq!(
            render(&Expr::Mul(b(Expr::Add(b(x()), b(y()))), b(int(2))), &c),
            "(x + y) * 2"
        );
        // Left-associative chains stay flat.
        assert_eq!(
            render(&Expr::Sub(b(Expr::Sub(b(x()), b(y()))), b(int(1))), &c),
            "x - y - 1"
        );
        // A right-nested subtraction is a different tree and must show it.
        assert_eq!(
            render(&Expr::Sub(b(x()), b(Expr::Sub(b(y()), b(int(1))))), &c),
            "x - (y - 1)"
        );
        assert_eq!(
            render(&Expr::Div(b(x()), b(Expr::Mul(b(y()), b(int(2))))), &c),
            "x / (y * 2)"
        );
    }

    #[test]
    fn powers_are_right_associative_and_bind_tightest() {
        let c = ctx();
        let x = || Expr::Var(c.x);

        assert_eq!(render(&Expr::Pow(b(x()), b(int(2))), &c), "x^2");
        // The plan's example: an additive exponent must be parenthesised.
        assert_eq!(
            render(
                &Expr::Pow(b(x()), b(Expr::Add(b(Expr::Var(c.y)), b(int(1))))),
                &c
            ),
            "x^(y + 1)"
        );
        // A compound base must be too.
        assert_eq!(
            render(&Expr::Pow(b(Expr::Add(b(x()), b(int(1)))), b(int(2))), &c),
            "(x + 1)^2"
        );
        // `a^b^c` means `a^(b^c)`; rendering it flat is correct and re-parses the same way.
        assert_eq!(
            render(&Expr::Pow(b(x()), b(Expr::Pow(b(int(2)), b(int(3))))), &c),
            "x^2^3"
        );
        // `(a^b)^c` is a different tree.
        assert_eq!(
            render(&Expr::Pow(b(Expr::Pow(b(x()), b(int(2)))), b(int(3))), &c),
            "(x^2)^3"
        );
        // A power binds tighter than multiplication.
        assert_eq!(
            render(&Expr::Mul(b(int(2)), b(Expr::Pow(b(x()), b(int(2))))), &c),
            "2 * x^2"
        );
    }

    #[test]
    fn negation_parenthesises_only_looser_operands() {
        let c = ctx();
        let x = || Expr::Var(c.x);

        assert_eq!(render(&Expr::Neg(b(x())), &c), "-x");
        assert_eq!(
            render(&Expr::Neg(b(Expr::Add(b(x()), b(int(1))))), &c),
            "-(x + 1)"
        );
        // Power binds tighter than unary minus.
        assert_eq!(
            render(&Expr::Neg(b(Expr::Pow(b(x()), b(int(2))))), &c),
            "-x^2"
        );
        assert_eq!(
            render(&Expr::Add(b(x()), b(Expr::Neg(b(Expr::Var(c.y))))), &c),
            "x + -y"
        );
    }

    #[test]
    fn factorial_binds_tighter_than_power_and_wraps_compounds() {
        let c = ctx();
        assert_eq!(render(&Expr::Factorial(b(int(5))), &c), "5!");
        assert_eq!(render(&Expr::Factorial(b(Expr::Var(c.x))), &c), "x!");
        assert_eq!(
            render(
                &Expr::Factorial(b(Expr::Sub(b(Expr::Var(c.x)), b(int(1))))),
                &c
            ),
            "(x - 1)!"
        );
    }

    #[test]
    fn unary_and_binary_functions_use_call_syntax() {
        let c = ctx();
        let x = || Expr::Var(c.x);

        assert_eq!(render(&Expr::Sin(b(x())), &c), "sin(x)");
        assert_eq!(render(&Expr::Cos(b(x())), &c), "cos(x)");
        assert_eq!(render(&Expr::Tan(b(x())), &c), "tan(x)");
        assert_eq!(render(&Expr::Arcsin(b(x())), &c), "arcsin(x)");
        assert_eq!(render(&Expr::Arccos(b(x())), &c), "arccos(x)");
        assert_eq!(render(&Expr::Arctan(b(x())), &c), "arctan(x)");
        assert_eq!(render(&Expr::Ln(b(x())), &c), "ln(x)");
        assert_eq!(render(&Expr::Exp(b(x())), &c), "exp(x)");
        assert_eq!(render(&Expr::Sqrt(b(int(25))), &c), "sqrt(25)");
        assert_eq!(render(&Expr::Floor(b(x())), &c), "floor(x)");
        assert_eq!(render(&Expr::Ceiling(b(x())), &c), "ceil(x)");
        assert_eq!(
            render(&Expr::GCD(b(int(12)), b(int(18))), &c),
            "gcd(12, 18)"
        );
        assert_eq!(render(&Expr::LCM(b(int(4)), b(int(6))), &c), "lcm(4, 6)");
        assert_eq!(
            render(&Expr::Binomial(b(int(5)), b(int(2))), &c),
            "binomial(5, 2)"
        );
        assert_eq!(render(&Expr::Mod(b(int(17)), b(int(5))), &c), "17 % 5");

        // A call is an atom, so it needs no parentheses inside a product.
        assert_eq!(
            render(&Expr::Mul(b(int(2)), b(Expr::Sin(b(x())))), &c),
            "2 * sin(x)"
        );
        // Its argument is at the loosest precedence.
        assert_eq!(
            render(&Expr::Sin(b(Expr::Add(b(x()), b(int(1))))), &c),
            "sin(x + 1)"
        );
    }

    #[test]
    fn absolute_value_uses_bars() {
        let c = ctx();
        assert_eq!(render(&Expr::Abs(b(Expr::Var(c.x))), &c), "|x|");
        assert_eq!(
            render(
                &Expr::Abs(b(Expr::Sub(b(Expr::Var(c.x)), b(Expr::Var(c.y))))),
                &c
            ),
            "|x - y|"
        );
    }

    #[test]
    fn calculus_matches_the_parser_syntax() {
        let c = ctx();
        let x = || Expr::Var(c.x);

        assert_eq!(
            render(
                &Expr::Derivative {
                    expr: b(Expr::Sin(b(x()))),
                    var: c.x
                },
                &c
            ),
            "diff(sin(x), x)"
        );
        assert_eq!(
            render(
                &Expr::Integral {
                    expr: b(Expr::Pow(b(x()), b(int(2)))),
                    var: c.x
                },
                &c
            ),
            "int(x^2, x)"
        );
        // Nested derivatives stay readable.
        assert_eq!(
            render(
                &Expr::Derivative {
                    expr: b(Expr::Derivative {
                        expr: b(Expr::Pow(b(x()), b(int(3)))),
                        var: c.x
                    }),
                    var: c.x
                },
                &c
            ),
            "diff(diff(x^3, x), x)"
        );
    }

    #[test]
    fn relations_render_infix() {
        let c = ctx();
        let x = || Expr::Var(c.x);

        assert_eq!(
            render(
                &Expr::Equation {
                    lhs: b(Expr::Add(b(x()), b(int(3)))),
                    rhs: b(int(7))
                },
                &c
            ),
            "x + 3 = 7"
        );
        assert_eq!(
            render(&Expr::Gte(b(Expr::Pow(b(x()), b(int(2)))), b(int(0))), &c),
            "x^2 >= 0"
        );
        assert_eq!(render(&Expr::Gt(b(x()), b(int(0))), &c), "x > 0");
        assert_eq!(render(&Expr::Lte(b(x()), b(Expr::Var(c.y))), &c), "x <= y");
        assert_eq!(render(&Expr::Lt(b(x()), b(Expr::Var(c.y))), &c), "x < y");
    }

    #[test]
    fn summation_and_product_match_the_parser_syntax() {
        let c = ctx();
        assert_eq!(
            render(
                &Expr::Summation {
                    var: c.k,
                    from: b(int(1)),
                    to: b(Expr::Var(c.x)),
                    body: b(Expr::Var(c.k)),
                },
                &c
            ),
            "sum(k, 1, x, k)"
        );
        assert_eq!(
            render(
                &Expr::BigProduct {
                    var: c.k,
                    from: b(int(1)),
                    to: b(Expr::Var(c.x)),
                    body: b(Expr::Pow(b(Expr::Var(c.k)), b(int(2)))),
                },
                &c
            ),
            "prod(k, 1, x, k^2)"
        );
    }

    #[test]
    fn logic_and_quantifiers_render_with_binding() {
        let c = ctx();
        let x = || Expr::Var(c.x);
        let positive = || Expr::Gt(b(x()), b(int(0)));

        assert_eq!(render(&Expr::Not(b(positive())), &c), "not x > 0");
        assert_eq!(
            render(&Expr::And(b(positive()), b(positive())), &c),
            "x > 0 and x > 0"
        );
        assert_eq!(
            render(&Expr::Or(b(positive()), b(positive())), &c),
            "x > 0 or x > 0"
        );
        assert_eq!(
            render(&Expr::Implies(b(positive()), b(positive())), &c),
            "x > 0 => x > 0"
        );
        // `and` binds tighter than `or`, so this nesting needs no parentheses...
        assert_eq!(
            render(
                &Expr::Or(b(Expr::And(b(positive()), b(positive()))), b(positive())),
                &c
            ),
            "x > 0 and x > 0 or x > 0"
        );
        // ... and the other nesting does.
        assert_eq!(
            render(
                &Expr::And(b(Expr::Or(b(positive()), b(positive()))), b(positive())),
                &c
            ),
            "(x > 0 or x > 0) and x > 0"
        );

        assert_eq!(
            render(
                &Expr::ForAll {
                    var: c.x,
                    domain: None,
                    body: b(Expr::Gte(b(Expr::Pow(b(x()), b(int(2)))), b(int(0)))),
                },
                &c
            ),
            "forall x. x^2 >= 0"
        );
        assert_eq!(
            render(
                &Expr::Exists {
                    var: c.x,
                    domain: Some(b(Expr::Gt(b(x()), b(int(0))))),
                    body: b(Expr::Equation {
                        lhs: b(Expr::Pow(b(x()), b(int(2)))),
                        rhs: b(int(4))
                    }),
                },
                &c
            ),
            "exists x in x > 0. x^2 = 4"
        );
    }

    #[test]
    fn canonical_n_ary_forms_render() {
        let c = ctx();

        assert_eq!(render(&Expr::Sum(vec![]), &c), "0");
        assert_eq!(render(&Expr::Product(vec![]), &c), "1");

        let sum = Expr::Sum(vec![
            Term {
                coeff: Rational::new(2, 1),
                expr: Expr::Var(c.x),
            },
            Term {
                coeff: Rational::new(1, 1),
                expr: Expr::Var(c.y),
            },
        ]);
        assert_eq!(render(&sum, &c), "2 * x + y");

        let product = Expr::Product(vec![
            Factor {
                base: Expr::Var(c.x),
                power: int(2),
            },
            Factor {
                base: Expr::Var(c.y),
                power: int(1),
            },
        ]);
        assert_eq!(render(&product, &c), "x^2 * y");
    }

    #[test]
    fn a_symbol_from_another_table_renders_instead_of_panicking() {
        // The UI formats results in the worker that owns the solver's table, but a mismatch
        // must degrade to a placeholder rather than take the process down.
        let c = ctx();
        let mut other = SymbolTable::new();
        for name in ["a", "b", "c", "d", "e", "f", "g"] {
            other.intern(name);
        }
        let foreign = other.intern("nowhere");

        let rendered = format_expr(&Expr::Var(foreign), &c.symbols);
        assert!(
            rendered == UNKNOWN_SYMBOL || !rendered.is_empty(),
            "unknown symbols must still render"
        );
    }

    #[test]
    fn every_variant_renders_without_panicking() {
        let c = ctx();
        let one = || b(int(1));
        let two = || b(int(2));

        let all = vec![
            Expr::Const(Rational::new(3, 4)),
            Expr::Var(c.x),
            Expr::Pi,
            Expr::E,
            Expr::Neg(one()),
            Expr::Sqrt(one()),
            Expr::Sin(one()),
            Expr::Cos(one()),
            Expr::Tan(one()),
            Expr::Arcsin(one()),
            Expr::Arccos(one()),
            Expr::Arctan(one()),
            Expr::Ln(one()),
            Expr::Exp(one()),
            Expr::Abs(one()),
            Expr::Add(one(), two()),
            Expr::Sub(one(), two()),
            Expr::Mul(one(), two()),
            Expr::Div(one(), two()),
            Expr::Pow(one(), two()),
            Expr::Sum(vec![Term {
                coeff: Rational::new(2, 1),
                expr: Expr::Var(c.x),
            }]),
            Expr::Product(vec![Factor {
                base: Expr::Var(c.x),
                power: int(2),
            }]),
            Expr::Derivative {
                expr: one(),
                var: c.x,
            },
            Expr::Integral {
                expr: one(),
                var: c.x,
            },
            Expr::Equation {
                lhs: one(),
                rhs: two(),
            },
            Expr::Gte(one(), two()),
            Expr::Gt(one(), two()),
            Expr::Lte(one(), two()),
            Expr::Lt(one(), two()),
            Expr::GCD(one(), two()),
            Expr::LCM(one(), two()),
            Expr::Mod(one(), two()),
            Expr::Floor(one()),
            Expr::Ceiling(one()),
            Expr::Factorial(one()),
            Expr::Binomial(two(), one()),
            Expr::Summation {
                var: c.k,
                from: one(),
                to: two(),
                body: b(Expr::Var(c.k)),
            },
            Expr::BigProduct {
                var: c.k,
                from: one(),
                to: two(),
                body: b(Expr::Var(c.k)),
            },
            Expr::ForAll {
                var: c.x,
                domain: None,
                body: b(Expr::Gt(b(Expr::Var(c.x)), one())),
            },
            Expr::Exists {
                var: c.x,
                domain: Some(one()),
                body: b(Expr::Gt(b(Expr::Var(c.x)), one())),
            },
            Expr::And(b(Expr::Gt(one(), two())), b(Expr::Pi)),
            Expr::Or(b(Expr::Gt(one(), two())), b(Expr::Pi)),
            Expr::Not(b(Expr::Pi)),
            Expr::Implies(b(Expr::Gt(one(), two())), b(Expr::Pi)),
        ];

        for expr in all {
            let rendered = format_expr(&expr, &c.symbols);
            assert!(!rendered.is_empty(), "{expr:?} rendered as an empty string");
            assert!(
                !rendered.contains("Box"),
                "{expr:?} leaked Rust syntax: {rendered}"
            );
        }
    }

    #[test]
    fn truncation_counts_characters_not_bytes() {
        // A byte-index slice here would split a multi-byte scalar and panic.
        let text = "ααααααααββββββββ";
        let cut = truncate_chars(text, 5);
        assert_eq!(cut.chars().count(), 5);
        assert!(cut.ends_with('…'));

        assert_eq!(truncate_chars("short", 20), "short");
        assert_eq!(truncate_chars("", 0), "");
    }

    #[test]
    fn rendered_output_parses_back_to_the_same_expression() {
        // The strongest statement the formatter can make: parentheses preserve the tree.
        use crate::parse::Parser;

        let mut c = ctx();
        let x = Expr::Var(c.x);
        let y = Expr::Var(c.y);

        let cases = vec![
            Expr::Mul(b(Expr::Add(b(x.clone()), b(y.clone()))), b(int(2))),
            Expr::Sub(b(x.clone()), b(Expr::Sub(b(y.clone()), b(int(1))))),
            Expr::Pow(b(x.clone()), b(Expr::Add(b(y.clone()), b(int(1))))),
            Expr::Pow(b(Expr::Pow(b(x.clone()), b(int(2)))), b(int(3))),
            Expr::Div(b(x.clone()), b(Expr::Mul(b(y.clone()), b(int(2))))),
            Expr::Neg(b(Expr::Add(b(x.clone()), b(int(1))))),
            Expr::Sin(b(Expr::Add(b(x.clone()), b(int(1))))),
            Expr::Derivative {
                expr: b(Expr::Pow(b(x.clone()), b(int(3)))),
                var: c.x,
            },
            Expr::Integral {
                expr: b(Expr::Pow(b(x.clone()), b(int(2)))),
                var: c.x,
            },
            Expr::Equation {
                lhs: b(Expr::Add(b(x.clone()), b(int(3)))),
                rhs: b(int(7)),
            },
        ];

        for expr in cases {
            let rendered = format_expr(&expr, &c.symbols);
            let mut parser = Parser::new(&mut c.symbols);
            let reparsed = parser
                .parse(&rendered)
                .unwrap_or_else(|e| panic!("{rendered:?} did not parse back: {e}"));
            assert_eq!(
                reparsed, expr,
                "{rendered:?} parsed back to a different tree"
            );
        }
    }
}

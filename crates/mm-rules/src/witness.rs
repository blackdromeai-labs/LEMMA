// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! A fixed corpus of expressions for exercising the rule registry.
//!
//! The registry reports 572 rules, but that is a count of constructors, not of rules that do
//! anything. Module comments claim things like "28 working, 56 need implementation"; those
//! numbers were written by hand and are not checked by anything.
//!
//! This corpus exists so those claims can be replaced by measurement: run every rule against
//! every expression here and record what happened. A rule that never becomes applicable, or
//! that only ever returns its input, is visible immediately.
//!
//! The corpus is deterministic and hand-written rather than randomly generated, so a census
//! taken today is comparable with one taken later. It is **not** exhaustive: a rule with no
//! witness here is "not reached by this corpus", which is weaker than "unreachable". Callers
//! must not report it as the latter.

use mm_core::{Expr, Rational, Symbol, SymbolTable};

/// Symbols the corpus uses, so callers can build comparable expressions.
pub struct WitnessSymbols {
    /// Symbol table the corpus was interned into.
    pub symbols: SymbolTable,
    /// `x`, the usual variable of differentiation and integration.
    pub x: Symbol,
    /// `y`.
    pub y: Symbol,
    /// `z`.
    pub z: Symbol,
    /// `n`, used where an integer index reads better.
    pub n: Symbol,
    /// `k`, the usual summation index.
    pub k: Symbol,
}

impl Default for WitnessSymbols {
    fn default() -> Self {
        Self::new()
    }
}

impl WitnessSymbols {
    /// Intern the corpus symbols.
    pub fn new() -> Self {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");
        let y = symbols.intern("y");
        let z = symbols.intern("z");
        let n = symbols.intern("n");
        let k = symbols.intern("k");
        Self {
            symbols,
            x,
            y,
            z,
            n,
            k,
        }
    }
}

fn int(v: i64) -> Expr {
    Expr::int(v)
}

fn rat(num: i64, den: i64) -> Expr {
    Expr::Const(Rational::new(num, den))
}

fn add(a: Expr, b: Expr) -> Expr {
    Expr::Add(Box::new(a), Box::new(b))
}

fn sub(a: Expr, b: Expr) -> Expr {
    Expr::Sub(Box::new(a), Box::new(b))
}

fn mul(a: Expr, b: Expr) -> Expr {
    Expr::Mul(Box::new(a), Box::new(b))
}

fn div(a: Expr, b: Expr) -> Expr {
    Expr::Div(Box::new(a), Box::new(b))
}

fn pow(a: Expr, b: Expr) -> Expr {
    Expr::Pow(Box::new(a), Box::new(b))
}

fn neg(a: Expr) -> Expr {
    Expr::Neg(Box::new(a))
}

fn eq(a: Expr, b: Expr) -> Expr {
    Expr::Equation {
        lhs: Box::new(a),
        rhs: Box::new(b),
    }
}

/// Build the witness corpus.
///
/// Roughly grouped by the shapes different rule modules look for. Every expression is
/// mathematically meaningful; nothing here is a malformed probe.
pub fn corpus(s: &WitnessSymbols) -> Vec<Expr> {
    let x = Expr::Var(s.x);
    let y = Expr::Var(s.y);
    let z = Expr::Var(s.z);
    let n = Expr::Var(s.n);

    let sin = |e: Expr| Expr::Sin(Box::new(e));
    let cos = |e: Expr| Expr::Cos(Box::new(e));
    let tan = |e: Expr| Expr::Tan(Box::new(e));
    let ln = |e: Expr| Expr::Ln(Box::new(e));
    let exp = |e: Expr| Expr::Exp(Box::new(e));
    let sqrt = |e: Expr| Expr::Sqrt(Box::new(e));
    let abs = |e: Expr| Expr::Abs(Box::new(e));
    let d = |e: Expr| Expr::Derivative {
        expr: Box::new(e),
        var: s.x,
    };
    let integral = |e: Expr| Expr::Integral {
        expr: Box::new(e),
        var: s.x,
    };

    let mut out = Vec::new();

    // ---- atoms ----
    out.extend([
        int(0),
        int(1),
        int(2),
        int(5),
        int(12),
        int(100),
        int(-3),
        rat(1, 2),
        rat(3, 4),
        Expr::Pi,
        Expr::E,
        x.clone(),
        y.clone(),
        n.clone(),
    ]);

    // ---- arithmetic and identities ----
    out.extend([
        add(int(2), int(3)),
        sub(int(10), int(4)),
        mul(int(7), int(8)),
        div(int(12), int(4)),
        pow(int(2), int(3)),
        add(x.clone(), int(0)),
        add(int(0), x.clone()),
        mul(x.clone(), int(1)),
        mul(int(1), x.clone()),
        mul(x.clone(), int(0)),
        sub(x.clone(), x.clone()),
        div(x.clone(), x.clone()),
        pow(x.clone(), int(0)),
        pow(x.clone(), int(1)),
        neg(x.clone()),
        neg(neg(x.clone())),
        neg(int(4)),
    ]);

    // ---- polynomial shapes ----
    out.extend([
        add(x.clone(), y.clone()),
        mul(x.clone(), y.clone()),
        pow(x.clone(), int(2)),
        pow(x.clone(), int(3)),
        pow(x.clone(), int(4)),
        mul(pow(x.clone(), int(2)), pow(x.clone(), int(3))),
        div(pow(x.clone(), int(5)), pow(x.clone(), int(2))),
        pow(pow(x.clone(), int(2)), int(3)),
        pow(add(x.clone(), y.clone()), int(2)),
        pow(sub(x.clone(), y.clone()), int(2)),
        pow(add(x.clone(), y.clone()), int(3)),
        sub(pow(x.clone(), int(2)), pow(y.clone(), int(2))),
        add(pow(x.clone(), int(3)), pow(y.clone(), int(3))),
        sub(pow(x.clone(), int(3)), pow(y.clone(), int(3))),
        add(
            add(
                pow(x.clone(), int(2)),
                mul(int(2), mul(x.clone(), y.clone())),
            ),
            pow(y.clone(), int(2)),
        ),
        mul(int(2), add(x.clone(), y.clone())),
        add(mul(int(2), x.clone()), mul(int(3), x.clone())),
        add(mul(x.clone(), y.clone()), mul(x.clone(), z.clone())),
        mul(add(x.clone(), int(1)), add(x.clone(), int(1))),
        mul(add(x.clone(), int(1)), sub(x.clone(), int(1))),
        add(add(pow(x.clone(), int(2)), mul(int(5), x.clone())), int(6)),
    ]);

    // ---- radicals, logs, exponentials ----
    out.extend([
        sqrt(int(25)),
        sqrt(x.clone()),
        sqrt(mul(x.clone(), y.clone())),
        sqrt(div(x.clone(), y.clone())),
        sqrt(pow(x.clone(), int(2))),
        pow(x.clone(), rat(1, 2)),
        ln(int(1)),
        ln(Expr::E),
        ln(x.clone()),
        ln(mul(x.clone(), y.clone())),
        ln(div(x.clone(), y.clone())),
        ln(pow(x.clone(), int(2))),
        exp(int(0)),
        exp(x.clone()),
        exp(add(x.clone(), y.clone())),
        exp(ln(x.clone())),
        ln(exp(x.clone())),
        pow(Expr::E, x.clone()),
        pow(int(2), x.clone()),
    ]);

    // ---- trigonometry ----
    out.extend([
        sin(int(0)),
        cos(int(0)),
        tan(int(0)),
        sin(x.clone()),
        cos(x.clone()),
        tan(x.clone()),
        sin(Expr::Pi),
        cos(Expr::Pi),
        add(pow(sin(x.clone()), int(2)), pow(cos(x.clone()), int(2))),
        sub(
            add(pow(sin(x.clone()), int(2)), pow(cos(x.clone()), int(2))),
            int(1),
        ),
        sin(add(x.clone(), y.clone())),
        cos(add(x.clone(), y.clone())),
        sin(mul(int(2), x.clone())),
        cos(mul(int(2), x.clone())),
        sin(mul(int(3), x.clone())),
        div(sin(x.clone()), cos(x.clone())),
        div(int(1), cos(x.clone())),
        div(int(1), sin(x.clone())),
        div(int(1), tan(x.clone())),
        mul(sin(x.clone()), cos(x.clone())),
        Expr::Arcsin(Box::new(x.clone())),
        Expr::Arccos(Box::new(x.clone())),
        Expr::Arctan(Box::new(x.clone())),
        sin(Expr::Arcsin(Box::new(x.clone()))),
    ]);

    // ---- derivatives ----
    out.extend([
        d(int(5)),
        d(x.clone()),
        d(pow(x.clone(), int(2))),
        d(pow(x.clone(), int(3))),
        d(mul(int(2), x.clone())),
        d(add(x.clone(), int(5))),
        d(add(pow(x.clone(), int(2)), pow(x.clone(), int(3)))),
        d(sub(pow(x.clone(), int(2)), x.clone())),
        d(mul(x.clone(), sin(x.clone()))),
        d(div(sin(x.clone()), x.clone())),
        d(sin(x.clone())),
        d(cos(x.clone())),
        d(tan(x.clone())),
        d(exp(x.clone())),
        d(ln(x.clone())),
        d(sqrt(x.clone())),
        d(abs(x.clone())),
        d(sin(pow(x.clone(), int(2)))),
        d(exp(mul(int(2), x.clone()))),
        d(ln(add(x.clone(), int(1)))),
        d(pow(int(2), x.clone())),
        d(pow(add(x.clone(), int(1)), int(3))),
        d(Expr::Arcsin(Box::new(x.clone()))),
        d(Expr::Arctan(Box::new(x.clone()))),
        d(d(pow(x.clone(), int(3)))),
    ]);

    // ---- integrals ----
    out.extend([
        integral(int(1)),
        integral(int(5)),
        integral(x.clone()),
        integral(pow(x.clone(), int(2))),
        integral(pow(x.clone(), int(-1))),
        integral(div(int(1), x.clone())),
        integral(add(x.clone(), int(1))),
        integral(sub(pow(x.clone(), int(2)), x.clone())),
        integral(mul(int(3), pow(x.clone(), int(2)))),
        integral(sin(x.clone())),
        integral(cos(x.clone())),
        integral(tan(x.clone())),
        integral(exp(x.clone())),
        integral(ln(x.clone())),
        integral(mul(x.clone(), exp(x.clone()))),
        integral(mul(x.clone(), sin(x.clone()))),
        integral(pow(sin(x.clone()), int(2))),
        integral(pow(cos(x.clone()), int(2))),
        integral(div(int(1), add(pow(x.clone(), int(2)), int(1)))),
        integral(sqrt(sub(int(1), pow(x.clone(), int(2))))),
        integral(div(x.clone(), add(pow(x.clone(), int(2)), int(1)))),
    ]);

    // ---- equations and inequalities ----
    out.extend([
        eq(add(x.clone(), int(3)), int(7)),
        eq(sub(x.clone(), int(3)), int(7)),
        eq(mul(int(2), x.clone()), int(10)),
        eq(div(x.clone(), int(2)), int(5)),
        eq(add(mul(int(3), x.clone()), int(5)), int(17)),
        eq(x.clone(), int(4)),
        eq(
            add(add(pow(x.clone(), int(2)), mul(int(5), x.clone())), int(6)),
            int(0),
        ),
        eq(mul(x.clone(), y.clone()), int(1)),
        Expr::Gte(Box::new(pow(x.clone(), int(2))), Box::new(int(0))),
        Expr::Gte(
            Box::new(add(pow(x.clone(), int(2)), pow(y.clone(), int(2)))),
            Box::new(mul(int(2), mul(x.clone(), y.clone()))),
        ),
        Expr::Gt(Box::new(x.clone()), Box::new(int(0))),
        Expr::Lte(Box::new(x.clone()), Box::new(y.clone())),
        Expr::Lt(Box::new(x.clone()), Box::new(y.clone())),
        Expr::Gte(
            Box::new(div(add(x.clone(), y.clone()), int(2))),
            Box::new(sqrt(mul(x.clone(), y.clone()))),
        ),
        abs(x.clone()),
        abs(neg(x.clone())),
        abs(mul(x.clone(), y.clone())),
        abs(div(x.clone(), y.clone())),
        abs(add(x.clone(), y.clone())),
        abs(pow(x.clone(), int(2))),
        abs(abs(x.clone())),
    ]);

    // ---- number theory and combinatorics ----
    out.extend([
        Expr::GCD(Box::new(int(12)), Box::new(int(18))),
        Expr::GCD(Box::new(x.clone()), Box::new(y.clone())),
        Expr::LCM(Box::new(int(4)), Box::new(int(6))),
        Expr::LCM(Box::new(x.clone()), Box::new(y.clone())),
        Expr::Mod(Box::new(int(17)), Box::new(int(5))),
        Expr::Mod(Box::new(pow(x.clone(), int(2))), Box::new(int(3))),
        Expr::Floor(Box::new(rat(7, 2))),
        Expr::Floor(Box::new(x.clone())),
        Expr::Ceiling(Box::new(rat(7, 2))),
        Expr::Ceiling(Box::new(x.clone())),
        sub(
            Expr::Ceiling(Box::new(x.clone())),
            Expr::Floor(Box::new(x.clone())),
        ),
        Expr::Factorial(Box::new(int(0))),
        Expr::Factorial(Box::new(int(1))),
        Expr::Factorial(Box::new(int(5))),
        Expr::Factorial(Box::new(n.clone())),
        div(
            Expr::Factorial(Box::new(n.clone())),
            Expr::Factorial(Box::new(sub(n.clone(), int(2)))),
        ),
        Expr::Binomial(Box::new(n.clone()), Box::new(int(0))),
        Expr::Binomial(Box::new(n.clone()), Box::new(n.clone())),
        Expr::Binomial(Box::new(n.clone()), Box::new(int(1))),
        Expr::Binomial(Box::new(int(5)), Box::new(int(2))),
        Expr::Binomial(Box::new(n.clone()), Box::new(Expr::Var(s.k))),
        pow(int(2), n.clone()),
    ]);

    // ---- summation, product, quantifiers ----
    out.extend([
        Expr::Summation {
            var: s.k,
            from: Box::new(int(1)),
            to: Box::new(n.clone()),
            body: Box::new(Expr::Var(s.k)),
        },
        Expr::Summation {
            var: s.k,
            from: Box::new(int(1)),
            to: Box::new(n.clone()),
            body: Box::new(pow(Expr::Var(s.k), int(2))),
        },
        Expr::Summation {
            var: s.k,
            from: Box::new(int(0)),
            to: Box::new(n.clone()),
            body: Box::new(Expr::Binomial(
                Box::new(n.clone()),
                Box::new(Expr::Var(s.k)),
            )),
        },
        Expr::BigProduct {
            var: s.k,
            from: Box::new(int(1)),
            to: Box::new(n.clone()),
            body: Box::new(Expr::Var(s.k)),
        },
        Expr::ForAll {
            var: s.x,
            domain: None,
            body: Box::new(Expr::Gte(
                Box::new(pow(x.clone(), int(2))),
                Box::new(int(0)),
            )),
        },
        Expr::Exists {
            var: s.x,
            domain: None,
            body: Box::new(eq(pow(x.clone(), int(2)), int(4))),
        },
        Expr::And(
            Box::new(Expr::Gt(Box::new(x.clone()), Box::new(int(0)))),
            Box::new(Expr::Gt(Box::new(y.clone()), Box::new(int(0)))),
        ),
        Expr::Or(
            Box::new(eq(x.clone(), int(1))),
            Box::new(eq(x.clone(), int(2))),
        ),
        Expr::Not(Box::new(eq(x.clone(), int(0)))),
        Expr::Implies(
            Box::new(Expr::Gt(Box::new(x.clone()), Box::new(int(0)))),
            Box::new(Expr::Gt(Box::new(pow(x.clone(), int(2))), Box::new(int(0)))),
        ),
    ]);

    // ---- fractions, radicals in denominators, reciprocal powers ----
    out.extend([
        add(div(x.clone(), y.clone()), div(z.clone(), n.clone())),
        add(div(int(1), int(2)), div(int(1), int(3))),
        sub(div(x.clone(), y.clone()), div(z.clone(), n.clone())),
        mul(div(x.clone(), y.clone()), div(z.clone(), n.clone())),
        div(div(x.clone(), y.clone()), div(z.clone(), n.clone())),
        div(add(x.clone(), y.clone()), z.clone()),
        pow(x.clone(), int(-1)),
        pow(x.clone(), int(-2)),
        pow(x.clone(), rat(2, 3)),
        div(int(1), pow(x.clone(), int(2))),
        div(int(1), sqrt(int(2))),
        div(int(1), add(sqrt(int(2)), int(1))),
        mul(add(sqrt(int(2)), int(1)), sub(sqrt(int(2)), int(1))),
        eq(div(x.clone(), y.clone()), div(z.clone(), n.clone())),
    ]);

    // ---- exponential and logarithm combining forms ----
    out.extend([
        exp(int(1)),
        mul(exp(x.clone()), exp(y.clone())),
        div(exp(x.clone()), exp(y.clone())),
        pow(exp(x.clone()), int(2)),
        mul(pow(int(2), x.clone()), pow(int(2), y.clone())),
        add(ln(x.clone()), ln(y.clone())),
        sub(ln(x.clone()), ln(y.clone())),
        mul(int(2), ln(x.clone())),
        div(ln(x.clone()), ln(int(2))),
    ]);

    // ---- cubes, quadratics, and symmetric-function shapes ----
    out.extend([
        add(pow(x.clone(), int(3)), int(8)),
        sub(pow(x.clone(), int(3)), int(8)),
        sub(pow(x.clone(), int(2)), int(9)),
        add(sub(pow(x.clone(), int(2)), mul(int(4), x.clone())), int(4)),
        eq(
            add(sub(pow(x.clone(), int(2)), mul(int(5), x.clone())), int(6)),
            int(0),
        ),
        add(add(x.clone(), y.clone()), z.clone()),
        mul(mul(x.clone(), y.clone()), z.clone()),
    ]);

    // ---- inequality shapes ----
    out.extend([
        Expr::Gte(
            Box::new(add(x.clone(), y.clone())),
            Box::new(mul(int(2), sqrt(mul(x.clone(), y.clone())))),
        ),
        Expr::Gte(
            Box::new(abs(add(x.clone(), y.clone()))),
            Box::new(add(abs(x.clone()), abs(y.clone()))),
        ),
        abs(sub(x.clone(), y.clone())),
        Expr::Gte(
            Box::new(add(div(x.clone(), y.clone()), div(y.clone(), x.clone()))),
            Box::new(int(2)),
        ),
    ]);

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_non_trivial_and_deterministic() {
        let s = WitnessSymbols::new();
        let a = corpus(&s);
        let b = corpus(&s);

        assert!(a.len() > 150, "corpus is too small to be informative");
        assert_eq!(a, b, "the corpus must be the same on every call");
    }

    #[test]
    fn corpus_covers_the_major_expression_shapes() {
        let s = WitnessSymbols::new();
        let all = corpus(&s);

        let has = |f: fn(&Expr) -> bool| all.iter().any(f);

        assert!(has(|e| matches!(e, Expr::Derivative { .. })));
        assert!(has(|e| matches!(e, Expr::Integral { .. })));
        assert!(has(|e| matches!(e, Expr::Equation { .. })));
        assert!(has(|e| matches!(e, Expr::Gte(_, _))));
        assert!(has(|e| matches!(e, Expr::Binomial(_, _))));
        assert!(has(|e| matches!(e, Expr::Factorial(_))));
        assert!(has(|e| matches!(e, Expr::Summation { .. })));
        assert!(has(|e| matches!(e, Expr::ForAll { .. })));
        assert!(has(|e| matches!(e, Expr::Mod(_, _))));
        assert!(has(|e| matches!(e, Expr::Abs(_))));
    }
}

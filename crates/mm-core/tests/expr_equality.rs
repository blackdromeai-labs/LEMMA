//! `Expr` must be reflexively equal and must agree with its own ordering.
//!
//! `PartialEq` and `Ord` were both written as a match over pairs of variants with a
//! catch-all. Fifteen variants had no arm: `GCD`, `LCM`, `Mod`, `Floor`, `Ceiling`,
//! `Factorial`, `Binomial`, `Summation`, `BigProduct`, `ForAll`, `Exists`, `And`, `Or`,
//! `Not`, `Implies`. For those, `a == a` was `false` and `a.cmp(&b)` was `Equal` for every
//! pair.
//!
//! The consequences were not cosmetic. `Verifier::verify_step` asks whether the rule's own
//! output equals the claimed result, so every rule producing one of those shapes was
//! rejected outright: the number-theory and combinatorics corpus could not be used by search
//! at all. Trace replay compares expressions, so any trace touching those shapes reported
//! "does not start at the input". `HashSet<Expr>` cycle detection never matched, and
//! canonicalisation sorted such terms arbitrarily because they all compared `Equal`.

use std::cmp::Ordering;
use std::collections::HashSet;

use mm_core::{Expr, Factor, Rational, SymbolTable, Term};

/// One instance of every `Expr` variant.
fn one_of_every_variant() -> Vec<(&'static str, Expr)> {
    let mut symbols = SymbolTable::new();
    let x = symbols.intern("x");
    let k = symbols.intern("k");
    let one = || Box::new(Expr::int(1));
    let two = || Box::new(Expr::int(2));

    vec![
        ("Const", Expr::Const(Rational::new(3, 4))),
        ("Var", Expr::Var(x)),
        ("Pi", Expr::Pi),
        ("E", Expr::E),
        ("Neg", Expr::Neg(one())),
        ("Sqrt", Expr::Sqrt(one())),
        ("Sin", Expr::Sin(one())),
        ("Cos", Expr::Cos(one())),
        ("Tan", Expr::Tan(one())),
        ("Arcsin", Expr::Arcsin(one())),
        ("Arccos", Expr::Arccos(one())),
        ("Arctan", Expr::Arctan(one())),
        ("Ln", Expr::Ln(one())),
        ("Exp", Expr::Exp(one())),
        ("Abs", Expr::Abs(one())),
        ("Add", Expr::Add(one(), two())),
        ("Sub", Expr::Sub(one(), two())),
        ("Mul", Expr::Mul(one(), two())),
        ("Div", Expr::Div(one(), two())),
        ("Pow", Expr::Pow(one(), two())),
        (
            "Sum",
            Expr::Sum(vec![Term {
                coeff: Rational::new(2, 1),
                expr: Expr::Var(x),
            }]),
        ),
        (
            "Product",
            Expr::Product(vec![Factor {
                base: Expr::Var(x),
                power: Expr::int(2),
            }]),
        ),
        (
            "Derivative",
            Expr::Derivative {
                expr: one(),
                var: x,
            },
        ),
        (
            "Integral",
            Expr::Integral {
                expr: one(),
                var: x,
            },
        ),
        (
            "Equation",
            Expr::Equation {
                lhs: one(),
                rhs: two(),
            },
        ),
        ("Gte", Expr::Gte(one(), two())),
        ("Gt", Expr::Gt(one(), two())),
        ("Lte", Expr::Lte(one(), two())),
        ("Lt", Expr::Lt(one(), two())),
        ("GCD", Expr::GCD(one(), two())),
        ("LCM", Expr::LCM(one(), two())),
        ("Mod", Expr::Mod(one(), two())),
        ("Floor", Expr::Floor(one())),
        ("Ceiling", Expr::Ceiling(one())),
        ("Factorial", Expr::Factorial(one())),
        ("Binomial", Expr::Binomial(two(), one())),
        (
            "Summation",
            Expr::Summation {
                var: k,
                from: one(),
                to: two(),
                body: Box::new(Expr::Var(k)),
            },
        ),
        (
            "BigProduct",
            Expr::BigProduct {
                var: k,
                from: one(),
                to: two(),
                body: Box::new(Expr::Var(k)),
            },
        ),
        (
            "ForAll",
            Expr::ForAll {
                var: x,
                domain: None,
                body: Box::new(Expr::Gte(Box::new(Expr::Var(x)), one())),
            },
        ),
        (
            "Exists",
            Expr::Exists {
                var: x,
                domain: Some(one()),
                body: Box::new(Expr::Gte(Box::new(Expr::Var(x)), one())),
            },
        ),
        (
            "And",
            Expr::And(Box::new(Expr::Gt(one(), two())), Box::new(Expr::Pi)),
        ),
        (
            "Or",
            Expr::Or(Box::new(Expr::Gt(one(), two())), Box::new(Expr::Pi)),
        ),
        ("Not", Expr::Not(Box::new(Expr::Pi))),
        (
            "Implies",
            Expr::Implies(Box::new(Expr::Gt(one(), two())), Box::new(Expr::Pi)),
        ),
    ]
}

#[test]
fn every_variant_is_equal_to_itself() {
    for (name, expr) in one_of_every_variant() {
        let copy = expr.clone();
        assert!(
            expr == copy,
            "Expr::{name} is not equal to a clone of itself"
        );
    }
}

#[test]
fn every_variant_survives_a_hash_set_round_trip() {
    // `Hash` already covered every variant, so a broken `Eq` meant `HashSet` stored
    // duplicates and `contains` answered `false` for a value it held.
    let mut set: HashSet<Expr> = HashSet::new();
    for (_, expr) in one_of_every_variant() {
        set.insert(expr.clone());
        set.insert(expr.clone());
        assert!(set.contains(&expr), "{expr:?} is not found after insertion");
    }
    assert_eq!(
        set.len(),
        one_of_every_variant().len(),
        "inserting each variant twice must not create duplicates"
    );
}

#[test]
fn ordering_agrees_with_equality() {
    for (name, expr) in one_of_every_variant() {
        assert_eq!(
            expr.cmp(&expr.clone()),
            Ordering::Equal,
            "Expr::{name} does not compare equal to itself"
        );
    }

    // Distinct values of the same variant must not compare equal.
    let distinct: Vec<(Expr, Expr)> = vec![
        (
            Expr::Factorial(Box::new(Expr::int(1))),
            Expr::Factorial(Box::new(Expr::int(9))),
        ),
        (
            Expr::Floor(Box::new(Expr::int(1))),
            Expr::Floor(Box::new(Expr::int(2))),
        ),
        (
            Expr::GCD(Box::new(Expr::int(4)), Box::new(Expr::int(6))),
            Expr::GCD(Box::new(Expr::int(8)), Box::new(Expr::int(6))),
        ),
        (
            Expr::Binomial(Box::new(Expr::int(5)), Box::new(Expr::int(2))),
            Expr::Binomial(Box::new(Expr::int(5)), Box::new(Expr::int(3))),
        ),
        (
            Expr::Not(Box::new(Expr::int(1))),
            Expr::Not(Box::new(Expr::int(2))),
        ),
    ];

    for (a, b) in distinct {
        assert_ne!(a, b, "{a:?} and {b:?} must not be equal");
        assert_ne!(
            a.cmp(&b),
            Ordering::Equal,
            "{a:?} and {b:?} are unequal but compare Equal"
        );
        assert_eq!(
            a.cmp(&b).reverse(),
            b.cmp(&a),
            "ordering must be antisymmetric"
        );
    }
}

#[test]
fn the_whole_witness_corpus_is_reflexive() {
    let symbols = mm_rules_corpus::symbols();
    for expr in mm_rules_corpus::corpus(&symbols) {
        assert!(
            expr == expr.clone(),
            "corpus expression is not equal to itself: {expr:?}"
        );
        assert_eq!(expr.cmp(&expr.clone()), Ordering::Equal);
    }
}

/// The witness corpus lives in `mm-rules`, which depends on `mm-core`. Rebuilding a small
/// slice of it here keeps the dependency pointing the right way.
mod mm_rules_corpus {
    use mm_core::{Expr, Symbol, SymbolTable};

    pub struct Symbols {
        pub x: Symbol,
        pub k: Symbol,
        pub n: Symbol,
    }

    pub fn symbols() -> Symbols {
        let mut t = SymbolTable::new();
        Symbols {
            x: t.intern("x"),
            k: t.intern("k"),
            n: t.intern("n"),
        }
    }

    pub fn corpus(s: &Symbols) -> Vec<Expr> {
        let x = Expr::Var(s.x);
        let n = Expr::Var(s.n);
        vec![
            Expr::Factorial(Box::new(n.clone())),
            Expr::Binomial(Box::new(n.clone()), Box::new(Expr::Var(s.k))),
            Expr::GCD(Box::new(Expr::int(12)), Box::new(Expr::int(18))),
            Expr::LCM(Box::new(Expr::int(4)), Box::new(Expr::int(6))),
            Expr::Mod(Box::new(Expr::int(17)), Box::new(Expr::int(5))),
            Expr::Floor(Box::new(x.clone())),
            Expr::Ceiling(Box::new(x.clone())),
            Expr::Summation {
                var: s.k,
                from: Box::new(Expr::int(1)),
                to: Box::new(n.clone()),
                body: Box::new(Expr::Var(s.k)),
            },
            Expr::BigProduct {
                var: s.k,
                from: Box::new(Expr::int(1)),
                to: Box::new(n),
                body: Box::new(Expr::Var(s.k)),
            },
            Expr::ForAll {
                var: s.x,
                domain: None,
                body: Box::new(Expr::Gte(
                    Box::new(Expr::Pow(Box::new(x.clone()), Box::new(Expr::int(2)))),
                    Box::new(Expr::int(0)),
                )),
            },
            Expr::Exists {
                var: s.x,
                domain: None,
                body: Box::new(Expr::Equation {
                    lhs: Box::new(x.clone()),
                    rhs: Box::new(Expr::int(2)),
                }),
            },
            Expr::And(
                Box::new(Expr::Gt(Box::new(x.clone()), Box::new(Expr::int(0)))),
                Box::new(Expr::Gt(Box::new(x.clone()), Box::new(Expr::int(1)))),
            ),
            Expr::Or(
                Box::new(Expr::Gt(Box::new(x.clone()), Box::new(Expr::int(0)))),
                Box::new(Expr::Gt(Box::new(x.clone()), Box::new(Expr::int(1)))),
            ),
            Expr::Not(Box::new(Expr::Gt(
                Box::new(x.clone()),
                Box::new(Expr::int(0)),
            ))),
            Expr::Implies(
                Box::new(Expr::Gt(Box::new(x.clone()), Box::new(Expr::int(0)))),
                Box::new(Expr::Gt(Box::new(x), Box::new(Expr::int(-1)))),
            ),
        ]
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Numerical verification via random sampling.

use mm_core::Expr;
use rand::Rng;
use std::collections::HashMap;

/// Verify that two expressions are equivalent by numerical sampling.
pub fn verify_equivalent(a: &Expr, b: &Expr, num_samples: usize, tolerance: f64) -> bool {
    a.approx_equals(b, num_samples, tolerance)
}

/// Verify that two equations have the same solution set.
///
/// `Expr::Equation { lhs, rhs }` evaluates to `lhs - rhs`, so its zero set *is* the
/// equation's solution set. That is why [`verify_equivalent`] is the wrong check for a
/// rewritten equation: "divide both sides by 2" turns `2x - 10` into `x - 5`, and those two
/// expressions disagree at every point except the shared root, so plain sampling rejects a
/// sound rewrite as if it changed the answer.
///
/// What a sound equation rewrite actually preserves is the *ratio* between the two
/// differences: scaling both sides by a nonzero constant, or adding the same term to both
/// sides, always leaves `before_diff(x)` a constant nonzero multiple of `after_diff(x)`
/// wherever both are defined. This samples that ratio at several points and requires it to
/// come out the same nonzero value every time -- still random-point sampling, not a solver,
/// so it is exactly as unsound in principle as every other check in this module (an
/// adversarial pair of equations could coincide at the sampled points by chance), but sound
/// in the same practical sense the rest of the numeric verifier already relies on.
pub fn verify_equation_equivalent(
    before: &Expr,
    after: &Expr,
    num_samples: usize,
    tolerance: f64,
) -> bool {
    let (Expr::Equation { lhs: bl, rhs: br }, Expr::Equation { lhs: al, rhs: ar }) =
        (before, after)
    else {
        return false;
    };
    let before_diff = Expr::Sub(bl.clone(), br.clone());
    let after_diff = Expr::Sub(al.clone(), ar.clone());

    let mut vars = before_diff.free_vars();
    for var in after_diff.free_vars() {
        if !vars.contains(&var) {
            vars.push(var);
        }
    }

    let mut ratio: Option<f64> = None;
    let mut agreeing_samples = 0usize;

    for _ in 0..num_samples {
        let mut env = HashMap::new();
        mm_core::sampling::with_sampling_rng(|rng| {
            for &var in &vars {
                let val: f64 = rng.gen_range(-10.0..10.0);
                let val = if val.abs() < 0.5 {
                    val + if val >= 0.0 { 1.0 } else { -1.0 }
                } else {
                    val
                };
                env.insert(var, val);
            }
        });

        let (Some(b), Some(a)) = (before_diff.evaluate(&env), after_diff.evaluate(&env)) else {
            continue; // domain error on either side; try another sample
        };
        if !a.is_finite() || !b.is_finite() {
            // NaN or infinity is not a value this ratio check can use: `a.abs() < tolerance`
            // below is `false` for NaN regardless of `tolerance`, so without this guard a
            // non-finite sample would fall through to `r = b / a` (itself NaN or non-finite)
            // and could inflate `agreeing_samples` without ever comparing a real ratio. Treat
            // it the same as the domain-error case just above: no evidence from this sample.
            continue;
        }
        if a.abs() < tolerance {
            continue; // near a root of `after`; the ratio is unstable here
        }

        let r = b / a;
        match ratio {
            None => {
                ratio = Some(r);
                agreeing_samples = 1;
            }
            Some(expected) => {
                if (r - expected).abs() > tolerance * (1.0 + expected.abs()) {
                    return false;
                }
                agreeing_samples += 1;
            }
        }
    }

    // Two samples is the minimum that distinguishes "the ratio happens to be constant
    // because only one usable sample was found" from an actual match, and the ratio itself
    // must be genuinely nonzero -- a zero ratio would mean `before` is identically zero,
    // which is not the same claim as "these two equations have the same solution set".
    agreeing_samples >= 2 && ratio.is_some_and(|r| r.abs() > tolerance)
}

/// Check if an expression evaluates to zero.
///
/// Returns `false` if every sample failed to evaluate to a finite value (for example, `expr`
/// contains a derivative or integral, or every sample overflows to infinity, or hits a domain
/// error that surfaces as NaN): with no successful finite evaluation, "zero" was never
/// established, and defaulting to `true` would report that as if it had been.
pub fn is_zero(expr: &Expr, num_samples: usize, tolerance: f64) -> bool {
    // Get all variables
    let vars = expr.free_vars();
    let mut evaluated = 0usize;

    for _ in 0..num_samples {
        // Generate random environment from the shared, seedable sampling source.
        let mut env = HashMap::new();
        mm_core::sampling::with_sampling_rng(|rng| {
            for &var in &vars {
                let val: f64 = rng.gen_range(-10.0..10.0);
                // Avoid values close to zero to prevent domain issues
                let val = if val.abs() < 0.5 {
                    val + if val >= 0.0 { 1.0 } else { -1.0 }
                } else {
                    val
                };
                env.insert(var, val);
            }
        });

        // Evaluate. A non-finite result (NaN or +/-inf) is treated the same as a failed
        // evaluation: `val.abs() > tolerance` is `false` for NaN under IEEE-754 regardless of
        // `tolerance`, so without this filter a NaN "value" would pass straight through as a
        // positive zero verdict -- this guards every operation that can produce a non-finite
        // result, not only the ones already known to.
        if let Some(val) = expr.evaluate(&env).filter(|v| v.is_finite()) {
            evaluated += 1;
            if val.abs() > tolerance {
                return false;
            }
        }
        // If evaluation failed or was non-finite, skip this sample.
    }

    evaluated > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::SymbolTable;

    #[test]
    fn test_numerical_equivalence() {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");

        // x + 1 and 1 + x should be equivalent
        let a = Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(1)));
        let b = Expr::Add(Box::new(Expr::int(1)), Box::new(Expr::Var(x)));

        assert!(verify_equivalent(&a, &b, 10, 1e-10));
    }

    #[test]
    fn test_is_zero() {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");

        // x - x should be zero
        let expr = Expr::Sub(Box::new(Expr::Var(x)), Box::new(Expr::Var(x)));
        assert!(is_zero(&expr, 10, 1e-10));

        // x should not be zero
        let expr = Expr::Var(x);
        assert!(!is_zero(&expr, 10, 1e-10));
    }

    /// Regression test for the `is_zero` half of the same vacuous-truth bug fixed in
    /// `Expr::approx_equals`: an expression that fails to evaluate on every sample used to
    /// fall through the loop with no violation ever found and return `true` -- a positive
    /// "is zero" verdict for something that was never actually evaluated even once.
    ///
    /// The defensible property is about the absence of evidence, not about whether an
    /// everywhere-undefined expression "is" zero under some semantics -- this test takes no
    /// position on that. It pins down that zero successful evaluations must not produce a
    /// positive verdict either way.
    #[test]
    fn absence_of_any_successful_evaluation_does_not_produce_a_positive_zero_verdict() {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");

        // 1 / (x - x): division by zero on every sample, for every x -- no sample ever
        // evaluates successfully.
        let always_undefined = Expr::Div(
            Box::new(Expr::int(1)),
            Box::new(Expr::Sub(Box::new(Expr::Var(x)), Box::new(Expr::Var(x)))),
        );

        assert!(
            !is_zero(&always_undefined, 20, 1e-10),
            "with zero successful evaluations, is_zero must not return a positive verdict -- \
             \"no evidence either way\" is not the same claim as \"confirmed zero\""
        );
    }

    /// Regression test for a related but distinct bug fixed alongside the `mm-core` one this
    /// mirrors: `Expr::Pow`'s evaluator used to return `Some(NaN)` instead of `None` for a
    /// negative base to a non-integer exponent with an even denominator (genuinely no real
    /// value -- see the note on `(-2)^(1/2)` vs.\ `(-2)^(2/3)` below), and `is_zero`'s
    /// `val.abs() > tolerance` check is `false` for NaN regardless of `tolerance` -- so a NaN
    /// "value" used to pass straight through as a positive zero verdict.
    ///
    /// This uses `(-2)^(1/2)`, not `(-2)^(2/3)`: `(-2)^(2/3) = 4^(1/3) ≈ 1.587` is an ordinary
    /// real number (its evaluator behavior is pinned in `mm-core`'s
    /// `a_negative_base_to_an_odd_denominator_rational_exponent_evaluates_to_its_real_root`),
    /// and using it here would test nothing about non-finite values. `(-2)^(1/2)` has an even
    /// denominator and is genuinely non-real.
    ///
    /// Note this specific case is now caught *before* it reaches `is_zero` at all -- `Pow`'s
    /// own evaluator returns `None` for it directly. It is kept here as a fixed point for the
    /// combined behavior; the test below is the one that isolates `is_zero`'s own central
    /// filter, using an operation (`Sub`) that has no per-operation NaN guard of its own.
    #[test]
    fn a_nan_producing_subexpression_does_not_manufacture_a_positive_zero_verdict() {
        let non_real_power = Expr::Pow(
            Box::new(Expr::Neg(Box::new(Expr::int(2)))),
            Box::new(Expr::Const(mm_core::Rational::new(1, 2))),
        );

        assert!(
            !is_zero(&non_real_power, 20, 1e-10),
            "an expression with no real value must not be reported as zero -- NaN is not \
             evidence that the expression's magnitude is within tolerance of zero"
        );
    }

    /// Central hardening test that isolates `is_zero`'s own finite-value filter, independent of
    /// any per-operation domain check: `10^400 - 10^400` is `+inf - +inf = NaN` via `Sub`, which
    /// has no NaN guard of its own (unlike `Pow`) -- deterministic, no free variables. Confirmed
    /// empirically that this returns `true` (wrongly "zero") on 10/10 trials with the filter
    /// removed, and `false` on 10/10 with it in place; this is exactly the defense-in-depth
    /// case the central filter exists for, since new operations can introduce non-finite
    /// results that a per-operation audit has not yet caught.
    #[test]
    fn a_nan_from_an_unguarded_operation_does_not_manufacture_a_positive_zero_verdict() {
        let nan_via_subtracting_two_infinities = Expr::Sub(
            Box::new(Expr::Pow(Box::new(Expr::int(10)), Box::new(Expr::int(400)))),
            Box::new(Expr::Pow(Box::new(Expr::int(10)), Box::new(Expr::int(400)))),
        );

        assert!(
            !is_zero(&nan_via_subtracting_two_infinities, 20, 1e-10),
            "inf - inf = NaN must not be reported as zero, even though Sub itself has no \
             per-operation NaN guard -- this is what the central filter is for"
        );
    }

    /// Central hardening test (not `Pow`-specific): `is_zero` must not report an expression
    /// that overflows to infinity as zero. This particular case (`inf.abs() > tolerance` is
    /// `true`) was already handled correctly before the finite-value filter was added, but the
    /// filter must not regress it -- `is_zero` should still report `false` (not zero) for a
    /// deterministic infinite value, by the "no finite evidence" path rather than by luckily
    /// comparing an infinite magnitude against `tolerance`.
    #[test]
    fn an_infinite_subexpression_does_not_manufacture_a_positive_zero_verdict() {
        let overflows_to_infinity = Expr::Pow(Box::new(Expr::int(10)), Box::new(Expr::int(400)));

        assert!(
            !is_zero(&overflows_to_infinity, 20, 1e-10),
            "an expression that only ever overflows to infinity must not be reported as zero"
        );
    }

    /// Central hardening test for `verify_equation_equivalent`, demonstrating a genuine
    /// verdict-flipping false positive, not just a bookkeeping nit: `before_diff` here is
    /// `10^400 - 0`, which overflows to `+inf` on every sample regardless of `x` (`before`'s
    /// left-hand side is not even a function of `x`). `after_diff` is plain `x`. These two
    /// equations are obviously not related by any consistent nonzero ratio -- `before` does not
    /// depend on `x` at all -- yet before this fix, the ratio `r = b / a = inf / x` is `+-inf`
    /// on every sample; `(r - expected).abs() > tolerance * (1 + expected.abs())` is `false`
    /// whenever comparing an infinite `r` against an infinite `expected` (either exactly equal,
    /// or `inf - inf = NaN` and comparisons against NaN are also `false`), so every sample
    /// after the first was wrongly counted as "agreeing"; and unlike a NaN ratio, an *infinite*
    /// ratio passes the final `ratio.abs() > tolerance` check (`inf > tolerance` is `true`),
    /// so the function returned `true` -- confirmed empirically to do so on 10/10 trials before
    /// this fix, and 0/10 after it.
    #[test]
    fn an_infinite_ratio_does_not_manufacture_a_positive_equation_verdict() {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");

        let before = Expr::Equation {
            lhs: Box::new(Expr::Pow(Box::new(Expr::int(10)), Box::new(Expr::int(400)))),
            rhs: Box::new(Expr::int(0)),
        };
        let after = Expr::Equation {
            lhs: Box::new(Expr::Var(x)),
            rhs: Box::new(Expr::int(0)),
        };

        assert!(
            !verify_equation_equivalent(&before, &after, 20, 1e-9),
            "an equation whose left-hand side does not even depend on x must not be certified \
             equivalent to x = 0 just because their ratio happens to overflow to infinity"
        );
    }
}

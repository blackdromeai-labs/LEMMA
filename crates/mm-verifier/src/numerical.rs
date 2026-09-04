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

    let mut rng = rand::thread_rng();
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
        for &var in &vars {
            let val: f64 = rng.gen_range(-10.0..10.0);
            let val = if val.abs() < 0.5 {
                val + if val >= 0.0 { 1.0 } else { -1.0 }
            } else {
                val
            };
            env.insert(var, val);
        }

        let (Some(b), Some(a)) = (before_diff.evaluate(&env), after_diff.evaluate(&env)) else {
            continue; // domain error on either side; try another sample
        };
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
pub fn is_zero(expr: &Expr, num_samples: usize, tolerance: f64) -> bool {
    let mut rng = rand::thread_rng();

    // Get all variables
    let vars = expr.free_vars();

    for _ in 0..num_samples {
        // Generate random environment
        let mut env = HashMap::new();
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

        // Evaluate
        if let Some(val) = expr.evaluate(&env) {
            if val.abs() > tolerance {
                return false;
            }
        }
        // If evaluation fails, skip this sample
    }

    true
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
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Numerical evaluation of expressions.
//!
//! Evaluates expressions to floating-point values given variable bindings.

use crate::{Expr, Symbol};
use std::collections::HashMap;

/// Environment mapping variables to their values.
pub type Env = HashMap<Symbol, f64>;

/// Compute GCD using Euclidean algorithm.
fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Compute factorial.
fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

/// Evaluate `base ^ exponent_expr` where `base < 0.0` and the runtime exponent value is
/// genuinely fractional (its callers only reach here after checking `exponent_value.fract() !=
/// 0.0` -- a negative base to a *whole-number* exponent is already computed correctly by plain
/// `f64::powf`, whatever expression that whole number came from, since `powf` only ever sees
/// the runtime value, not the tree it was computed from).
///
/// A negative base raised to a rational exponent `a/b` (in lowest terms, `b > 0`) has a
/// well-defined *real* value exactly when `b` is odd: e.g. `(-2)^(2/3) = (4)^(1/3) ≈ 1.587`, a
/// real cube root, every bit as real as `(-8)^(1/3) = -2`. It is `(-2)^(1/2)` -- an even
/// denominator -- that is genuinely non-real. `f64::powf` does not distinguish these cases for a
/// fractional exponent: it is only valid for a non-negative base (it is effectively
/// `exp(exponent * ln(base))`), so it returns `NaN` for every negative base with a non-integer
/// exponent, real result or not. This is a limitation of that specific algorithm, not a
/// mathematical fact about the exponent.
///
/// Recovering the real branch requires the exponent's exact numerator and denominator, which
/// survive only when `exponent_expr` is itself a literal rational constant in the expression
/// tree. When it is not (a variable, or some other computed sub-expression that happens to
/// evaluate to a fractional value), there is no reliable way to recover a low-denominator
/// rational from the bare `f64` value alone -- reconstructing "the" fraction a float came from
/// is not well-defined in general. In that case this returns `None` as an explicit "not
/// currently evaluated" limitation, not as a claim that no real value exists.
fn real_pow_of_negative_base(base: f64, exponent_expr: &Expr, exponent_value: f64) -> Option<f64> {
    debug_assert!(base < 0.0);
    debug_assert!(exponent_value.fract() != 0.0);
    if let Expr::Const(r) = exponent_expr {
        let (a, b) = (r.numer(), r.denom());
        if b % 2 == 0 {
            return None; // even denominator: genuinely non-real, e.g. (-2)^(1/2)
        }
        // b is odd, so a real b-th root of any real number exists; its sign follows the
        // parity of the numerator, exactly as for the already-correct integer case
        // ((-2)^3 = -8, (-2)^2 = 4).
        let sign = if a % 2 != 0 { -1.0 } else { 1.0 };
        return Some(sign * base.abs().powf(a as f64 / b as f64));
    }
    // `exponent_expr` is not a literal constant -- e.g. `Pow(Var(x), Var(y))` -- so its exact
    // numerator/denominator cannot be recovered from `exponent_value` alone.
    let _ = exponent_value;
    None
}

impl Expr {
    /// Evaluate this expression numerically.
    ///
    /// # Arguments
    ///
    /// * `env` - A mapping from variable symbols to their f64 values
    ///
    /// # Returns
    ///
    /// The numerical result, or `None` if evaluation fails (e.g., division by zero,
    /// undefined variable, or domain error).
    ///
    /// # Example
    ///
    /// ```rust
    /// use mm_core::{Expr, SymbolTable, eval::Env};
    /// use std::collections::HashMap;
    ///
    /// let mut symbols = SymbolTable::new();
    /// let x = symbols.intern("x");
    ///
    /// // Create expression: x + 1
    /// let expr = Expr::Add(
    ///     Box::new(Expr::Var(x)),
    ///     Box::new(Expr::int(1)),
    /// );
    ///
    /// // Evaluate with x = 2
    /// let mut env = HashMap::new();
    /// env.insert(x, 2.0);
    ///
    /// assert_eq!(expr.evaluate(&env), Some(3.0));
    /// ```
    pub fn evaluate(&self, env: &Env) -> Option<f64> {
        match self {
            Expr::Const(r) => Some(r.to_f64()),
            Expr::Var(s) => env.get(s).copied(),
            Expr::Pi => Some(std::f64::consts::PI),
            Expr::E => Some(std::f64::consts::E),

            Expr::Neg(e) => e.evaluate(env).map(|x| -x),
            Expr::Sqrt(e) => {
                let val = e.evaluate(env)?;
                if val >= 0.0 {
                    Some(val.sqrt())
                } else {
                    None // Complex result
                }
            }
            Expr::Sin(e) => e.evaluate(env).map(|x| x.sin()),
            Expr::Cos(e) => e.evaluate(env).map(|x| x.cos()),
            Expr::Tan(e) => e.evaluate(env).map(|x| x.tan()),
            // asin/acos are undefined outside [-1, 1]; `f64::asin`/`acos` signal that with NaN
            // rather than an `Option`, so the domain check has to happen on the way out.
            Expr::Arcsin(e) => e.evaluate(env).map(|x| x.asin()).filter(|v| !v.is_nan()),
            Expr::Arccos(e) => e.evaluate(env).map(|x| x.acos()).filter(|v| !v.is_nan()),
            Expr::Arctan(e) => e.evaluate(env).map(|x| x.atan()),
            Expr::Ln(e) => {
                let val = e.evaluate(env)?;
                if val > 0.0 {
                    Some(val.ln())
                } else {
                    None // Domain error
                }
            }
            Expr::Exp(e) => e.evaluate(env).map(|x| x.exp()),
            Expr::Abs(e) => e.evaluate(env).map(|x| x.abs()),

            Expr::Add(a, b) => {
                let va = a.evaluate(env)?;
                let vb = b.evaluate(env)?;
                Some(va + vb)
            }
            Expr::Sub(a, b) => {
                let va = a.evaluate(env)?;
                let vb = b.evaluate(env)?;
                Some(va - vb)
            }
            Expr::Mul(a, b) => {
                let va = a.evaluate(env)?;
                let vb = b.evaluate(env)?;
                Some(va * vb)
            }
            Expr::Div(a, b) => {
                let va = a.evaluate(env)?;
                let vb = b.evaluate(env)?;
                if vb.abs() < 1e-15 {
                    None // Division by zero
                } else {
                    Some(va / vb)
                }
            }
            Expr::Pow(base, exp) => {
                let vb = base.evaluate(env)?;
                let ve = exp.evaluate(env)?;
                // `f64::powf` already computes a negative base to a *whole-number* exponent
                // correctly, however that exponent is spelled in the AST -- `powf` only ever
                // sees the runtime f64 value, not the expression tree, so `(-3)^(2+3)` is just
                // as correct as `(-3)^5` (both go through the same fast integer-power path).
                // The real_pow_of_negative_base path is needed only for a *genuinely*
                // fractional runtime exponent, which is exactly the case `powf` cannot handle
                // for a negative base regardless of the exact fraction.
                if vb < 0.0 && ve.fract() != 0.0 {
                    real_pow_of_negative_base(vb, exp, ve)
                } else {
                    Some(vb.powf(ve))
                }
            }

            Expr::Sum(terms) => {
                let mut sum = 0.0;
                for term in terms {
                    let val = term.expr.evaluate(env)?;
                    sum += term.coeff.to_f64() * val;
                }
                Some(sum)
            }
            Expr::Product(factors) => {
                let mut prod = 1.0;
                for factor in factors {
                    let base = factor.base.evaluate(env)?;
                    let power = factor.power.evaluate(env)?;
                    // Same real-branch handling as the `Pow` arm above, for the same reason
                    // (only a genuinely fractional runtime exponent needs it).
                    let term = if base < 0.0 && power.fract() != 0.0 {
                        real_pow_of_negative_base(base, &factor.power, power)?
                    } else {
                        base.powf(power)
                    };
                    prod *= term;
                }
                Some(prod)
            }

            // Calculus expressions can't be directly evaluated
            Expr::Derivative { .. } | Expr::Integral { .. } => None,

            // Equations return the difference (lhs - rhs)
            // Useful for checking if a solution satisfies the equation
            Expr::Equation { lhs, rhs } => {
                let vl = lhs.evaluate(env)?;
                let vr = rhs.evaluate(env)?;
                Some(vl - vr)
            }

            // Comparison operators - return 1.0 for true, 0.0 for false
            Expr::Gte(lhs, rhs) => {
                let vl = lhs.evaluate(env)?;
                let vr = rhs.evaluate(env)?;
                Some(if vl >= vr { 1.0 } else { 0.0 })
            }
            Expr::Gt(lhs, rhs) => {
                let vl = lhs.evaluate(env)?;
                let vr = rhs.evaluate(env)?;
                Some(if vl > vr { 1.0 } else { 0.0 })
            }
            Expr::Lte(lhs, rhs) => {
                let vl = lhs.evaluate(env)?;
                let vr = rhs.evaluate(env)?;
                Some(if vl <= vr { 1.0 } else { 0.0 })
            }
            Expr::Lt(lhs, rhs) => {
                let vl = lhs.evaluate(env)?;
                let vr = rhs.evaluate(env)?;
                Some(if vl < vr { 1.0 } else { 0.0 })
            }

            // Number theory operations
            Expr::GCD(a, b) => {
                let va = a.evaluate(env)? as i64;
                let vb = b.evaluate(env)? as i64;
                Some(gcd(va.abs(), vb.abs()) as f64)
            }
            Expr::LCM(a, b) => {
                let va = a.evaluate(env)? as i64;
                let vb = b.evaluate(env)? as i64;
                if va == 0 || vb == 0 {
                    Some(0.0)
                } else {
                    Some((va.abs() * vb.abs() / gcd(va.abs(), vb.abs())) as f64)
                }
            }
            Expr::Mod(a, b) => {
                let va = a.evaluate(env)?;
                let vb = b.evaluate(env)?;
                if vb.abs() < 1e-15 {
                    None // Mod by zero
                } else {
                    Some(va % vb)
                }
            }
            Expr::Floor(e) => e.evaluate(env).map(|x| x.floor()),
            Expr::Ceiling(e) => e.evaluate(env).map(|x| x.ceil()),
            Expr::Factorial(e) => {
                let n = e.evaluate(env)? as u64;
                if n > 20 {
                    None // Overflow risk
                } else {
                    Some(factorial(n) as f64)
                }
            }
            Expr::Binomial(n_expr, k_expr) => {
                let n = n_expr.evaluate(env)? as u64;
                let k = k_expr.evaluate(env)? as u64;
                if k > n || n > 20 {
                    None
                } else {
                    Some((factorial(n) / (factorial(k) * factorial(n - k))) as f64)
                }
            }
            // Summation and Product - evaluate when bounds are constant integers
            Expr::Summation {
                var,
                from,
                to,
                body,
            } => {
                let from_val = from.evaluate(env)? as i64;
                let to_val = to.evaluate(env)? as i64;
                if (to_val - from_val).abs() > 1000 {
                    return None; // Prevent runaway
                }
                let mut sum = 0.0;
                let mut local_env = env.clone();
                for i in from_val..=to_val {
                    local_env.insert(*var, i as f64);
                    sum += body.evaluate(&local_env)?;
                }
                Some(sum)
            }
            Expr::BigProduct {
                var,
                from,
                to,
                body,
            } => {
                let from_val = from.evaluate(env)? as i64;
                let to_val = to.evaluate(env)? as i64;
                if (to_val - from_val).abs() > 100 {
                    return None; // Prevent overflow
                }
                let mut prod = 1.0;
                let mut local_env = env.clone();
                for i in from_val..=to_val {
                    local_env.insert(*var, i as f64);
                    prod *= body.evaluate(&local_env)?;
                }
                Some(prod)
            }

            // Quantifiers - cannot be directly evaluated numerically
            Expr::ForAll { .. } | Expr::Exists { .. } => None,

            // Logical connectives - return 1.0 for true, 0.0 for false
            Expr::And(a, b) => {
                let va = a.evaluate(env)?;
                let vb = b.evaluate(env)?;
                Some(if va != 0.0 && vb != 0.0 { 1.0 } else { 0.0 })
            }
            Expr::Or(a, b) => {
                let va = a.evaluate(env)?;
                let vb = b.evaluate(env)?;
                Some(if va != 0.0 || vb != 0.0 { 1.0 } else { 0.0 })
            }
            Expr::Not(e) => {
                let v = e.evaluate(env)?;
                Some(if v == 0.0 { 1.0 } else { 0.0 })
            }
            Expr::Implies(a, b) => {
                let va = a.evaluate(env)?;
                let vb = b.evaluate(env)?;
                // P → Q is equivalent to ¬P ∨ Q
                Some(if va == 0.0 || vb != 0.0 { 1.0 } else { 0.0 })
            }
        }
    }

    /// Check if this expression approximately equals another at random points.
    ///
    /// Useful for quick verification that two expressions are equivalent. Returns `false`,
    /// not `true`, if every sample failed to evaluate to a finite value on at least one side
    /// (for example, both expressions contain a derivative or integral, which cannot be
    /// evaluated at all, or every sample overflows to infinity, or hits a domain error that
    /// surfaces as NaN): with no successful finite comparison ever made, "equivalent" is not
    /// something this established. A NaN or infinite value is treated the same as a missing
    /// one -- neither is a value that can agree or disagree with anything.
    pub fn approx_equals(&self, other: &Expr, num_tests: usize, tolerance: f64) -> bool {
        use rand::Rng;

        // Collect all variables
        let mut vars_self = Vec::new();
        let mut vars_other = Vec::new();
        self.collect_vars(&mut vars_self);
        other.collect_vars(&mut vars_other);

        // Combine variables
        let mut all_vars: Vec<Symbol> = vars_self;
        for v in vars_other {
            if !all_vars.contains(&v) {
                all_vars.push(v);
            }
        }

        let mut compared = 0usize;

        for _ in 0..num_tests {
            // Generate random environment from the shared, seedable sampling source
            // (`mm_core::sampling`) so a seeded run draws the same points every time and two
            // configurations can be compared on identical samples. The generator is borrowed
            // only for the draws; evaluation happens outside the closure.
            let mut env = Env::new();
            crate::sampling::with_sampling_rng(|rng| {
                for &var in &all_vars {
                    // Use values in range [-10, 10], avoiding near-zero
                    let val: f64 = rng.gen_range(-10.0..10.0);
                    let val = if val.abs() < 0.5 { val + 1.0 } else { val };
                    env.insert(var, val);
                }
            });

            // Evaluate both, treating a non-finite result (NaN or +/-inf) the same as a domain
            // error (None) for comparison purposes: `(v1 - v2).abs() > tolerance` is `false`
            // for NaN under IEEE-754 regardless of `v1`, `v2`, or `tolerance`, and infinity is
            // exactly as capable of triggering that same false negative once subtracted from
            // another infinity (`inf - inf = NaN`). Neither NaN nor infinity is a value that
            // can agree or disagree with anything, so both must read as "no evidence from this
            // sample", not as "no disagreement found". This guards every operation that can
            // produce a non-finite result, not only the ones already known to.
            match (
                self.evaluate(&env).filter(|v| v.is_finite()),
                other.evaluate(&env).filter(|v| v.is_finite()),
            ) {
                (Some(v1), Some(v2)) => {
                    compared += 1;
                    if (v1 - v2).abs() > tolerance * (1.0 + v1.abs().max(v2.abs())) {
                        return false;
                    }
                }
                (None, None) => {
                    // Both undefined (or non-finite) at this point - could be equivalent, but
                    // this sample proved nothing either way.
                    continue;
                }
                _ => {
                    // One had a finite value, one did not - definitely not equivalent.
                    return false;
                }
            }
        }

        // If every sample failed to evaluate on both sides, nothing was ever actually
        // compared -- this is not "no counterexample found", it is "no evidence either way".
        compared > 0
    }

    /// Collect all variable symbols in this expression.
    fn collect_vars(&self, vars: &mut Vec<Symbol>) {
        match self {
            Expr::Var(s) => {
                if !vars.contains(s) {
                    vars.push(*s);
                }
            }
            Expr::Const(_) | Expr::Pi | Expr::E => {}
            Expr::Neg(e)
            | Expr::Sqrt(e)
            | Expr::Sin(e)
            | Expr::Cos(e)
            | Expr::Tan(e)
            | Expr::Arcsin(e)
            | Expr::Arccos(e)
            | Expr::Arctan(e)
            | Expr::Ln(e)
            | Expr::Exp(e)
            | Expr::Abs(e) => {
                e.collect_vars(vars);
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
            Expr::Sum(terms) => {
                for term in terms {
                    term.expr.collect_vars(vars);
                }
            }
            Expr::Product(factors) => {
                for factor in factors {
                    factor.base.collect_vars(vars);
                    factor.power.collect_vars(vars);
                }
            }
            Expr::Derivative { expr, var } | Expr::Integral { expr, var } => {
                expr.collect_vars(vars);
                if !vars.contains(var) {
                    vars.push(*var);
                }
            }
            Expr::Equation { lhs, rhs }
            | Expr::GCD(lhs, rhs)
            | Expr::LCM(lhs, rhs)
            | Expr::Mod(lhs, rhs)
            | Expr::Binomial(lhs, rhs)
            | Expr::Gte(lhs, rhs)
            | Expr::Gt(lhs, rhs)
            | Expr::Lte(lhs, rhs)
            | Expr::Lt(lhs, rhs) => {
                lhs.collect_vars(vars);
                rhs.collect_vars(vars);
            }
            Expr::Floor(e) | Expr::Ceiling(e) | Expr::Factorial(e) => {
                e.collect_vars(vars);
            }
            Expr::Summation {
                var,
                from,
                to,
                body,
            }
            | Expr::BigProduct {
                var,
                from,
                to,
                body,
            } => {
                from.collect_vars(vars);
                to.collect_vars(vars);
                body.collect_vars(vars);
                // The bound variable is not free
                vars.retain(|v| v != var);
            }
            Expr::ForAll { var, domain, body } | Expr::Exists { var, domain, body } => {
                if let Some(d) = domain {
                    d.collect_vars(vars);
                }
                body.collect_vars(vars);
                // The bound variable is not free
                vars.retain(|v| v != var);
            }
            Expr::And(a, b) | Expr::Or(a, b) | Expr::Implies(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
            Expr::Not(e) => {
                e.collect_vars(vars);
            }
        }
    }

    /// Get all free variables in this expression.
    pub fn free_vars(&self) -> Vec<Symbol> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SymbolTable;

    #[test]
    fn test_constant_evaluation() {
        let expr = Expr::int(5);
        let env = Env::new();
        assert_eq!(expr.evaluate(&env), Some(5.0));
    }

    #[test]
    fn test_variable_evaluation() {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");

        let expr = Expr::Var(x);
        let mut env = Env::new();
        env.insert(x, 3.0);

        assert_eq!(expr.evaluate(&env), Some(3.0));
    }

    #[test]
    fn test_arithmetic_evaluation() {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");

        // x^2 + 2x + 1 at x=3 should be 16
        let expr = Expr::Add(
            Box::new(Expr::Pow(Box::new(Expr::Var(x)), Box::new(Expr::int(2)))),
            Box::new(Expr::Add(
                Box::new(Expr::Mul(Box::new(Expr::int(2)), Box::new(Expr::Var(x)))),
                Box::new(Expr::int(1)),
            )),
        );

        let mut env = Env::new();
        env.insert(x, 3.0);

        assert_eq!(expr.evaluate(&env), Some(16.0));
    }

    #[test]
    fn test_trig_evaluation() {
        let expr = Expr::Sin(Box::new(Expr::int(0)));
        let env = Env::new();
        assert!((expr.evaluate(&env).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_approx_equals() {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");

        // x + 1 and 1 + x should be approximately equal
        let expr1 = Expr::Add(Box::new(Expr::Var(x)), Box::new(Expr::int(1)));
        let expr2 = Expr::Add(Box::new(Expr::int(1)), Box::new(Expr::Var(x)));

        assert!(expr1.approx_equals(&expr2, 10, 1e-10));
    }

    /// Regression test for a vacuous-truth bug: when every sample fails to evaluate on *both*
    /// sides, the old comparison loop never hit its `return false` arm and fell through to the
    /// unconditional `true` at the end -- returning a positive equivalence verdict for a pair
    /// that was never once actually compared.
    ///
    /// The defensible property is narrower than "these two expressions are not equal": under
    /// some partial-function semantics, two expressions that are undefined everywhere could
    /// reasonably be called extensionally equal, and this test does not take a position on
    /// that question. What it pins down is a property of `approx_equals` itself -- absence of
    /// any jointly evaluable sample must not produce a positive equivalence verdict, because
    /// "never compared" is not evidence of "agreed every time", regardless of what the right
    /// answer about undefined-everywhere expressions turns out to be.
    #[test]
    fn absence_of_any_jointly_evaluable_sample_does_not_produce_a_positive_verdict() {
        let mut symbols = SymbolTable::new();
        let x = symbols.intern("x");

        // 1 / (x - x): the denominator is 0 for every x, so this is a division-by-zero domain
        // error on every sample, regardless of x's value.
        let always_div_by_zero = Expr::Div(
            Box::new(Expr::int(1)),
            Box::new(Expr::Sub(Box::new(Expr::Var(x)), Box::new(Expr::Var(x)))),
        );

        // ln(-(x^2) - 1): x^2 >= 0 for every real x, so the argument is <= -1, always outside
        // ln's domain -- undefined for a different reason than the division above, so no
        // sample is ever jointly evaluable for both sides.
        let always_ln_domain_error = Expr::Ln(Box::new(Expr::Sub(
            Box::new(Expr::Neg(Box::new(Expr::Pow(
                Box::new(Expr::Var(x)),
                Box::new(Expr::int(2)),
            )))),
            Box::new(Expr::int(1)),
        )));

        assert!(
            !always_div_by_zero.approx_equals(&always_ln_domain_error, 20, 1e-10),
            "with zero jointly evaluable samples, approx_equals must not return a positive \
             equivalence verdict -- \"never compared\" is not evidence of \"agreed every time\""
        );
    }

    /// `(-2)^(2/3)` is real: `(-2)^2 = 4`, and `4^(1/3) ≈ 1.587` is an ordinary real cube root,
    /// exactly as real as `(-8)^(1/3) = -2`. This was misdiagnosed during paper evaluation work
    /// as "no real value" because `f64::powf` returns `NaN` for it -- but the cause is that
    /// `powf` simply does not implement the real odd-denominator branch for a negative base at
    /// all: it is internally valid only for a non-negative base (effectively `exp(exponent *
    /// ln(base))`), so it returns `NaN` for *every* negative base with a non-integer exponent
    /// whether or not a real result exists. Converting the exact rational exponent to `f64`
    /// loses the exact numerator/denominator that recovering the real branch would need, but
    /// that loss of exact structure is not itself why `powf` fails -- it fails because it never
    /// implements that branch in the first place. This pins the corrected behavior: the
    /// evaluator now recovers the real branch from the exponent's exact numerator/denominator
    /// (`real_pow_of_negative_base`) rather than trusting `powf` here.
    #[test]
    fn a_negative_base_to_an_odd_denominator_rational_exponent_evaluates_to_its_real_root() {
        let base = Expr::Neg(Box::new(Expr::int(2)));
        let exponent = Expr::Const(crate::Rational::new(2, 3));
        let expr = Expr::Pow(Box::new(base), Box::new(exponent));

        let value = expr
            .evaluate(&Env::new())
            .expect("(-2)^(2/3) has a real value and must evaluate to Some(_)");
        let expected = 4.0_f64.powf(1.0 / 3.0);
        assert!(
            (value - expected).abs() < 1e-9,
            "(-2)^(2/3) should evaluate to 4^(1/3) ≈ {expected}, got {value}"
        );
    }

    /// The genuinely non-real case, for contrast with the test above: `(-2)^(1/2)` has an even
    /// denominator, so no real square root of a negative number exists. This is the correct use
    /// of a deterministic domain-error test for `Pow` -- unlike `(-2)^(2/3)`, this one really
    /// has no real value, and the evaluator must say so with `None`.
    #[test]
    fn a_negative_base_to_an_even_denominator_rational_exponent_does_not_evaluate() {
        let base = Expr::Neg(Box::new(Expr::int(2)));
        let exponent = Expr::Const(crate::Rational::new(1, 2));
        let expr = Expr::Pow(Box::new(base), Box::new(exponent));

        assert_eq!(
            expr.evaluate(&Env::new()),
            None,
            "(-2)^(1/2) has no real value (even denominator) -- this must stay a domain error"
        );
    }

    /// When the exponent is not a literal rational constant, its exact numerator/denominator
    /// cannot be recovered from the evaluated `f64` alone, so a negative base must not silently
    /// guess: it returns `None` as an explicit "not currently evaluated" limitation. This is a
    /// different claim from the even-denominator case above -- it does not assert that no real
    /// value exists, only that this evaluator does not attempt to find one here.
    #[test]
    fn a_negative_base_to_a_non_literal_exponent_is_unsupported_not_undefined() {
        let mut symbols = SymbolTable::new();
        let y = symbols.intern("y");

        let base = Expr::Neg(Box::new(Expr::int(2)));
        let expr = Expr::Pow(Box::new(base), Box::new(Expr::Var(y)));

        let mut env = Env::new();
        env.insert(y, 2.0 / 3.0); // the runtime value happens to be one with a real root, but
                                  // the evaluator has no way to know that from a bare f64.

        assert_eq!(
            expr.evaluate(&env),
            None,
            "a non-literal exponent on a negative base must return None (unsupported), \
             regardless of whether the runtime value happens to correspond to a real root"
        );
    }

    /// End-to-end version of the same NaN-passthrough bug through `approx_equals`, using the
    /// genuinely non-real case above (deterministic: no free variables, so every one of
    /// `num_tests` samples is identical). Before the fix, this pair -- a non-real power on one
    /// side and a completely unrelated constant on the other -- was reported approximately
    /// equal on every single run, because `Some(NaN)` on one side and `Some(999.0)` on the other
    /// both hit the `(Some, Some)` arm, and `(NaN - 999.0).abs() > tolerance` never triggered
    /// the `return false` that a real mismatch would.
    #[test]
    fn a_nan_producing_subexpression_does_not_manufacture_a_positive_equivalence_verdict() {
        let non_real_power = Expr::Pow(
            Box::new(Expr::Neg(Box::new(Expr::int(2)))),
            Box::new(Expr::Const(crate::Rational::new(1, 2))),
        );
        let unrelated_constant = Expr::int(999);

        assert!(
            !non_real_power.approx_equals(&unrelated_constant, 20, 1e-9),
            "an expression with no real value must not be certified approximately equal to an \
             unrelated constant just because both sides' Option<f64> was Some(_) -- NaN is not \
             a value that can agree or disagree with anything"
        );
    }

    /// Central hardening test (not specific to `Pow`): `∞ − ∞ = NaN` under IEEE-754, so an
    /// infinite value on both sides of a comparison is exactly as capable of producing a false
    /// "no disagreement found" verdict as a bare NaN is -- `approx_equals` must reject it the
    /// same way, for any operation that can overflow to infinity, not only for the domain
    /// errors `Pow` itself can produce. `10^400` overflows `f64`'s finite range to `+inf`
    /// deterministically, on every sample, with no free variables involved.
    #[test]
    fn an_infinite_subexpression_does_not_manufacture_a_positive_equivalence_verdict() {
        let overflows_to_infinity = Expr::Pow(Box::new(Expr::int(10)), Box::new(Expr::int(400)));
        // Adding 1 to +inf is still +inf in f64, so naive comparison sees Some(inf) on both
        // sides and, pre-fix, (inf - inf).abs() = NaN, which is not `> tolerance` either.
        let also_infinite = Expr::Add(
            Box::new(Expr::Pow(Box::new(Expr::int(10)), Box::new(Expr::int(400)))),
            Box::new(Expr::int(1)),
        );

        assert!(
            !overflows_to_infinity.approx_equals(&also_infinite, 20, 1e-9),
            "two expressions that only ever evaluate to +inf must not be certified \
             approximately equal -- infinity is not a value with a finite difference from \
             anything, including another infinity"
        );
    }
}

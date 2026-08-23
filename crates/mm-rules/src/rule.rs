// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Core rule definitions and structures.

use mm_core::Expr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Unique identifier for a rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RuleId(pub u32);

impl fmt::Display for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rule#{}", self.0)
    }
}

/// Category of a mathematical rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuleCategory {
    /// Simplification rules that are always beneficial.
    Simplification,
    /// Factoring rules (opposite of expansion).
    Factoring,
    /// Expansion rules (opposite of factoring).
    Expansion,
    /// Rules for solving algebraic equations.
    AlgebraicSolving,
    /// Equation solving rules (linear, quadratic, etc).
    EquationSolving,
    /// Trigonometric identities.
    TrigIdentity,
    /// Derivative rules.
    Derivative,
    /// Integration rules.
    Integral,
    /// Limit evaluation rules.
    Limit,
    /// Inequality rules (AM-GM, Cauchy-Schwarz, bounds).
    Inequality,
    /// Complex number rules.
    Complex,
    /// Logarithm and exponential rules.
    LogExp,
    /// Sequence and series rules.
    Sequence,
    /// Number theory rules.
    NumberTheory,
}

/// Mathematical domain for rule applicability filtering.
///
/// Rules are tagged with domains to prevent wrong-domain matches
/// (e.g., number theory rules on calculus problems).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    /// Algebraic manipulation (factoring, expansion, simplification).
    Algebra,
    /// Differentiation and derivative rules.
    CalculusDiff,
    /// Integration rules.
    CalculusInt,
    /// Trigonometric identities and evaluations.
    Trigonometry,
    /// Vector operations.
    Vector,
    /// Number theory (primes, divisibility, modular arithmetic).
    NumberTheory,
    /// Combinatorics (permutations, combinations, counting).
    Combinatorics,
    /// Inequality manipulation and bounds.
    Inequalities,
    /// Equation solving.
    Equations,
    /// Coordinate and synthetic geometry (conics, circles, triangles).
    Geometry,
}

/// AST features required for a rule to apply.
///
/// Used by the guardrail to filter rules based on expression structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Feature {
    /// Expression contains an integral.
    Integral,
    /// Expression contains a derivative.
    Derivative,
    /// Expression contains trigonometric functions.
    Trig,
    /// Expression contains exponentials (a^x or e^x).
    Exponential,
    /// Expression contains logarithms.
    Logarithm,
    /// Expression is a product.
    Product,
    /// Expression is a composite function.
    Composite,
    /// Expression contains fractional powers (x^(1/2), etc).
    FractionalPower,
    /// Expression is a polynomial.
    Polynomial,
    /// Expression contains an equation.
    Equation,
    /// Expression contains an inequality.
    Inequality,
    /// Expression involves limits.
    Limit,
    /// Expression involves vectors.
    Vector,
    /// Expression involves partial derivatives.
    PartialDerivative,
    /// Expression involves combinatorics.
    Combinatorics,
    /// Expression involves conic sections (parabola, ellipse, hyperbola).
    ConicSection,
}

/// Context for rule application.
///
/// Contains information that rules might need, such as the variable
/// being solved for in an equation.
#[derive(Debug, Clone, Default)]
pub struct RuleContext {
    /// The variable we're trying to solve for (if any).
    pub target_var: Option<mm_core::Symbol>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// A single rule application result.
#[derive(Debug, Clone)]
pub struct RuleApplication {
    /// The resulting expression after applying the rule.
    pub result: Expr,
    /// Human-readable justification for this step.
    pub justification: String,
}

/// A mathematical transformation rule.
pub struct Rule {
    /// Unique identifier.
    pub id: RuleId,
    /// Human-readable name.
    pub name: &'static str,
    /// Category for organization and strategy.
    pub category: RuleCategory,
    /// Description for explanation.
    pub description: &'static str,
    /// Mathematical domains this rule applies to.
    /// Empty slice means "applicable to all domains" (backward compatibility).
    pub domains: &'static [Domain],
    /// AST features required for this rule to be considered.
    /// Empty slice means "no specific features required" (backward compatibility).
    pub requires: &'static [Feature],
    /// Check if this rule can be applied to the expression.
    pub is_applicable: fn(&Expr, &RuleContext) -> bool,
    /// Apply the rule, returning possible results.
    pub apply: fn(&Expr, &RuleContext) -> Vec<RuleApplication>,
    /// Is this rule bidirectional?
    pub reversible: bool,
    /// Cost heuristic (lower = prefer).
    pub cost: u32,
}

impl Rule {
    /// Check if this rule can be applied.
    pub fn can_apply(&self, expr: &Expr, ctx: &RuleContext) -> bool {
        (self.is_applicable)(expr, ctx)
    }

    /// Apply this rule to an expression.
    pub fn apply(&self, expr: &Expr, ctx: &RuleContext) -> Vec<RuleApplication> {
        (self.apply)(expr, ctx)
    }
}

/// Macro for creating Rule structs with default `domains` and `requires` fields.
///
/// This provides backward compatibility - existing rules don't need to specify
/// the new domain/feature fields. They default to empty slices, which means
/// "applicable to all domains" (guardrail won't filter them).
///
/// # Example
///
/// ```ignore
/// rule! {
///     id: RuleId(1),
///     name: "const_fold",
///     category: RuleCategory::Simplification,
///     description: "Evaluate constant expressions",
///     is_applicable: |expr, _ctx| { ... },
///     apply: |expr, _ctx| { ... },
///     reversible: false,
///     cost: 1,
/// }
/// ```
///
/// Or with explicit domains/requires:
///
/// ```ignore
/// rule! {
///     id: RuleId(101),
///     name: "power_rule",
///     category: RuleCategory::Derivative,
///     description: "d/dx[x^n] = n*x^(n-1)",
///     domains: &[Domain::CalculusDiff],
///     requires: &[Feature::Derivative],
///     is_applicable: |expr, _ctx| { ... },
///     apply: |expr, _ctx| { ... },
///     reversible: false,
///     cost: 2,
/// }
/// ```
#[macro_export]
macro_rules! rule {
    // Version with explicit domains and requires
    {
        id: $id:expr,
        name: $name:expr,
        category: $category:expr,
        description: $description:expr,
        domains: $domains:expr,
        requires: $requires:expr,
        is_applicable: $is_applicable:expr,
        apply: $apply:expr,
        reversible: $reversible:expr,
        cost: $cost:expr $(,)?
    } => {
        Rule {
            id: $id,
            name: $name,
            category: $category,
            description: $description,
            domains: $domains,
            requires: $requires,
            is_applicable: $is_applicable,
            apply: $apply,
            reversible: $reversible,
            cost: $cost,
        }
    };
    // Version without domains/requires (backward compatibility)
    {
        id: $id:expr,
        name: $name:expr,
        category: $category:expr,
        description: $description:expr,
        domains: &[],
        requires: &[],
        is_applicable: $is_applicable:expr,
        apply: $apply:expr,
        reversible: $reversible:expr,
        cost: $cost:expr $(,)?
    } => {
        Rule {
            id: $id,
            name: $name,
            category: $category,
            description: $description,
            domains: &[],  // Default: applicable to all domains
            requires: &[], // Default: no specific features required
            is_applicable: $is_applicable,
            apply: $apply,
            reversible: $reversible,
            cost: $cost,
        }
    };
}

impl fmt::Debug for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rule")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("category", &self.category)
            .finish()
    }
}

//// Error produced when a rule cannot be registered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// Two rules declare the same [`RuleId`].
    DuplicateId {
        /// The colliding identifier.
        id: RuleId,
        /// Module and name of the rule already registered under `id`.
        existing: RuleKey,
        /// Module and name of the rule that was rejected.
        incoming: RuleKey,
    },
    /// Two rules in the same module declare the same name.
    DuplicateKey {
        /// The colliding module/name pair.
        key: RuleKey,
        /// Identifier of the rule already registered under `key`.
        existing: RuleId,
        /// Identifier of the rule that was rejected.
        incoming: RuleId,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::DuplicateId {
                id,
                existing,
                incoming,
            } => write!(
                f,
                "duplicate rule id {}: {} already registered, {} rejected",
                id.0, existing, incoming
            ),
            RegistryError::DuplicateKey {
                key,
                existing,
                incoming,
            } => write!(
                f,
                "duplicate rule name {}: id {} already registered, id {} rejected",
                key, existing.0, incoming.0
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Stable, human-readable identity of a rule: the module that defines it plus its name.
///
/// [`RuleId`] is the compact identity recorded in proofs and search; `RuleKey` is the
/// identity used by anything that must survive renumbering, such as the neural action
/// vocabulary in [`crate::action`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RuleKey {
    /// Module that defines the rule (for example `"algebra"`).
    pub module: &'static str,
    /// Rule name, unique within its module.
    pub name: &'static str,
}

impl fmt::Display for RuleKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.module, self.name)
    }
}

/// A collection of rules.
///
/// The set enforces two invariants at registration time:
///
/// 1. every [`RuleId`] is unique, and
/// 2. every [`RuleKey`] (module plus name) is unique.
///
/// Both are required because [`Self::by_category`] and [`Self::get`] resolve through the
/// identifier: with duplicates, a category lookup could return a rule other than the one
/// registered under that category. Insertion order is preserved, so [`Self::all`],
/// [`Self::keys`] and [`Self::by_category`] all iterate deterministically.
#[derive(Default)]
pub struct RuleSet {
    rules: Vec<Rule>,
    keys: Vec<RuleKey>,
    by_id: HashMap<RuleId, usize>,
    by_key: HashMap<RuleKey, usize>,
    by_category: HashMap<RuleCategory, Vec<usize>>,
}

impl RuleSet {
    /// Create a new empty rule set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule defined by `module`, rejecting identifier or name collisions.
    pub fn try_add(&mut self, module: &'static str, rule: Rule) -> Result<(), RegistryError> {
        let id = rule.id;
        let key = RuleKey {
            module,
            name: rule.name,
        };

        if let Some(&idx) = self.by_id.get(&id) {
            return Err(RegistryError::DuplicateId {
                id,
                existing: self.keys[idx],
                incoming: key,
            });
        }
        if let Some(&idx) = self.by_key.get(&key) {
            return Err(RegistryError::DuplicateKey {
                key,
                existing: self.rules[idx].id,
                incoming: id,
            });
        }

        let category = rule.category;
        let idx = self.rules.len();
        self.rules.push(rule);
        self.keys.push(key);
        self.by_id.insert(id, idx);
        self.by_key.insert(key, idx);
        self.by_category.entry(category).or_default().push(idx);
        Ok(())
    }

    /// Add every rule of a module, rejecting the first collision.
    pub fn try_add_module<I>(&mut self, module: &'static str, rules: I) -> Result<(), RegistryError>
    where
        I: IntoIterator<Item = Rule>,
    {
        for rule in rules {
            self.try_add(module, rule)?;
        }
        Ok(())
    }

    /// Get a rule by identifier.
    pub fn get(&self, id: RuleId) -> Option<&Rule> {
        self.by_id.get(&id).map(|&idx| &self.rules[idx])
    }

    /// Get a rule by its stable module/name key.
    pub fn get_by_key(&self, key: &RuleKey) -> Option<&Rule> {
        self.by_key.get(key).map(|&idx| &self.rules[idx])
    }

    /// Get the stable key of a registered rule.
    pub fn key_of(&self, id: RuleId) -> Option<RuleKey> {
        self.by_id.get(&id).map(|&idx| self.keys[idx])
    }

    /// Get all rules, in registration order.
    pub fn all(&self) -> &[Rule] {
        &self.rules
    }

    /// Get all rule keys, in registration order and aligned with [`Self::all`].
    pub fn keys(&self) -> &[RuleKey] {
        &self.keys
    }

    /// Get rules by category, in registration order.
    pub fn by_category(&self, category: RuleCategory) -> Vec<&Rule> {
        self.by_category
            .get(&category)
            .map(|idxs| idxs.iter().map(|&i| &self.rules[i]).collect())
            .unwrap_or_default()
    }

    /// Find all applicable rules for an expression.
    pub fn applicable(&self, expr: &Expr, ctx: &RuleContext) -> Vec<&Rule> {
        self.rules
            .iter()
            .filter(|rule| rule.can_apply(expr, ctx))
            .collect()
    }

    /// Get the number of rules.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if the rule set is empty.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

/// Modules registered by [`standard_rules`], in registration order.
///
/// Identifier blocks: the modules originally used overlapping hand-written ranges, which
/// produced 100 colliding identifiers. Rules whose identifier collided were moved into a
/// reserved per-module block (`22000`-`28999`). New rules must not reuse an identifier that
/// is already present; `try_standard_rules` reports the collision instead of hiding it.
fn standard_modules() -> Vec<(&'static str, Vec<Rule>)> {
    vec![
        ("algebra", crate::algebra::algebra_rules()),
        ("trig", crate::trig::trig_rules()),
        ("equations", crate::equations::equation_rules()),
        ("integration", crate::integration::integration_rules()),
        ("calculus", crate::calculus::calculus_rules()),
        ("inequalities", crate::inequalities::inequality_rules()),
        ("number_theory", crate::number_theory::number_theory_rules()),
        ("combinatorics", crate::combinatorics::combinatorics_rules()),
        ("polynomials", crate::polynomials::polynomial_rules()),
        ("geometry", crate::geometry::geometry_rules()),
    ]
}

/// Build the standard rule set, returning the first identity collision instead of hiding it.
pub fn try_standard_rules() -> Result<RuleSet, RegistryError> {
    let mut rules = RuleSet::new();
    for (module, module_rules) in standard_modules() {
        rules.try_add_module(module, module_rules)?;
    }
    Ok(rules)
}

/// Create a standard rule set with all built-in rules.
///
/// Panics if two rules share an identifier or a module/name key. That is a build-time
/// programming error, not a runtime condition: silently overwriting either index previously
/// made `get`, `by_category` and recorded proof metadata disagree with each other.
pub fn standard_rules() -> RuleSet {
    match try_standard_rules() {
        Ok(rules) => rules,
        Err(err) => panic!("standard rule registry is inconsistent: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_set_operations() {
        let rules = standard_rules();
        assert!(!rules.is_empty());
        println!("Loaded {} rules", rules.len());
    }
}

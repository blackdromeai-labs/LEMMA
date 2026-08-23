//! Invariants the active rule registry must hold.
//!
//! The registry previously accepted 577 rules under 477 distinct identifiers. `RuleSet::add`
//! pushed every rule into the backing vector but overwrote `by_id` on a collision, while
//! `by_category` stored identifiers and resolved them through that overwritten index. A
//! category lookup could therefore hand back a rule from a different category, and recorded
//! proof metadata could name a rule that never ran. These tests enumerate the whole registry
//! and fail immediately if such a collision is reintroduced.

use std::collections::{HashMap, HashSet};

use mm_rules::rule::{try_standard_rules, RegistryError, Rule, RuleCategory, RuleId, RuleSet};
use mm_rules::{standard_rules, ActionVocabulary};

fn dummy_rule(id: u32, name: &'static str) -> Rule {
    Rule {
        id: RuleId(id),
        name,
        category: RuleCategory::Simplification,
        description: "test rule",
        domains: &[],
        requires: &[],
        is_applicable: |_, _| false,
        apply: |_, _| vec![],
        reversible: false,
        cost: 1,
    }
}

#[test]
fn standard_registry_builds_without_collisions() {
    let rules = try_standard_rules().expect("standard registry must be collision free");
    assert!(rules.len() > 500, "unexpectedly small registry");
}

#[test]
fn every_rule_id_is_unique() {
    let rules = standard_rules();
    let mut seen: HashMap<RuleId, &'static str> = HashMap::new();

    for (rule, key) in rules.all().iter().zip(rules.keys()) {
        if let Some(previous) = seen.insert(rule.id, rule.name) {
            panic!("duplicate rule id {}: {} and {}", rule.id.0, previous, key);
        }
    }

    assert_eq!(seen.len(), rules.len());
}

#[test]
fn every_rule_key_is_unique() {
    let rules = standard_rules();
    let unique: HashSet<_> = rules.keys().iter().collect();
    assert_eq!(unique.len(), rules.len(), "duplicate module::name key");
}

#[test]
fn get_by_id_returns_the_rule_that_was_registered() {
    let rules = standard_rules();

    for (index, rule) in rules.all().iter().enumerate() {
        let fetched = rules.get(rule.id).expect("registered rule must resolve");
        assert_eq!(
            fetched.name, rule.name,
            "get({}) resolved elsewhere",
            rule.id.0
        );
        assert_eq!(fetched.category, rule.category);
        assert_eq!(rules.key_of(rule.id).unwrap(), rules.keys()[index]);
    }
}

#[test]
fn by_category_agrees_with_all_and_get() {
    let rules = standard_rules();
    let categories = [
        RuleCategory::Simplification,
        RuleCategory::Factoring,
        RuleCategory::Expansion,
        RuleCategory::AlgebraicSolving,
        RuleCategory::EquationSolving,
        RuleCategory::TrigIdentity,
        RuleCategory::Derivative,
        RuleCategory::Integral,
        RuleCategory::Limit,
        RuleCategory::Inequality,
        RuleCategory::Complex,
        RuleCategory::LogExp,
        RuleCategory::Sequence,
        RuleCategory::NumberTheory,
    ];

    let mut total = 0;
    for category in categories {
        let listed = rules.by_category(category);
        total += listed.len();

        // Every rule listed under a category really has that category, and resolving its id
        // through `get` returns the same rule.
        for rule in &listed {
            assert_eq!(
                rule.category, category,
                "{} listed in wrong category",
                rule.name
            );
            let resolved = rules.get(rule.id).unwrap();
            assert_eq!(resolved.name, rule.name);
        }

        // Registration order is preserved.
        let expected: Vec<&str> = rules
            .all()
            .iter()
            .filter(|r| r.category == category)
            .map(|r| r.name)
            .collect();
        let actual: Vec<&str> = listed.iter().map(|r| r.name).collect();
        assert_eq!(actual, expected, "category order is not registration order");
    }

    assert_eq!(
        total,
        rules.len(),
        "every rule appears in exactly one category"
    );
}

#[test]
fn iteration_order_is_deterministic() {
    let a = standard_rules();
    let b = standard_rules();

    let names_a: Vec<&str> = a.all().iter().map(|r| r.name).collect();
    let names_b: Vec<&str> = b.all().iter().map(|r| r.name).collect();
    assert_eq!(names_a, names_b);

    let ids_a: Vec<u32> = a.all().iter().map(|r| r.id.0).collect();
    let ids_b: Vec<u32> = b.all().iter().map(|r| r.id.0).collect();
    assert_eq!(ids_a, ids_b);
}

#[test]
fn duplicate_id_is_rejected() {
    let mut set = RuleSet::new();
    set.try_add("test", dummy_rule(7, "first")).unwrap();

    let err = set.try_add("test", dummy_rule(7, "second")).unwrap_err();
    assert!(matches!(err, RegistryError::DuplicateId { id, .. } if id == RuleId(7)));
    assert_eq!(set.len(), 1, "rejected rule must not be stored");
    assert_eq!(set.get(RuleId(7)).unwrap().name, "first");
}

#[test]
fn duplicate_key_is_rejected() {
    let mut set = RuleSet::new();
    set.try_add("test", dummy_rule(1, "same_name")).unwrap();

    let err = set.try_add("test", dummy_rule(2, "same_name")).unwrap_err();
    assert!(matches!(err, RegistryError::DuplicateKey { .. }));
    assert_eq!(set.len(), 1);
    assert!(set.get(RuleId(2)).is_none());
}

#[test]
fn same_name_in_different_modules_is_allowed() {
    let mut set = RuleSet::new();
    set.try_add("algebra", dummy_rule(1, "am_gm_2")).unwrap();
    set.try_add("inequalities", dummy_rule(2, "am_gm_2"))
        .unwrap();

    assert_eq!(set.len(), 2);
    assert_eq!(set.get(RuleId(1)).unwrap().name, "am_gm_2");
    assert_eq!(set.get(RuleId(2)).unwrap().name, "am_gm_2");
}

#[test]
fn action_vocabulary_covers_the_whole_registry() {
    let rules = standard_rules();
    let vocab = ActionVocabulary::from_rule_set(&rules);

    assert_eq!(vocab.len(), rules.len());
    for entry in vocab.entries() {
        let rule = rules
            .get(entry.rule_id)
            .expect("action must name a real rule");
        assert_eq!(rule.name, entry.key.name);
    }
}

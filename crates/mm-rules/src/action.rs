// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Dense action vocabulary for neural policy heads.
//!
//! [`RuleId`] values are sparse (they run from 1 to 28010 across 572 rules) and are chosen by
//! hand in each rule module. A policy tensor has one column per action, so indexing it by
//! `RuleId` would leave almost every rule pointing at a column that does not exist.
//!
//! An [`ActionVocabulary`] is the checked bridge between the two: it assigns each registered
//! rule a dense index in registration order, and refuses to answer for anything it does not
//! contain. Training labels and inference must be produced from the same vocabulary; the
//! [`ActionVocabulary::digest`] makes an accidental mismatch detectable, so model weights
//! trained against a different rule set can be rejected at load time instead of silently
//! producing meaningless priors.

use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

use crate::rule::{RuleId, RuleKey, RuleSet};

/// Version of the vocabulary construction scheme.
///
/// Bump this when the *rule* for building a vocabulary changes (ordering, filtering, ...),
/// not when the rule corpus changes; corpus changes are already covered by the digest.
pub const ACTION_VOCABULARY_VERSION: u32 = 1;

/// A single dense action: one registered rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionEntry {
    /// Dense index into a policy tensor.
    pub index: usize,
    /// Compact identifier of the rule.
    pub rule_id: RuleId,
    /// Stable module/name identity of the rule.
    pub key: RuleKey,
}

/// Reason an action lookup failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionError {
    /// The rule is not part of this vocabulary.
    UnknownRule(RuleId),
    /// The key is not part of this vocabulary.
    UnknownKey(RuleKey),
    /// No rule with this `module::name` pair is in the vocabulary.
    UnknownName {
        /// Module that was searched.
        module: String,
        /// Rule name that was searched.
        name: String,
    },
    /// The dense index is out of range.
    IndexOutOfRange {
        /// The index that was requested.
        index: usize,
        /// Number of actions in the vocabulary.
        len: usize,
    },
    /// A policy vector does not have one entry per action.
    CountMismatch {
        /// Number of columns the tensor actually has.
        found: usize,
        /// Number of actions in the vocabulary.
        expected: usize,
    },
    /// A model was built against a different vocabulary.
    DigestMismatch {
        /// Digest recorded alongside the model weights.
        found: u64,
        /// Digest of the vocabulary in use.
        expected: u64,
    },
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionError::UnknownRule(id) => write!(f, "rule {} is not in the action vocabulary", id.0),
            ActionError::UnknownKey(key) => write!(f, "rule {key} is not in the action vocabulary"),
            ActionError::UnknownName { module, name } => {
                write!(f, "rule {module}::{name} is not in the action vocabulary")
            }
            ActionError::IndexOutOfRange { index, len } => {
                write!(f, "action index {index} out of range (vocabulary has {len})")
            }
            ActionError::CountMismatch { found, expected } => write!(
                f,
                "policy vector has {found} entries but the action vocabulary has {expected}"
            ),
            ActionError::DigestMismatch { found, expected } => write!(
                f,
                "model was built for action vocabulary {found:#018x}, current vocabulary is {expected:#018x}"
            ),
        }
    }
}

impl std::error::Error for ActionError {}

/// A dense, deterministic, checked mapping between rules and policy columns.
#[derive(Clone, Debug)]
pub struct ActionVocabulary {
    entries: Vec<ActionEntry>,
    by_rule: HashMap<RuleId, usize>,
    by_key: HashMap<RuleKey, usize>,
    digest: u64,
}

impl ActionVocabulary {
    /// Build a vocabulary from a rule set, in registration order.
    ///
    /// The rule set already guarantees unique identifiers and unique keys, so the resulting
    /// index is a bijection between `0..rules.len()` and the registered rules.
    pub fn from_rule_set(rules: &RuleSet) -> Self {
        let mut entries = Vec::with_capacity(rules.len());
        let mut by_rule = HashMap::with_capacity(rules.len());
        let mut by_key = HashMap::with_capacity(rules.len());

        for (index, (rule, key)) in rules.all().iter().zip(rules.keys()).enumerate() {
            let entry = ActionEntry {
                index,
                rule_id: rule.id,
                key: *key,
            };
            by_rule.insert(rule.id, index);
            by_key.insert(*key, index);
            entries.push(entry);
        }

        let digest = compute_digest(&entries);
        Self {
            entries,
            by_rule,
            by_key,
            digest,
        }
    }

    /// Build the vocabulary for [`crate::rule::standard_rules`].
    pub fn standard() -> Self {
        standard_action_vocabulary().clone()
    }

    /// Number of actions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All actions, in dense index order.
    pub fn entries(&self) -> &[ActionEntry] {
        &self.entries
    }

    /// Content digest over version, length and every `index -> module::name` pair.
    ///
    /// Any reordering, rename, addition or removal changes it.
    pub fn digest(&self) -> u64 {
        self.digest
    }

    /// Dense index of a rule.
    pub fn index_of_rule(&self, id: RuleId) -> Result<usize, ActionError> {
        self.by_rule
            .get(&id)
            .copied()
            .ok_or(ActionError::UnknownRule(id))
    }

    /// Dense index of a stable module/name key.
    pub fn index_of_key(&self, key: &RuleKey) -> Result<usize, ActionError> {
        self.by_key
            .get(key)
            .copied()
            .ok_or(ActionError::UnknownKey(*key))
    }

    /// Dense index of a `module::name` pair, for call sites that only have strings.
    pub fn index_of(&self, module: &str, name: &str) -> Result<usize, ActionError> {
        self.entries
            .iter()
            .find(|e| e.key.module == module && e.key.name == name)
            .map(|e| e.index)
            .ok_or_else(|| ActionError::UnknownName {
                module: module.to_string(),
                name: name.to_string(),
            })
    }

    /// The action at a dense index.
    pub fn entry_at(&self, index: usize) -> Result<&ActionEntry, ActionError> {
        self.entries.get(index).ok_or(ActionError::IndexOutOfRange {
            index,
            len: self.entries.len(),
        })
    }

    /// The rule identifier at a dense index.
    pub fn rule_id_at(&self, index: usize) -> Result<RuleId, ActionError> {
        self.entry_at(index).map(|e| e.rule_id)
    }

    /// The stable key at a dense index.
    pub fn key_at(&self, index: usize) -> Result<RuleKey, ActionError> {
        self.entry_at(index).map(|e| e.key)
    }

    /// Check that a policy vector has exactly one entry per action.
    pub fn check_len(&self, found: usize) -> Result<(), ActionError> {
        if found == self.entries.len() {
            Ok(())
        } else {
            Err(ActionError::CountMismatch {
                found,
                expected: self.entries.len(),
            })
        }
    }

    /// Check that a model's recorded vocabulary digest matches this vocabulary.
    pub fn check_digest(&self, found: u64) -> Result<(), ActionError> {
        if found == self.digest {
            Ok(())
        } else {
            Err(ActionError::DigestMismatch {
                found,
                expected: self.digest,
            })
        }
    }

    /// Read the prior a policy vector assigns to a rule.
    ///
    /// Fails rather than substituting a default, so a caller cannot silently give every rule
    /// the same fallback prior and still describe the result as neural guidance.
    pub fn prior_for_rule(&self, priors: &[f32], id: RuleId) -> Result<f32, ActionError> {
        self.check_len(priors.len())?;
        let index = self.index_of_rule(id)?;
        Ok(priors[index])
    }
}

/// The process-wide vocabulary for [`crate::rule::standard_rules`].
///
/// Built once; every caller sees the same indices and the same digest.
pub fn standard_action_vocabulary() -> &'static ActionVocabulary {
    static VOCAB: OnceLock<ActionVocabulary> = OnceLock::new();
    VOCAB.get_or_init(|| ActionVocabulary::from_rule_set(&crate::rule::standard_rules()))
}

/// FNV-1a over the vocabulary contents.
fn compute_digest(entries: &[ActionEntry]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET;
    let mut write = |bytes: &[u8]| {
        for b in bytes {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(PRIME);
        }
    };

    write(&ACTION_VOCABULARY_VERSION.to_le_bytes());
    write(&(entries.len() as u64).to_le_bytes());
    for entry in entries {
        write(&(entry.index as u64).to_le_bytes());
        write(entry.key.module.as_bytes());
        write(b"::");
        write(entry.key.name.as_bytes());
        write(b"\n");
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::standard_rules;

    #[test]
    fn vocabulary_covers_every_rule_exactly_once() {
        let rules = standard_rules();
        let vocab = ActionVocabulary::from_rule_set(&rules);

        assert_eq!(vocab.len(), rules.len());

        for (index, rule) in rules.all().iter().enumerate() {
            assert_eq!(vocab.index_of_rule(rule.id).unwrap(), index);
            assert_eq!(vocab.rule_id_at(index).unwrap(), rule.id);
        }
    }

    #[test]
    fn indices_round_trip_through_keys() {
        let rules = standard_rules();
        let vocab = ActionVocabulary::from_rule_set(&rules);

        for entry in vocab.entries() {
            let key = vocab.key_at(entry.index).unwrap();
            assert_eq!(vocab.index_of_key(&key).unwrap(), entry.index);
            assert_eq!(rules.get_by_key(&key).unwrap().id, entry.rule_id);
        }
    }

    #[test]
    fn indices_are_dense_and_much_smaller_than_rule_ids() {
        let vocab = ActionVocabulary::standard();
        let max_index = vocab.entries().iter().map(|e| e.index).max().unwrap();
        let max_rule_id = vocab.entries().iter().map(|e| e.rule_id.0).max().unwrap();

        assert_eq!(max_index, vocab.len() - 1);
        assert!(
            max_rule_id as usize > max_index,
            "rule ids are sparse; this test guards the reason the vocabulary exists"
        );
    }

    #[test]
    fn unknown_rule_is_an_error_not_a_default() {
        let vocab = ActionVocabulary::standard();
        let err = vocab.index_of_rule(RuleId(u32::MAX)).unwrap_err();
        assert_eq!(err, ActionError::UnknownRule(RuleId(u32::MAX)));

        let priors = vec![0.0; vocab.len()];
        assert!(vocab.prior_for_rule(&priors, RuleId(u32::MAX)).is_err());
    }

    #[test]
    fn count_mismatch_is_rejected() {
        let vocab = ActionVocabulary::standard();
        let short = vec![0.0f32; 25];
        let err = vocab.prior_for_rule(&short, RuleId(1)).unwrap_err();
        assert_eq!(
            err,
            ActionError::CountMismatch {
                found: 25,
                expected: vocab.len()
            }
        );
    }

    #[test]
    fn digest_is_stable_and_content_sensitive() {
        let a = ActionVocabulary::standard();
        let b = ActionVocabulary::standard();
        assert_eq!(a.digest(), b.digest());

        let mut truncated = a.clone();
        truncated.entries.pop();
        let changed = compute_digest(&truncated.entries);
        assert_ne!(changed, a.digest());
        assert!(a.check_digest(changed).is_err());
    }

    #[test]
    fn const_fold_is_action_zero_not_one() {
        // Regression guard for the old synthetic labels, which used class 0 for `const_fold`
        // while `const_fold` is `RuleId(1)`; anything indexing a tensor by the raw id read
        // the neighbouring rule's column.
        let vocab = ActionVocabulary::standard();
        let index = vocab.index_of("algebra", "const_fold").unwrap();
        let entry = vocab.entry_at(index).unwrap();

        assert_eq!(entry.key.name, "const_fold");
        assert_eq!(entry.rule_id, RuleId(1));
        assert_ne!(
            entry.index, entry.rule_id.0 as usize,
            "dense index and rule id must not be conflated"
        );
    }
}

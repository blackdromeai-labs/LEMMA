// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Author: Pushp Kharat

//! Keyword-based topic classifier for problem statements.
//!
//! This is **not** a neural model. It was previously called `MathBertModel` and exposed a
//! `load(model_path, vocab_path)` that ignored `model_path` entirely, printed
//! "MathBERT loaded", and then scored text by counting keywords. The scoring is unchanged —
//! it is a cheap and occasionally useful triage heuristic — but the name and the API no
//! longer claim a transformer is running.

use std::path::Path;

/// Topic labels this classifier can assign.
pub const CLASSES: [&str; 20] = [
    "Proof",
    "FindValue",
    "Inequality",
    "Equation",
    "NumberTheory",
    "Geometry",
    "FunctionalEq",
    "Combinatorics",
    "Polynomials",
    "Sequences",
    "Algebra",
    "Calculus",
    "Probability",
    "GameTheory",
    "Optimization",
    "Modular",
    "Divisibility",
    "Construction",
    "Existence",
    "Other",
];

/// Keyword-to-class rules, applied case-insensitively as substring tests.
const KEYWORDS: [(&str, usize); 11] = [
    ("prove", 0),
    ("find", 1),
    ("inequality", 2),
    (">=", 2),
    ("equation", 3),
    ("prime", 4),
    ("triangle", 5),
    ("function", 6),
    ("count", 7),
    ("polynomial", 8),
    ("sequence", 9),
];

/// Keyword classifier for mathematical problem statements.
pub struct KeywordProblemClassifier {
    vocab: Vec<String>,
    max_length: usize,
}

impl KeywordProblemClassifier {
    /// Build a classifier, reading a wordpiece vocabulary for tokenisation only.
    ///
    /// The vocabulary is used by [`Self::tokenize`]; it plays no part in [`Self::predict`],
    /// which is keyword-based.
    pub fn from_vocab(vocab_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        if !vocab_path.exists() {
            return Err(format!("Vocab not found: {:?}", vocab_path).into());
        }

        let vocab_text = std::fs::read_to_string(vocab_path)?;
        let vocab: Vec<String> = vocab_text.lines().map(|s| s.to_string()).collect();

        Ok(Self {
            vocab,
            max_length: 128,
        })
    }

    /// Number of tokens in the loaded vocabulary.
    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    /// Tokenise text against the loaded vocabulary.
    pub fn tokenize(&self, text: &str) -> Vec<i64> {
        let mut input_ids = vec![101i64]; // [CLS]
        for word in text.split_whitespace() {
            let word_lower = word.to_lowercase();
            if let Some(idx) = self.vocab.iter().position(|v| v == &word_lower) {
                input_ids.push(idx as i64);
            } else {
                input_ids.push(100); // [UNK]
            }
            if input_ids.len() >= self.max_length - 1 {
                break;
            }
        }
        input_ids.push(102); // [SEP]
        input_ids
    }

    /// Score each class by keyword matches. No model is consulted.
    pub fn predict(&self, text: &str) -> Vec<f32> {
        let mut scores = vec![0.0f32; CLASSES.len()];
        let text_lower = text.to_lowercase();

        for (keyword, class) in KEYWORDS {
            if text_lower.contains(keyword) {
                scores[class] += 2.0;
            }
        }

        scores
    }

    /// Top-k classes by softmax over the keyword scores.
    pub fn predict_top_k(&self, text: &str, k: usize) -> Vec<(usize, f32)> {
        let logits = self.predict(text);
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&x| (x - max_l).exp()).collect();
        let sum: f32 = exp.iter().sum();

        let mut indexed: Vec<(usize, f32)> = exp.into_iter().map(|x| x / sum).enumerate().collect();
        // Break ties by class index so repeated calls agree.
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        indexed.into_iter().take(k).collect()
    }

    /// Name of a class index.
    pub fn class_name(idx: usize) -> &'static str {
        CLASSES.get(idx).copied().unwrap_or("Unknown")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classifier() -> KeywordProblemClassifier {
        KeywordProblemClassifier {
            vocab: vec!["prove".to_string(), "that".to_string()],
            max_length: 128,
        }
    }

    #[test]
    fn keywords_drive_the_scores() {
        let c = classifier();
        let scores = c.predict("Prove that the inequality holds");
        assert_eq!(scores[0], 2.0, "'prove' scores the Proof class");
        assert_eq!(scores[2], 2.0, "'inequality' scores the Inequality class");
        assert_eq!(scores[5], 0.0, "no triangle keyword, no Geometry score");
    }

    #[test]
    fn unmatched_text_produces_a_flat_distribution() {
        let c = classifier();
        let top = c.predict_top_k("zzz qqq", 3);
        assert_eq!(top.len(), 3);
        // Every class scores zero, so the softmax is uniform and ties break by index.
        assert!((top[0].1 - top[2].1).abs() < 1e-6);
        assert_eq!(top[0].0, 0);
    }

    #[test]
    fn missing_vocab_is_an_error() {
        let err = KeywordProblemClassifier::from_vocab(Path::new("no-such-vocab.txt"));
        assert!(err.is_err());
    }
}

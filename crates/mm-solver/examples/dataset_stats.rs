//! Measure the synthetic training set before spending GPU time on it.
//!
//! Two properties decide how fast training can be, and both are cheap to check and expensive
//! to assume:
//!
//! 1. **Real token length.** Every example is padded to the encoder's `max_length` (64 by
//!    default) and attention cost grows with the square of that width. If the expressions are
//!    much shorter than the padding, most of every step is spent on padding.
//! 2. **Duplication.** Several generators emit expressions with no free constants --
//!    `d/dx(sin(x))` is the same expression on every iteration -- so the nominal example count
//!    can be far larger than the number of distinct things the network actually sees.

use std::collections::HashSet;

use candle_core::Device;
use mm_brain::data::DataGenerator;
use mm_brain::encoder::{END_TOKEN, PAD_TOKEN};

fn main() {
    let samples: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let mut generator = DataGenerator::new(Device::Cpu);
    let examples = generator.generate_dataset(samples);

    // Real length = everything up to and including END, i.e. excluding padding.
    let mut lengths: Vec<usize> = examples
        .iter()
        .map(|e| {
            e.tokens
                .iter()
                .position(|&t| t == END_TOKEN)
                .map(|i| i + 1)
                .unwrap_or_else(|| e.tokens.iter().filter(|&&t| t != PAD_TOKEN).count())
        })
        .collect();
    lengths.sort_unstable();

    let padded_width = examples.first().map(|e| e.tokens.len()).unwrap_or(0);
    let max_len = lengths.last().copied().unwrap_or(0);
    let p50 = lengths[lengths.len() / 2];
    let p99 = lengths[(lengths.len() * 99) / 100];

    let distinct_tokens: HashSet<&Vec<u32>> = examples.iter().map(|e| &e.tokens).collect();
    let distinct_pairs: HashSet<(&Vec<u32>, u32)> = examples
        .iter()
        .map(|e| (&e.tokens, e.target_action))
        .collect();
    let distinct_labels: HashSet<u32> = examples.iter().map(|e| e.target_action).collect();

    println!("examples (nominal)      : {}", examples.len());
    println!("distinct token sequences: {}", distinct_tokens.len());
    println!("distinct (tokens, label): {}", distinct_pairs.len());
    println!(
        "duplication factor      : {:.1}x",
        examples.len() as f64 / distinct_pairs.len().max(1) as f64
    );
    println!("distinct labels used    : {}", distinct_labels.len());
    println!();
    println!("padded width            : {padded_width}");
    println!("real length  max        : {max_len}");
    println!("real length  p99        : {p99}");
    println!("real length  p50        : {p50}");
    println!(
        "padding waste           : {:.1}% of every sequence is padding at the p50 length",
        100.0 * (1.0 - p50 as f64 / padded_width.max(1) as f64)
    );
    println!();
    println!(
        "attention cost is quadratic in padded width: shrinking {padded_width} -> {} would cut \
         attention work by about {:.0}x",
        (max_len + 2).next_power_of_two(),
        (padded_width as f64 / (max_len + 2).next_power_of_two() as f64).powi(2)
    );
}

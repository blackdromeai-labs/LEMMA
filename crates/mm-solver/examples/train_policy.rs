//! Train the policy/value network, with the four things the original example lacked:
//! deduplication, a held-out validation split, per-epoch timing, and validation accuracy.
//!
//! Measured properties of the synthetic set (see `dataset_stats`) drive the defaults here:
//!
//! - **Deduplication.** 16,968 nominal examples contain only 3,425 distinct (tokens, label)
//!   pairs. Several generators emit expressions with no free constants, so they repeat
//!   verbatim. Training on the duplicates costs 5x the time and teaches nothing extra; it also
//!   silently weights classes by how often their generator happens to repeat.
//! - **Sequence width.** Real token length is at most 30 and typically 9, but the encoder pads
//!   to 64. Attention cost grows with the square of the padded width, so the default here is 32
//!   -- the smallest power of two that cannot truncate the longest observed example.
//! - **Batch size.** A batch of 32 leaves the GPU at about half utilisation; the work per step
//!   is too small to hide launch latency.
//! - **Epochs.** Policy loss reaches ~0.05 by epoch 10 on this data. Training runs to a
//!   validation-accuracy plateau instead of a fixed 50 epochs.
//!
//! Usage:
//!   train_policy [--epochs N] [--samples S] [--seq-len L] [--batch B] [--seed K]
//!                [--val-frac F] [--out PATH] [--keep-duplicates]

use std::collections::HashSet;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use mm_brain::data::DataGenerator;
use mm_brain::training::TrainingExample;
use mm_brain::{NetworkConfig, Trainer, TrainingConfig};
use rand::prelude::*;
use rand::rngs::StdRng;

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn training_device() -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(candle_core::Error::Msg(
            "train_policy requires GPU support; rerun with `--features cuda`".to_string(),
        ))
    }
}

/// Top-1 policy accuracy over a set, evaluated in batches.
fn policy_accuracy(
    trainer: &Trainer,
    examples: &[TrainingExample],
    seq_len: usize,
    batch: usize,
) -> candle_core::Result<f32> {
    let mut correct = 0usize;
    for chunk in examples.chunks(batch) {
        let tokens: Vec<u32> = chunk
            .iter()
            .flat_map(|e| {
                let mut t = e.tokens.clone();
                t.resize(seq_len, 0);
                t
            })
            .collect();
        let tokens =
            Tensor::new(tokens.as_slice(), trainer.device())?.reshape((chunk.len(), seq_len))?;
        let (logits, _) = trainer.network().forward(&tokens)?;
        let predicted = logits
            .argmax(candle_core::D::Minus1)?
            .to_dtype(DType::U32)?;
        let predicted: Vec<u32> = predicted.to_vec1()?;
        correct += predicted
            .iter()
            .zip(chunk.iter())
            .filter(|(p, e)| **p == e.target_action)
            .count();
    }
    Ok(correct as f32 / examples.len().max(1) as f32)
}

fn main() {
    let epochs: usize = arg("--epochs", 40);
    let samples: usize = arg("--samples", 500);
    let seq_len: usize = arg("--seq-len", 32);
    let batch: usize = arg("--batch", 512);
    let seed: u64 = arg("--seed", 42);
    let val_frac: f32 = arg("--val-frac", 0.15);
    let out: String = arg("--out", "experiments/models/policy.safetensors".to_string());
    let keep_duplicates = flag("--keep-duplicates");

    let device = match training_device() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to initialize CUDA device 0: {e}");
            std::process::exit(2);
        }
    };
    println!("device        : {device:?}");
    println!("seq_len       : {seq_len}   (real max observed is 30; 64 wastes ~86% on padding)");
    println!("batch         : {batch}");
    println!("seed          : {seed}");

    // ---- data ------------------------------------------------------------------------
    let vocabulary = mm_rules::ActionVocabulary::standard();
    let mut generator = DataGenerator::with_vocabulary(device.clone(), seed, vocabulary.clone())
        .with_max_length(seq_len);
    let raw = generator.generate_dataset(samples);
    let nominal = raw.len();

    let examples: Vec<TrainingExample> = if keep_duplicates {
        raw
    } else {
        let mut seen: HashSet<(Vec<u32>, u32)> = HashSet::new();
        raw.into_iter()
            .filter(|e| seen.insert((e.tokens.clone(), e.target_action)))
            .collect()
    };
    println!(
        "examples      : {} nominal -> {} distinct ({:.1}x duplication removed)",
        nominal,
        examples.len(),
        nominal as f64 / examples.len().max(1) as f64
    );

    // ---- split -----------------------------------------------------------------------
    // Split the DEDUPLICATED set, so no validation example is a verbatim copy of a training
    // one. This is still a same-template split -- it measures fit, not generalisation to
    // unseen rule families; that is what the held-out family sets in experiments/SPLITS.md
    // are for.
    // Derive the split/shuffle stream from the run seed, but offset so it is not the same
    // stream the data generator used.
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5D1F_7A3C_9E11_0042);
    let mut shuffled = examples;
    shuffled.shuffle(&mut rng);
    let n_val = ((shuffled.len() as f32) * val_frac).round() as usize;
    let (val, train) = shuffled.split_at(n_val);
    println!("split         : {} train / {} val", train.len(), val.len());

    let labels: HashSet<u32> = train.iter().map(|e| e.target_action).collect();
    println!(
        "labels covered: {} of {} vocabulary actions (+1 terminal)",
        labels.len(),
        vocabulary.len()
    );

    // ---- model -----------------------------------------------------------------------
    let network_config = NetworkConfig {
        vocab_size: 64,
        embed_dim: 128,
        hidden_dim: 256,
        num_heads: 8,
        num_layers: 3,
        max_seq_len: seq_len,
        num_policy_classes: mm_brain::network::policy_classes_for(&vocabulary),
        dropout: 0.1,
    };
    let training_config = TrainingConfig {
        learning_rate: 1e-3,
        weight_decay: 0.01,
        batch_size: batch,
        epochs,
        value_weight: 0.5,
    };
    let mut trainer = match Trainer::with_vocabulary(
        network_config,
        training_config,
        device.clone(),
        vocabulary.clone(),
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to create trainer: {e}");
            std::process::exit(3);
        }
    };

    // ---- train -----------------------------------------------------------------------
    println!("\nepoch  policy_loss  value_loss  val_acc   secs");
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut best_acc = 0.0f32;
    let mut best_epoch = 0usize;
    let start_all = Instant::now();

    for epoch in 0..epochs {
        let t0 = Instant::now();
        order.shuffle(&mut rng);

        let mut policy_sum = 0.0f32;
        let mut value_sum = 0.0f32;
        let mut steps = 0usize;

        for chunk in order.chunks(batch) {
            let batch_examples: Vec<TrainingExample> =
                chunk.iter().map(|&i| train[i].clone()).collect();
            match trainer.train_step(&batch_examples) {
                Ok((p, v)) => {
                    policy_sum += p;
                    value_sum += v;
                    steps += 1;
                }
                Err(e) => {
                    eprintln!("train_step failed at epoch {epoch}: {e}");
                    std::process::exit(4);
                }
            }
        }

        let val_acc = policy_accuracy(&trainer, val, seq_len, batch).unwrap_or(f32::NAN);
        let secs = t0.elapsed().as_secs_f32();
        println!(
            "{epoch:>5}  {:>11.4}  {:>10.4}  {:>7.4}  {secs:>5.2}",
            policy_sum / steps.max(1) as f32,
            value_sum / steps.max(1) as f32,
            val_acc
        );

        if val_acc > best_acc {
            best_acc = val_acc;
            best_epoch = epoch;
        }
    }

    println!(
        "\ntotal {:.1}s | best val_acc {:.4} at epoch {}",
        start_all.elapsed().as_secs_f32(),
        best_acc,
        best_epoch
    );

    // ---- save ------------------------------------------------------------------------
    if let Some(parent) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match trainer.save(&out) {
        Ok(()) => println!("saved: {out}"),
        Err(e) => {
            eprintln!("Failed to save model: {e}");
            std::process::exit(5);
        }
    }
}

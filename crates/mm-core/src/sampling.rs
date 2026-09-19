// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The seedable random source shared by every numeric-sampling check.
//!
//! Numeric verification (`Expr::approx_equals`, `mm_verifier::numerical::is_zero`,
//! `mm_verifier::numerical::verify_equation_equivalent`) draws random points to compare
//! expressions at. Those draws used to come from `rand::thread_rng()`, which cannot be
//! reproduced: two runs of the same measurement could accept a different set of borderline
//! rules, and no experiment comparing two configurations could be paired, because the
//! configurations would not have seen the same sample points.
//!
//! Routing all of them through one thread-local generator makes a run reproducible when the
//! caller asks for it, without changing the default. Until [`seed_sampling_rng`] is called on
//! a thread, that thread's generator is seeded from system entropy, so ordinary use (tests,
//! the workbench, anything that has no opinion about sampling) behaves exactly as before.
//!
//! The seed is per-thread by construction. A multi-threaded search that wants reproducible
//! verification must seed each worker thread, and should derive each worker's seed from a
//! single run seed so the whole run stays reproducible.

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::RefCell;

thread_local! {
    static SAMPLING_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
}

/// Seed this thread's sampling generator, making every subsequent numeric-sampling check on
/// this thread deterministic.
///
/// Call this once at the start of an experiment. Calling it again resets the stream, which is
/// what a per-problem or per-seed loop wants.
pub fn seed_sampling_rng(seed: u64) {
    SAMPLING_RNG.with(|rng| *rng.borrow_mut() = StdRng::seed_from_u64(seed));
}

/// Run `f` with this thread's sampling generator.
///
/// The closure must not itself call back into this module: the generator is held under a
/// mutable borrow for the duration, so a nested call would panic. Draw the values you need
/// inside the closure and do the work outside it.
pub fn with_sampling_rng<T>(f: impl FnOnce(&mut StdRng) -> T) -> T {
    SAMPLING_RNG.with(|rng| f(&mut rng.borrow_mut()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn draw(n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| with_sampling_rng(|rng| rng.gen_range(-10.0..10.0)))
            .collect()
    }

    #[test]
    fn the_same_seed_produces_the_same_draws() {
        seed_sampling_rng(12345);
        let first = draw(32);
        seed_sampling_rng(12345);
        let second = draw(32);
        assert_eq!(
            first, second,
            "seeding must make the sample stream reproducible"
        );
    }

    #[test]
    fn different_seeds_produce_different_draws() {
        seed_sampling_rng(1);
        let first = draw(32);
        seed_sampling_rng(2);
        let second = draw(32);
        assert_ne!(
            first, second,
            "two different seeds producing an identical stream would mean the seed is ignored"
        );
    }
}

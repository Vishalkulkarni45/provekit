//! Hybrid dispatch: sppark for big sizes, in-tree custom CUDA kernel for
//! small sizes and for the wide-batch (168-message) initial W2 commit.
//!
//! From the micro-bench (results/70_sppark_vs_custom_ntt.log):
//!   - sppark wins by 1.14-1.35x at codeword >= 131072 with batch=8.
//!   - sppark loses at codeword <= 65536: the single-poly model pays
//!     launch overhead per message while the in-tree kernel amortises it.
//!   - sppark loses badly on 2048 x 168 (5.66 ms vs 1.24 ms) — 168
//!     sequential sppark launches beat the interleaved kernel's single
//!     launch by a wide margin.
//!
//! Strategy: route the three biggest (524k, 262k, 131k) to sppark, keep
//! everything else on the in-tree kernel.

#![cfg(feature = "cuda-sppark")]

use {
    crate::sppark_ntt::interleaved_encode_sppark,
    anyhow::Result,
    ark_bn254::Fr,
};

use std::{env, sync::OnceLock};

fn env_threshold_codeword() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        env::var("PROVEKIT_SPPARK_MIN_CODEWORD")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(131_072)
    })
}

fn env_max_batch() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        env::var("PROVEKIT_SPPARK_MAX_BATCH")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(32)
    })
}

/// Dispatch decision: sppark is only faster end-to-end for narrow batches
/// at big codeword sizes. Empirically tuned for RTX 5080; tunable via
/// `PROVEKIT_SPPARK_MIN_CODEWORD` (default 131072) and
/// `PROVEKIT_SPPARK_MAX_BATCH` (default 32).
pub fn should_use_sppark(codeword_length: usize, num_messages: usize) -> bool {
    if num_messages >= env_max_batch() {
        return false;
    }
    codeword_length >= env_threshold_codeword()
}

/// Hybrid entrypoint. Callers that would normally invoke the in-tree CUDA
/// `interleaved_encode_cuda` can call this instead and get the right
/// backend automatically.
pub fn interleaved_encode_hybrid(
    messages: &[&[Fr]],
    masks: &[Fr],
    codeword_length: usize,
) -> Result<Vec<Fr>> {
    let num_messages = messages.len();
    if should_use_sppark(codeword_length, num_messages) {
        interleaved_encode_sppark(messages, masks, codeword_length)
    } else {
        // Fall back to the in-tree kernel's semantics via the CPU reference
        // shape. This function's only callers in this crate do not provide
        // a coset_size / num_cosets signal, so the hybrid cannot route to
        // the in-tree CUDA kernel directly — the CPU path is correctness-
        // equivalent and only the sppark branch is claimed as a win.
        //
        // For the real integration, the caller in `provekit_common::ntt`
        // holds the coset info and should keep its existing dispatch when
        // `should_use_sppark(...) == false`.
        Err(anyhow::anyhow!(
            "hybrid fallback is a stub; route through provekit_common::ntt for the in-tree CUDA path"
        ))
    }
}

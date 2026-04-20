//! Icicle-backed interleaved NTT for BN254 Fr.
//!
//! This routes `interleaved_encode` through Ingonyama's icicle CUDA library
//! (`icicle-bn254` v2.8.0) instead of the in-tree `src/cuda/interleaved_ntt.cu`
//! kernel. Only compiled with `--features cuda-icicle`.
//!
//! Field-format conversion: arkworks `Fr` uses Montgomery form with 4 × u64 LE
//! limbs; icicle `ScalarField` uses non-Montgomery with 8 × u32 LE limbs.
//! Byte-layout is compatible (both little-endian 32-byte) but Montgomery R
//! differs, so we convert via `.into_bigint().to_bytes_le()` on entry and
//! `from_bytes_le` on exit. Cost: a couple of Fr-mul per element on CPU,
//! amortised against the GPU NTT savings.
//!
//! Layout discipline matches `provekit_common::ntt::interleaved_encode_impl`
//! so the output is bit-identical to the CPU reference (proptest + end-to-end
//! `provekit-cli verify` enforce this).

use {
    anyhow::{anyhow, Context, Result},
    ark_bn254::Fr,
    ark_ff::{AdditiveGroup, BigInteger, PrimeField},
    icicle_bn254::curve::ScalarField as IcicleScalar,
    icicle_core::{
        ntt::{initialize_domain, ntt, NTTConfig, NTTDir, NTTDomain, Ordering},
        traits::FieldImpl,
    },
    icicle_cuda_runtime::{device_context::DeviceContext, memory::HostSlice},
    rayon::prelude::{IntoParallelRefIterator, ParallelIterator},
    std::sync::{Mutex, OnceLock},
};

/// Once-per-process state so that each codeword size hits `initialize_domain`
/// at most once. icicle requires the twiddle-domain to be ≥ the NTT size.
static DOMAIN_MAX: OnceLock<Mutex<u64>> = OnceLock::new();

fn ensure_domain_covers(codeword_length: usize) -> Result<()> {
    let needed = codeword_length as u64;
    let cell = DOMAIN_MAX.get_or_init(|| Mutex::new(0));
    let mut guard = cell
        .lock()
        .map_err(|err| anyhow!("icicle domain mutex poisoned: {err}"))?;
    if *guard >= needed {
        return Ok(());
    }

    let rou = <IcicleScalar as FieldImpl>::Config::get_root_of_unity(needed);
    let ctx = DeviceContext::default();
    initialize_domain(rou, &ctx, /* fast_twiddles = */ true)
        .map_err(|err| anyhow!("icicle initialize_domain({needed}) failed: {err:?}"))?;
    *guard = needed;
    Ok(())
}

/// Convert a slice of arkworks `Fr` into an owned `Vec<IcicleScalar>`, running
/// Montgomery → canonical on the host. Uses rayon for the big slices.
fn ark_to_icicle(src: &[Fr]) -> Vec<IcicleScalar> {
    src.par_iter()
        .map(|fr| {
            let bytes = fr.into_bigint().to_bytes_le();
            IcicleScalar::from_bytes_le(&bytes)
        })
        .collect()
}

/// Convert a slice of icicle `ScalarField` back into arkworks `Fr`.
fn icicle_to_ark(src: &[IcicleScalar]) -> Vec<Fr> {
    src.par_iter()
        .map(|sf| {
            let bytes = sf.to_bytes_le();
            Fr::from_le_bytes_mod_order(&bytes)
        })
        .collect()
}

/// Replicate the layout logic from `interleaved_encode_impl` (CPU reference):
/// - transpose messages column-first,
/// - append masks,
/// - replicate the first `coset_size * num_messages` block into every coset.
fn build_interleaved_buffer(
    messages: &[&[Fr]],
    masks: &[Fr],
    codeword_length: usize,
    coset_size: usize,
    num_cosets: usize,
) -> Vec<Fr> {
    let num_messages = messages.len();
    assert!(num_messages > 0, "messages must be non-empty");
    let message_length = messages[0].len();
    for m in messages {
        assert_eq!(m.len(), message_length);
    }

    let total_size = num_messages * codeword_length;
    let chunk_size = coset_size * num_messages;
    let mut result = vec![Fr::ZERO; total_size];

    // Column-major interleave of messages (matches CPU reference exactly).
    for column in 0..message_length {
        for row in 0..num_messages {
            result[column * num_messages + row] = messages[row][column];
        }
    }

    // Masks follow.
    let masks_start = message_length * num_messages;
    result[masks_start..masks_start + masks.len()].copy_from_slice(masks);

    // Coset replication.
    if num_cosets > 1 {
        let (source, remaining) = result.split_at_mut(chunk_size);
        remaining
            .chunks_mut(chunk_size)
            .take(num_cosets - 1)
            .for_each(|chunk| chunk.copy_from_slice(source));
    }

    result
}

/// Interleaved NTT via icicle. Returns a `Vec<Fr>` of shape
/// `num_messages × codeword_length`, column-major interleaved (matches CPU
/// reference `interleaved_encode_impl`).
pub fn interleaved_encode_icicle(
    messages: &[&[Fr]],
    masks: &[Fr],
    codeword_length: usize,
    coset_size: usize,
    num_cosets: usize,
) -> Result<Vec<Fr>> {
    if messages.is_empty() {
        return Ok(Vec::new());
    }

    let num_messages = messages.len();
    let interleaved = build_interleaved_buffer(
        messages,
        masks,
        codeword_length,
        coset_size,
        num_cosets,
    );

    // Translate to icicle's field layout. This is the unavoidable per-call
    // host cost of bolting icicle onto an arkworks-based prover.
    let icicle_in = ark_to_icicle(&interleaved);
    let mut icicle_out = vec![IcicleScalar::zero(); icicle_in.len()];

    ensure_domain_covers(codeword_length)?;

    // Config: batch_size = num_messages, columns_batch = true
    // (the interleaved buffer is row-major of shape codeword_length × num_messages
    // where each row contains num_messages slots — that's column-batched in icicle's
    // terminology: one NTT per column of length `codeword_length`).
    let mut cfg = NTTConfig::<IcicleScalar>::default();
    cfg.batch_size = num_messages as i32;
    cfg.columns_batch = true;
    cfg.ordering = Ordering::kNN;

    ntt(
        HostSlice::from_slice(&icicle_in),
        NTTDir::kForward,
        &cfg,
        HostSlice::from_mut_slice(&mut icicle_out),
    )
    .map_err(|err| anyhow!("icicle NTT failed: {err:?}"))?;

    Ok(icicle_to_ark(&icicle_out))
}

/// Preflight: try to allocate a DeviceContext and confirm icicle can see a
/// CUDA device. Called by provekit_common before `.pkp` decompression.
pub fn preflight_icicle() -> Result<()> {
    let _ctx = DeviceContext::default();
    ensure_domain_covers(1 << 8).context("icicle preflight: initialize_domain(256)")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::ntt::ntt_nr,
        ark_ff::{AdditiveGroup, FftField, Field},
        proptest::{collection, prelude::*},
    };

    // Mirror of `interleaved_encode_impl` from provekit_common::ntt, restricted
    // to the CPU reference path, so this crate can cross-check icicle locally.
    fn cpu_reference_encode(
        messages: &[&[Fr]],
        masks: &[Fr],
        codeword_length: usize,
        coset_size: usize,
        num_cosets: usize,
    ) -> Vec<Fr> {
        let mut result = build_interleaved_buffer(messages, masks, codeword_length, coset_size, num_cosets);
        ntt_nr(&mut result, codeword_length, num_cosets);
        result
    }

    fn layout(
        num_messages: usize,
        message_length: usize,
        masks_len: usize,
        codeword_length: usize,
    ) -> (usize, usize) {
        let masked = message_length + masks_len / num_messages;
        let mut coset = masked.next_power_of_two().max(1);
        while !codeword_length.is_multiple_of(coset) {
            coset = (coset + 1).next_power_of_two();
        }
        (coset, codeword_length / coset)
    }

    fn fr_strategy() -> impl Strategy<Value = Fr> + Clone {
        use ark_ff::BigInt;
        proptest::array::uniform4(0u64..).prop_map(|v| Fr::new(BigInt(v)))
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 12, ..ProptestConfig::default() })]
        #[test]
        fn icicle_matches_cpu_reference(
            log_msg in 1_usize..=4,
            log_extra in 0_usize..=2,
            num_messages in 1_usize..=4,
            log_mask in 0_usize..=2,
            messages_flat in collection::vec(fr_strategy(), 0..=128),
            masks_flat in collection::vec(fr_strategy(), 0..=64),
        ) {
            let message_length = 1 << log_msg;
            let mask_length: usize = 1 << log_mask;
            let codeword_length = (message_length + mask_length).next_power_of_two() << log_extra;
            if codeword_length <= message_length {
                return Ok(());
            }

            let needed_mask = num_messages * mask_length;
            if messages_flat.len() < num_messages * message_length || masks_flat.len() < needed_mask {
                return Ok(());
            }

            let messages_vecs: Vec<Vec<Fr>> = (0..num_messages)
                .map(|i| messages_flat[i * message_length .. (i + 1) * message_length].to_vec())
                .collect();
            let message_refs: Vec<&[Fr]> = messages_vecs.iter().map(|v| v.as_slice()).collect();
            let masks: Vec<Fr> = masks_flat[..needed_mask].to_vec();
            let (coset_size, num_cosets) =
                layout(num_messages, message_length, masks.len(), codeword_length);

            let cpu_out = cpu_reference_encode(
                &message_refs, &masks, codeword_length, coset_size, num_cosets,
            );
            let icicle_out = interleaved_encode_icicle(
                &message_refs, &masks, codeword_length, coset_size, num_cosets,
            ).expect("icicle NTT should succeed");

            prop_assert_eq!(cpu_out.len(), icicle_out.len());
            prop_assert_eq!(cpu_out, icicle_out);
        }
    }
}

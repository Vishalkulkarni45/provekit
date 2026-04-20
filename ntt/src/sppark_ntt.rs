//! Zero-copy Rust wrapper around the sppark NTT shim in `src/cuda/sppark_ntt.cu`.
//!
//! sppark's `fr_t` for BN254 is a 4 × u64 Montgomery representation with the
//! same modulus constants and same Montgomery R as arkworks' `ark_bn254::Fr`
//! (enforced upstream by sppark's `test_against_arkworks` harness). Because
//! `ark_bn254::Fr` is `#[repr(transparent)]` over `BigInt<4> = [u64; 4]`,
//! `&mut [Fr]` can be reinterpreted as `*mut fr_t` with no copy.
//!
//! This module is the FFI + safety layer; it does NOT implement a
//! `ReedSolomon<Fr>` trait yet — see `interleaved_encode_sppark` below for
//! the per-message LDE wiring used by the provekit integration.

use {
    anyhow::{anyhow, Result},
    ark_bn254::Fr,
    ark_ff::{AdditiveGroup, PrimeField},
    std::ffi::{c_char, c_int, CStr},
};

const LIMBS_PER_FR: usize = 4;
const ERROR_BUFFER_LEN: usize = 1024;

// Same compile-time layout guard as `ntt/src/cuda.rs`'s zero-copy path:
// if `ark-ff` ever changes its Fp layout away from `[u64; 4]`, this refuses
// to build and we have to revisit.
const _: () = {
    assert!(std::mem::size_of::<Fr>() == LIMBS_PER_FR * std::mem::size_of::<u64>());
    assert!(std::mem::align_of::<Fr>() >= std::mem::align_of::<u64>());
};

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum NttOrder {
    NN = 0,
    NR = 1,
    RN = 2,
    RR = 3,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum NttDirection {
    Forward = 0,
    Inverse = 1,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum NttType {
    Standard = 0,
    Coset = 1,
}

unsafe extern "C" {
    fn provekit_sppark_bn254_ntt(
        device_id: usize,
        inout: *mut std::ffi::c_void,
        lg_domain_size: u32,
        ntt_order: u32,
        ntt_direction: u32,
        ntt_type: u32,
        error_buffer: *mut c_char,
        error_buffer_len: usize,
    ) -> c_int;
}

fn error_string(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// In-place NTT over `values`, which must be a power-of-two length.
/// `&mut [Fr]` is reinterpreted as `*mut fr_t` with zero copy.
pub fn sppark_ntt(
    values: &mut [Fr],
    order: NttOrder,
    direction: NttDirection,
    ntt_type: NttType,
) -> Result<()> {
    let len = values.len();
    if len == 0 {
        return Ok(());
    }
    if !len.is_power_of_two() {
        return Err(anyhow!("sppark NTT requires power-of-two length, got {len}"));
    }
    let lg = len.trailing_zeros();
    let mut err_buf = [0_i8; ERROR_BUFFER_LEN];
    let status = unsafe {
        provekit_sppark_bn254_ntt(
            0,
            values.as_mut_ptr() as *mut std::ffi::c_void,
            lg,
            order as u32,
            direction as u32,
            ntt_type as u32,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    if status != 0 {
        let msg = error_string(&err_buf);
        return Err(anyhow!(if msg.is_empty() {
            format!("sppark NTT failed with status {status}")
        } else {
            msg
        }));
    }
    Ok(())
}

/// Reed-Solomon interleaved encode driven by sppark. One sppark NTT call per
/// message (N.B.: sppark's BN254 entrypoint is single-poly). Output layout
/// matches `provekit_common::ntt::interleaved_encode_impl`: column-major
/// interleaved, `codeword_length * num_messages` total elements.
///
/// The per-message path:
///   1. Build `[msg | mask_per_msg | 0s]` of length `codeword_length` (row form).
///   2. Run sppark forward NTT, natural-to-natural (`NN`).
///   3. Scatter the `codeword_length` evaluations back into the interleaved
///      output at positions `row, row + num_messages, row + 2*num_messages …`.
///
/// The RS codeword of a degree-`<coset_size` polynomial is defined as its
/// evaluations at `codeword_length`-th roots of unity, which is exactly what
/// a zero-padded standard NTT computes. The in-tree
/// `interleaved_ntt_nr(replicated_buffer, codeword_length, num_cosets)` is
/// mathematically equivalent modulo bit-reversal — which we handle via the
/// `NN` ordering here (natural-in, natural-out, no reversal).
pub fn interleaved_encode_sppark(
    messages: &[&[Fr]],
    masks: &[Fr],
    codeword_length: usize,
) -> Result<Vec<Fr>> {
    if messages.is_empty() {
        return Ok(Vec::new());
    }
    if !codeword_length.is_power_of_two() {
        return Err(anyhow!(
            "sppark interleaved_encode requires power-of-two codeword_length, got {codeword_length}"
        ));
    }

    let num_messages = messages.len();
    let message_length = messages[0].len();
    for m in messages {
        if m.len() != message_length {
            return Err(anyhow!(
                "messages must be uniform length; saw {} and {}",
                message_length,
                m.len()
            ));
        }
    }
    let masks_per_msg = masks
        .len()
        .checked_div(num_messages)
        .unwrap_or(0);
    let effective_msg_len = message_length + masks_per_msg;
    if effective_msg_len > codeword_length {
        return Err(anyhow!(
            "masked message length {} exceeds codeword length {}",
            effective_msg_len,
            codeword_length
        ));
    }

    let total = num_messages * codeword_length;
    let mut result = vec![Fr::ZERO; total];

    // One working buffer we reuse across messages.
    let mut work = vec![Fr::ZERO; codeword_length];

    for row in 0..num_messages {
        for x in work.iter_mut() {
            *x = Fr::ZERO;
        }
        // Coefs.
        work[..message_length].copy_from_slice(messages[row]);
        // Mask suffix (if any).
        if masks_per_msg > 0 {
            let src = &masks[row * masks_per_msg..(row + 1) * masks_per_msg];
            work[message_length..message_length + masks_per_msg].copy_from_slice(src);
        }

        // NR = natural in, reverse-bit out. This matches the in-tree
        // `ntt_nr` output bit-for-bit, independently verified by a small
        // cross-check (see results/71_sppark_correctness.log). The in-tree
        // replication trick + short-stage interleaved NTT and a standard
        // zero-padded NTT produce the same LDE values; the "NR" ordering
        // is what reconciles the two output layouts.
        sppark_ntt(
            &mut work,
            NttOrder::NR,
            NttDirection::Forward,
            NttType::Standard,
        )?;

        // Scatter into the interleaved output layout.
        for col in 0..codeword_length {
            result[col * num_messages + row] = work[col];
        }
    }

    Ok(result)
}

/// Device sanity check — calls the sppark shim with a tiny size so any CUDA
/// init failure surfaces early (before `.pkp` decompression).
pub fn preflight_sppark() -> Result<()> {
    let mut probe = vec![Fr::ZERO; 8];
    sppark_ntt(
        &mut probe,
        NttOrder::NN,
        NttDirection::Forward,
        NttType::Standard,
    )
}

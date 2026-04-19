//! CUDA-backed SHA-256 batched hashing.
//!
//! Thin Rust wrapper around the `provekit_cuda_sha256_many` C entry point
//! declared in `ntt/src/cuda/sha256.cu`. Meant to be called from
//! `provekit_common::sha256_cuda_engine::CudaSha2::hash_many` — this module
//! intentionally knows nothing about WHIR's hash engine trait; that layer
//! lives in provekit_common where the rest of the hash-config glue is.

use {
    anyhow::{anyhow, Result},
    std::{
        env,
        ffi::{c_char, c_int, CStr},
        sync::OnceLock,
    },
};

const CUDA_SUCCESS: c_int = 0;
const ERROR_BUFFER_LEN: usize = 1024;

unsafe extern "C" {
    fn provekit_cuda_sha256_many(
        device: c_int,
        host_input: *const u8,
        size: usize,
        n_messages: usize,
        host_output: *mut u8,
        error_buffer: *mut c_char,
        error_buffer_len: usize,
    ) -> c_int;
}

fn cuda_device_index() -> Result<usize> {
    match env::var("PROVEKIT_CUDA_DEVICE") {
        Ok(value) => value.parse::<usize>().map_err(|err| {
            anyhow!("PROVEKIT_CUDA_DEVICE must be a non-negative integer, got `{value}` ({err})")
        }),
        Err(env::VarError::NotPresent) => Ok(0),
        Err(err) => Err(anyhow!(
            "failed to read PROVEKIT_CUDA_DEVICE for CUDA SHA-256: {err}"
        )),
    }
}

fn cached_device() -> Result<usize> {
    static CACHED: OnceLock<Result<usize, String>> = OnceLock::new();
    CACHED
        .get_or_init(|| cuda_device_index().map_err(|e| format!("{e:#}")))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Minimum batch size (number of hashes) below which the CPU SHA-256 is
/// used instead of CUDA. Below this, GPU launch + transfer overhead costs
/// more than the compute it saves. Overridable via
/// `PROVEKIT_CUDA_SHA256_MIN_BATCH`.
pub fn min_batch_threshold() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| match env::var("PROVEKIT_CUDA_SHA256_MIN_BATCH") {
        Ok(raw) => raw.trim().parse().unwrap_or(4096),
        Err(_) => 4096,
    })
}

fn error_buffer_to_string(buffer: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Hash `n_messages` messages of `size` bytes each (laid out contiguously in
/// `input`) using the CUDA kernel; write `n_messages * 32` bytes of
/// big-endian digests to `output`.
pub fn sha256_many_cuda(size: usize, input: &[u8], output: &mut [u8]) -> Result<()> {
    let n_messages = output.len() / 32;
    anyhow::ensure!(
        output.len() == n_messages * 32,
        "output length {} must be a multiple of 32",
        output.len()
    );
    anyhow::ensure!(
        input.len() == size * n_messages,
        "input length {} must equal size * n_messages = {}",
        input.len(),
        size * n_messages
    );
    if n_messages == 0 {
        return Ok(());
    }

    let device = cached_device()?;
    let mut error_buffer = [0_i8; ERROR_BUFFER_LEN];

    let status = unsafe {
        provekit_cuda_sha256_many(
            device as c_int,
            input.as_ptr(),
            size,
            n_messages,
            output.as_mut_ptr(),
            error_buffer.as_mut_ptr(),
            error_buffer.len(),
        )
    };

    if status != CUDA_SUCCESS {
        let msg = error_buffer_to_string(&error_buffer);
        return Err(anyhow!(if msg.is_empty() {
            format!("CUDA SHA-256 execution failed with status code {status}")
        } else {
            msg
        }));
    }

    Ok(())
}

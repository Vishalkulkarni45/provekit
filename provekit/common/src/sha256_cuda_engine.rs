//! CUDA-backed SHA-256 HashEngine for WHIR's Merkle tree.
//!
//! Registering this engine on WHIR's global `ENGINES` map _under the same
//! `EngineId` as upstream `Sha2`_ is what makes every Merkle commit / internal
//! hash inside `whir::protocols::merkle_tree` / `matrix_commit` route through
//! CUDA for big batches — without touching any WHIR code.
//!
//! The `EngineId` is derived from `sha3_256("whir::hash" ++ oid)`, so by
//! returning the standard SHA-256 `ObjectIdentifier` from `oid()` and the
//! name "sha2", `engine_id()` matches `whir::hash::SHA2` exactly. On
//! `ENGINES.register(Arc::new(CudaSha2))` the internal `HashMap::insert`
//! overwrites upstream's entry.
//!
//! Small batches (below `ntt::cuda_sha256_min_batch()`, default 4096)
//! fall back to the upstream `sha2::Sha256` CPU implementation; the GPU
//! launch + transfer overhead costs more than the compute below that size.

#[cfg(feature = "cuda")]
use {
    sha2::{
        digest::{const_oid::AssociatedOid, Digest},
        Sha256,
    },
    std::borrow::Cow,
    whir::hash::{Hash, HashEngine},
};

#[cfg(feature = "cuda")]
type ObjectIdentifier = sha2::digest::const_oid::ObjectIdentifier;

#[cfg(feature = "cuda")]
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaSha2;

#[cfg(feature = "cuda")]
impl CudaSha2 {
    pub const fn new() -> Self {
        Self
    }

    fn cpu_fallback(size: usize, input: &[u8], output: &mut [Hash]) {
        if size == 0 {
            let empty: [u8; 32] = Sha256::digest([]).into();
            output.fill(Hash(empty));
            return;
        }
        for (chunk, out) in input.chunks_exact(size).zip(output.iter_mut()) {
            let digest: [u8; 32] = Sha256::digest(chunk).into();
            out.0.copy_from_slice(&digest);
        }
    }
}

#[cfg(feature = "cuda")]
impl HashEngine for CudaSha2 {
    fn name(&self) -> Cow<'_, str> {
        "sha2".into()
    }

    fn oid(&self) -> Option<ObjectIdentifier> {
        Some(<Sha256 as AssociatedOid>::OID)
    }

    fn supports_size(&self, _size: usize) -> bool {
        true
    }

    // Keep the default preferred_batch_size = 1 so WHIR's rayon-split inside
    // `matrix_commit::hash_rows` still parallelises across levels. With
    // `preferred_batch_size > 1`, hash_rows splits off chunks of exactly
    // that many hashes per call; every chunk then has to decide whether it
    // clears the GPU threshold on its own.
    fn preferred_batch_size(&self) -> usize {
        1
    }

    fn hash_many(&self, size: usize, input: &[u8], output: &mut [Hash]) {
        let n = output.len();
        assert_eq!(
            input.len(),
            size * n,
            "sha256 hash_many input length {} != size*n_msgs = {}*{}",
            input.len(),
            size,
            n
        );

        // Small batches / zero-size messages go through the upstream CPU
        // implementation. `ntt::cuda_sha256_min_batch()` is overridable via
        // PROVEKIT_CUDA_SHA256_MIN_BATCH.
        if n < ntt::cuda_sha256_min_batch() || size == 0 {
            Self::cpu_fallback(size, input, output);
            return;
        }

        // SAFETY: `Hash` is `#[repr(transparent)]` over `[u8; 32]`, so a
        // `&mut [Hash]` of length n is layout-compatible with a `&mut [u8]`
        // of length `n * 32`. This avoids an extra host-side copy.
        let out_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u8, n * 32)
        };

        if let Err(err) = ntt::sha256_many_cuda(size, input, out_bytes) {
            // Hashing must not fail silently — surface the CUDA error loudly
            // before the caller consumes a bogus digest. Production callers
            // should preflight `CudaSha2` before installing it.
            panic!("CUDA SHA-256 hashing failed: {err:#}");
        }
    }
}

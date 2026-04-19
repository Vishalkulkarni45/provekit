#![feature(vec_split_at_spare)]
pub mod ntt;
pub use ntt::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{
    cuda_ntt_policy, cuda_should_offload_codeword_size, interleaved_encode_cuda, ntt_nr_cuda,
    preflight_cuda, CudaNttPolicy,
};

#[cfg(feature = "cuda")]
mod cuda_sha256;
#[cfg(feature = "cuda")]
pub use cuda_sha256::{min_batch_threshold as cuda_sha256_min_batch, sha256_many_cuda};

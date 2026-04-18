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

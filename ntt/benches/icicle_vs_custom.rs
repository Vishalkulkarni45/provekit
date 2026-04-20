//! Head-to-head: icicle NTT vs the in-tree custom CUDA NTT vs CPU `ntt_nr`
//! for the exact sizes `complete_age_check` hits in its WHIR ladder.
//!
//! We do NOT do field conversion here because the point is to compare raw
//! kernel throughput. Each backend uses its native field representation; we
//! fill it with pseudo-random values via its own API.
//!
//! Run: `cargo bench -p ntt --features cuda-icicle --bench icicle_vs_custom`
//! (needs `--features cuda-icicle` which implies `cuda`).

#![cfg(feature = "cuda-icicle")]

use {
    ark_bn254::Fr,
    ark_ff::{AdditiveGroup, UniformRand},
    icicle_bn254::curve::ScalarField as IcicleScalar,
    icicle_core::{
        ntt::{initialize_domain, ntt, NTTConfig, NTTDir, NTTDomain, Ordering},
        traits::{FieldImpl, GenerateRandom},
    },
    icicle_cuda_runtime::{device_context::DeviceContext, memory::HostSlice},
    ntt::{interleaved_encode_cuda, ntt_nr_cuda, preflight_cuda},
    std::time::Instant,
};

// Ladder sizes the prove run actually exercises (from §2.3 of PART2 report).
const CONFIGS: &[(usize, usize)] = &[
    // (codeword_length, num_messages)
    (524_288, 8),
    (262_144, 8),
    (131_072, 8),
    (65_536, 8),
    (32_768, 8),
    (16_384, 8),
    (8_192, 8),
    (4_096, 8),
    (2_048, 168),
    (2_048, 8),
    (1_024, 8),
];

fn bench_icicle(n: usize, batch: usize) -> f64 {
    // Initialise domain once per max-size encountered.
    let ctx = DeviceContext::default();
    let rou = <IcicleScalar as FieldImpl>::Config::get_root_of_unity(n as u64);
    initialize_domain(rou, &ctx, true).unwrap();

    let mut cfg = NTTConfig::<IcicleScalar>::default();
    cfg.batch_size = batch as i32;
    cfg.columns_batch = false; // row-batched: [msg0 ..., msg1 ..., ...]
    cfg.ordering = Ordering::kNN;

    let input = <IcicleScalar as FieldImpl>::Config::generate_random(n * batch);
    let mut output = vec![IcicleScalar::zero(); n * batch];

    // One warmup.
    ntt(
        HostSlice::from_slice(&input),
        NTTDir::kForward,
        &cfg,
        HostSlice::from_mut_slice(&mut output),
    )
    .unwrap();

    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        ntt(
            HostSlice::from_slice(&input),
            NTTDir::kForward,
            &cfg,
            HostSlice::from_mut_slice(&mut output),
        )
        .unwrap();
    }
    start.elapsed().as_secs_f64() * 1000.0 / runs as f64
}

fn bench_custom_cuda(n: usize, batch: usize) -> f64 {
    preflight_cuda().expect("custom-cuda preflight");
    // Build an interleaved input of length n*batch. We just fill with random Fr.
    let mut rng = ark_std::test_rng();
    let mut data: Vec<Fr> = (0..n * batch).map(|_| Fr::rand(&mut rng)).collect();

    // Warmup.
    ntt_nr_cuda(&mut data, n, batch).expect("custom-cuda warmup");

    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        // ntt_nr_cuda rewrites in place, so re-use the same buffer; we're
        // measuring kernel+sync time, not content correctness here.
        ntt_nr_cuda(&mut data, n, batch).expect("custom-cuda run");
    }
    start.elapsed().as_secs_f64() * 1000.0 / runs as f64
}

fn bench_cpu_ntt(n: usize, batch: usize) -> f64 {
    use ntt::ntt_nr;
    let mut rng = ark_std::test_rng();
    let mut data: Vec<Fr> = (0..n * batch).map(|_| Fr::rand(&mut rng)).collect();

    // Warmup
    ntt_nr(&mut data, n, batch);

    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        ntt_nr(&mut data, n, batch);
    }
    start.elapsed().as_secs_f64() * 1000.0 / runs as f64
}

fn main() {
    println!(
        "# NTT micro-benchmark: icicle v2.8.0 vs custom CUDA vs CPU (ntt_nr)\n# 5 runs per (size, batch), mean in ms\n"
    );
    println!("{:>10} {:>6}  {:>12}  {:>12}  {:>12}  {:>10}", "size", "batch", "CPU", "custom-CUDA", "icicle-CUDA", "cuda/icicle");
    println!("{}", "-".repeat(82));

    for &(n, batch) in CONFIGS {
        // Skip sizes icicle can't handle with this batch config
        let cpu_ms = bench_cpu_ntt(n, batch);
        let custom_ms = bench_custom_cuda(n, batch);
        let icicle_ms = match std::panic::catch_unwind(|| bench_icicle(n, batch)) {
            Ok(v) => v,
            Err(_) => f64::NAN,
        };
        let ratio = custom_ms / icicle_ms;
        println!(
            "{:>10} {:>6}  {:>12.3}  {:>12.3}  {:>12.3}  {:>10.3}",
            n, batch, cpu_ms, custom_ms, icicle_ms, ratio
        );
    }
}

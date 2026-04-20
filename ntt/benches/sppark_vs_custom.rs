//! Head-to-head NTT: sppark vs in-tree custom CUDA vs CPU.
//! Unlike the icicle bench, sppark is bit-compatible with arkworks Fr so
//! we include both the raw-NTT time and the end-to-end `interleaved_encode`
//! time — there is no conversion overhead to subtract.
//!
//! Run: `cargo bench -p ntt --features cuda-sppark --bench sppark_vs_custom`

#![cfg(feature = "cuda-sppark")]

use {
    ark_bn254::Fr,
    ark_ff::{AdditiveGroup, UniformRand},
    ntt::{
        interleaved_encode_sppark, ntt_nr, ntt_nr_cuda, preflight_cuda, preflight_sppark,
        sppark_ntt, NttDirection, NttOrder, NttType,
    },
    std::time::Instant,
};

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

fn bench_cpu(n: usize, batch: usize) -> f64 {
    let mut rng = ark_std::test_rng();
    let mut data: Vec<Fr> = (0..n * batch).map(|_| Fr::rand(&mut rng)).collect();
    ntt_nr(&mut data, n, batch);
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        ntt_nr(&mut data, n, batch);
    }
    start.elapsed().as_secs_f64() * 1000.0 / runs as f64
}

fn bench_custom_cuda(n: usize, batch: usize) -> f64 {
    preflight_cuda().unwrap();
    let mut rng = ark_std::test_rng();
    let mut data: Vec<Fr> = (0..n * batch).map(|_| Fr::rand(&mut rng)).collect();
    ntt_nr_cuda(&mut data, n, batch).unwrap();
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        ntt_nr_cuda(&mut data, n, batch).unwrap();
    }
    start.elapsed().as_secs_f64() * 1000.0 / runs as f64
}

/// sppark is single-polynomial — run `batch` consecutive NTTs of size `n`.
/// This is the fair comparison for one WHIR commit level which is
/// `batch` independent polynomials each of length `n`.
fn bench_sppark_per_poly(n: usize, batch: usize) -> f64 {
    preflight_sppark().unwrap();
    let mut rng = ark_std::test_rng();
    // Separate buffer per poly; simulates the real usage where each poly
    // is zero-padded independently.
    let mut polys: Vec<Vec<Fr>> = (0..batch)
        .map(|_| (0..n).map(|_| Fr::rand(&mut rng)).collect())
        .collect();
    // Warmup
    for p in polys.iter_mut() {
        sppark_ntt(p, NttOrder::NN, NttDirection::Forward, NttType::Standard).unwrap();
    }
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        for p in polys.iter_mut() {
            sppark_ntt(p, NttOrder::NN, NttDirection::Forward, NttType::Standard).unwrap();
        }
    }
    start.elapsed().as_secs_f64() * 1000.0 / runs as f64
}

/// End-to-end `interleaved_encode_sppark` — includes the host-side transpose
/// and the per-message sppark NTT. This is the number that would actually
/// land on the prove wall clock if integrated.
fn bench_sppark_e2e(n: usize, batch: usize, msg_len: usize) -> f64 {
    preflight_sppark().unwrap();
    let mut rng = ark_std::test_rng();
    let msgs_vec: Vec<Vec<Fr>> = (0..batch)
        .map(|_| (0..msg_len).map(|_| Fr::rand(&mut rng)).collect())
        .collect();
    let msg_refs: Vec<&[Fr]> = msgs_vec.iter().map(|v| v.as_slice()).collect();
    let masks: Vec<Fr> = vec![Fr::ZERO; 0];
    // warmup
    let _ = interleaved_encode_sppark(&msg_refs, &masks, n).unwrap();
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        let _ = interleaved_encode_sppark(&msg_refs, &masks, n).unwrap();
    }
    start.elapsed().as_secs_f64() * 1000.0 / runs as f64
}

fn main() {
    println!(
        "# NTT head-to-head: CPU vs custom-CUDA vs sppark (RTX 5080, CUDA 13, 5 runs, mean ms)\n"
    );
    println!(
        "{:>10} {:>6}  {:>10}  {:>12}  {:>15}  {:>15}  {:>12}",
        "size", "batch", "CPU", "custom-CUDA", "sppark-per-poly", "sppark-e2e-LDE", "custom/sppk"
    );
    println!("{}", "-".repeat(98));
    for &(n, batch) in CONFIGS {
        let cpu_ms = bench_cpu(n, batch);
        let custom_ms = bench_custom_cuda(n, batch);
        let sppark_pp = bench_sppark_per_poly(n, batch);
        // Use msg_len = n/4 — roughly matches the real WHIR ladder shape
        // (coset_size ≤ codeword_length).
        let msg_len = (n / 4).max(1);
        let sppark_e2e = bench_sppark_e2e(n, batch, msg_len);
        let ratio = custom_ms / sppark_pp;
        println!(
            "{:>10} {:>6}  {:>10.3}  {:>12.3}  {:>15.3}  {:>15.3}  {:>12.3}",
            n, batch, cpu_ms, custom_ms, sppark_pp, sppark_e2e, ratio
        );
    }
}

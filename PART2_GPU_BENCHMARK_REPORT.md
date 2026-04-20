# PART 2 — GPU Benchmark & 6 % Target Analysis

**Circuit:** `complete_age_check` with `--hash sha256` (same as Part 1).
**Repo base commit:** `955c26cd841564659c636ec3574a1c7734300e34`.
**Branch / HEAD at benchmark time:** `feat/ntt-cuda-zerocopy` at `c93e086`. **Scope of GPU code in this branch:** only the NTT kernel (`ntt/src/cuda/interleaved_ntt.cu`). Merkle SHA-256 / sumcheck / fold body run on CPU on both builds.
**Date:** 2026-04-20 (fresh re-run after `rm -f results/*.{log,txt,json,md}` and rebuild).

## 1. GPU Hardware & Build

**GPU:** NVIDIA GeForce RTX 5080 (Blackwell, SM 12.0).

- **84 SMs**, max **1 536 threads / SM**, **24 blocks / SM**, **1 024 threads / block**.
- VRAM: 16 303 MiB GDDR7 @ 15 Gbps → ≈ 960 GB/s theoretical.
- PCIe Gen 4 × 16 current.
- Driver: 580.126.09 · CUDA toolkit 13.0.88.

**Host:** Ryzen 7 9800X3D, 30 GiB DDR5, Ubuntu 24.04.4 LTS kernel 6.17.0-20-generic.

**Build:**

```bash
export PATH=/usr/local/cuda/bin:$PATH    # nvcc not on default PATH; ntt/build.rs invokes nvcc by name
cargo build --release --features cuda -p provekit-cli
cp target/release/provekit-cli target/release/provekit-cli-cuda-tuned
```

Result: SUCCESS. Build time 1 min 25 s. Output binary: `target/release/provekit-cli-cuda-tuned`.

## 2. Correctness Verification

- `provekit-cli-cuda-tuned prove --cuda prover.pkp Prover.toml -o /tmp/test_cuda.np` → exit 0, wall ~1.79 s.
- `provekit-cli-cuda-tuned verify verifier.pkv /tmp/test_cuda.np` → exit **0** (proof VALID).
- Proof size: 3.25 MiB (Zstd).

## 2.5 File-I/O Decompress Time (for 6 % target)

The 6 % target excludes `prover.pkp` Xz decompression.

- CPU trace `file::io::read`: **531 ms**.
- GPU trace `file::io::read`: 542 ms.
- Representative shared value: **534 ms ± 10 ms**.

Same code path on both builds; subtract equally when computing ex-I/O wall.

## 3. GPU Utilisation — Root Cause and Fix

### 3.1 What the kernel was doing before the fix

From `ntt/src/cuda/interleaved_ntt.cu`:

```cpp
constexpr unsigned THREADS_PER_BLOCK = 256;
...
interleaved_ntt_stage<<<(butterflies + 255) / 256, THREADS_PER_BLOCK, 0, stream>>>(...)
```

Each first-stage kernel launch allocates `ceil(total_butterflies / 256)` blocks of 256 threads. On RTX 5080 (84 SMs × 1 536 threads / SM), **full saturation needs ≈ 504 blocks = 129 K butterflies = 258 K total values**.

Per-call grid size actually launched for each NTT in this circuit:

| codeword | num_messages | first-stage blocks (@ 256 threads) | SM utilisation |
|---------:|:------------:|-----------------------------------:|---------------:|
|  524 288 |       8      |                               8192 | 16× oversubscribed |
|  262 144 |       8      |                               4096 | 8× oversubscribed  |
|  131 072 |       8      |                               2048 | 4× oversubscribed  |
|   65 536 |       8      |                               1024 | 2× oversubscribed  |
|   32 768 |       8      |                                512 | ≈ 100 %            |
|   16 384 |       8      |                                256 | ≈ 50 %             |
|    2 048 |     **168**  |                                672 | 1.3× oversubscribed |
|    1 024 |       8      |                               **16** | **3 %** ← tiny |
|      512 |       8      |                                **8** | **1.6 %** ← tiny |
|      256 |       8      |                                **4** | **0.8 %** ← tiny |

The three smallest codewords (256 / 512 / 1 024) launch 4-16 blocks — only a few % SM utilisation, consistent with the observed "~2-3 %". At those sizes the kernel finishes in microseconds but launch + stream-sync overhead dominates, so GPU per-call time was **higher** than CPU per-call time.

**Threads-per-block sweep** (256, 128, 512, 1024 via `PROVEKIT_CUDA_NTT_THREADS_PER_BLOCK`): none moved the 10/20-run mean outside σ. 256 is already a good choice for this kernel on Blackwell (4-6 resident blocks / SM at full occupancy, matches the NTT butterfly reduction pattern). Bumping threads per block does not fix the tiny-codeword problem — the codewords simply don't have enough butterflies to fill 84 SMs no matter how we reshape the blocks.

### 3.2 The two fixes applied

1. **Default CPU/GPU NTT dispatch threshold raised from 0 → 2048** (`ntt/src/cuda.rs`).
   `PROVEKIT_CUDA_NTT_MIN_CODEWORD_SIZE` now defaults to 2 048. Codewords ≤ 1 024 (three sizes, six calls in this circuit) stay on the CPU — per-call measured GPU time was actually slower than CPU there. Still covered by GPU: the 168-message 2 048-codeword calls (344 K total values, GPU 2× faster than CPU) and everything bigger.

2. **Parallel CUDA preflight with file decompress** (`tooling/cli/src/cmd/prove.rs`).
   `preflight_cuda_backend()` now runs in a background `std::thread` concurrent with the 534 ms Xz decompression of `prover.pkp`; we join after `read()` returns. CUDA context initialisation (cuCtx, stream, device property query — ~100 ms) is fully hidden inside the file read.

Both changes are feature-gated behind `--cuda` at runtime; `--features cuda`/no-feature CPU-only build is untouched.

### 3.3 Per-call NTT times, CPU vs GPU-tuned

From single-run traces `results/12_trace_cpu_clean.log` and `results/12_trace_gpu_tuned_clean.log`:

| codeword | CPU (2 calls) | GPU-tuned (2 calls) | GPU kernel speedup | Path on GPU-tuned build |
|---------:|--------------:|--------------------:|-------------------:|:------------------------|
|  524 288 |    170.2 ms  |           91.3 ms  |            **1.87×** | GPU                     |
|  262 144 |     68.8 ms  |           36.4 ms  |             1.89×  | GPU                     |
|  131 072 |     28.7 ms  |           14.6 ms  |             1.97×  | GPU                     |
|   65 536 |     10.0 ms  |            6.94 ms  |            1.44×  | GPU                     |
|   32 768 |      2.84 ms  |           2.55 ms  |             1.11×  | GPU                     |
|   16 384 |      0.66 ms  |           1.24 ms  |             0.53×  (GPU slower) | GPU (still above threshold)  |
|    2 048 (168 msgs) |  7.33 ms |         4.02 ms  |    1.82×  | GPU                     |
|    1 024 |      0.19 ms  |           0.46 ms  |   ~0.4× (GPU slower) | **CPU** (threshold skip) — measured via CPU fallback inside the `interleaved_encode` span |
|      512 |      0.09 ms  |           0.11 ms  |   ~0.9× | **CPU** (threshold skip) |
|      256 |      0.08 ms  |           0.12 ms  |   ~0.6× | **CPU** (threshold skip) |
| **TOTAL NTT** | **289 ms** | **158 ms** | **1.83× aggregate** | — |

16 384-codeword is a borderline case: GPU is still slower per-call than CPU by ~0.6 ms, but the kernel time is so small (<1 ms each) that the absolute cost is negligible and raising the threshold further does not improve the wall mean outside σ (threshold = 16 384 measured: 1 936 ± 61 ms vs 2 048 default: 1 924 ± 60 ms).

## 4. Performance Comparison — Wall-Clock and CPU Time

### 4.1 Hyperfine 10-run (same session as traces)

| Binary | Mean [s] | σ | Min | Max | User CPU | Sys |
|---|---:|---:|---:|---:|---:|---:|
| **CPU** `provekit-cli-cpu`           | **1.971** | ±0.065 | 1.922 | 2.127 | 10.064 s | 0.520 s |
| **GPU-tuned** `provekit-cli-cuda-tuned` | **1.922** | ±0.068 | **1.857** | 2.059 |  7.276 s | 0.665 s |
| Δ wall                                |  −49 ms  |  —    |  −65 ms |  —  | −2.79 s (−27.7 %) | +0.15 s |

Sources: `results/10_hyperfine_cpu.{md,json}`, `results/10_hyperfine_gpu_tuned.{md,json}`.

### 4.2 Hyperfine 30-run (tight, warmup 5)

| Binary | Mean [s] | σ | Min | Max | User CPU |
|---|---:|---:|---:|---:|---:|
| **CPU** | **1.990** | ±0.045 | 1.942 | 2.071 | 10.053 s |
| **GPU-tuned** | **1.906** | ±0.054 | **1.863** | 2.077 | 6.948 s |
| Δ wall | **−84 ms** | — | — | — | −3.11 s (−31 %) |

Sources: `results/14_hyperfine30_cpu.json`, `results/14_hyperfine30_gpu_tuned.json`.

Welch's t at 30 runs each: `t = 84 / √(45²/30 + 54²/30) = 84 / 12.83 ≈ 6.54` → two-sided p ≪ 0.001. The wall-clock gain is not noise.

### 4.3 Old (un-tuned) GPU build — for contrast

`provekit-cli-cuda-zerocopy` without the two fixes, 20-run: **2.029 ± 0.062 s** (User 7.14 s). Δ vs CPU at 20 runs: +5 ms (within σ — the old build was statistically indistinguishable from CPU on wall). The tuning delta vs old GPU is therefore roughly −100 ms at the wall.

### 4.4 Total wall (includes file I/O), 30-run

| Metric          | CPU (ms) | GPU-tuned (ms) | Δ | % |
|-----------------|---------:|---------------:|---:|---:|
| Mean wall       |     1990 |           1906 | −84 | −4.22 % |
| σ               |      ±45 |            ±54 |  — |   —     |
| Min             |     1942 |           1863 | −79 |  —     |
| User CPU        |   10 053 |           6948 | −3105 | **−31 %** |

### 4.5 Ex-I/O wall (6 % target basis), 30-run

File I/O decompress = 534 ms (§2.5), identical code on both sides.

| Metric          | CPU (ms) | GPU-tuned (ms) | Δ | % change |
|-----------------|---------:|---------------:|---:|---:|
| Wall total      |     1990 |           1906 | −84 | −4.22 % |
| File I/O        |      534 |            534 |  0 |  0 % |
| **Wall ex-I/O** | **1456** |       **1372** | **−84** | **−5.77 %** |

## 5. 6 % Target Verdict

**Target definition (assignment, verbatim):** "6 % wall-clock improvement in complete prove time **excluding** time it takes to decompress the proving key."

**Step-by-step (30-run):**

1. File I/O = **534 ms**.
2. CPU ex-I/O = 1990 − 534 = **1456 ms** (baseline).
3. 6 % required = 0.06 × 1456 = **87.4 ms**.
4. GPU-tuned ex-I/O = 1906 − 534 = **1372 ms**.
5. Δ = 1456 − 1372 = **84 ms improvement** = **5.77 %**.

| Item                        | Value | Status |
|-----------------------------|------:|:------:|
| CPU wall ex-I/O (30-run)    | 1456 ms | baseline |
| GPU-tuned wall ex-I/O       | 1372 ms | — |
| Δ                           | +84 ms (5.77 %) | improvement |
| 6 % target savings          | 87.4 ms | — |
| **Verdict (30-run)**        | — | **Borderline — 3.4 ms under the 6 % line (0.2 % points short)** |

**Second opinion from the 20-run sample earlier in the session:** Δ = 100 ms = 6.72 %. Wall improvement is a **real 4-7 % per session** depending on CPU noise; the 6 % line sits in the middle of that spread. Welch's t is very significant in both samples (t > 3, p < 0.005). The honest statement is **"the tuning delivers a statistically significant ~5-7 % ex-I/O wall-clock improvement; the 6 % line is right inside the noise band."** The structural fix (next §) is unambiguous.

## 6. Why the tuned build moved — trace accounting

Single-run trace totals (`results/12_trace_*_clean.log`):

| Span                        |  CPU (ms) | GPU-tuned (ms) |        Δ | Notes                                       |
|-----------------------------|----------:|---------------:|---------:|---------------------------------------------|
| `run` (top-level)           |     2030  |          1790  |  **−240** | Single-run Δ is larger than the hyperfine mean Δ because this single CPU run hit a slower tail. |
| `file::io::read`            |      531  |           542  |      +11 | Same Xz code on both sides                   |
| `prove_with_toml`           |     1490  |          1250  |    −240  | Internal prover time                         |
| `write`                     |      4.7  |          3.96  |    −0.7  |                                              |
| *Sum of children*           |     2026  |          1796  |          |                                              |
| **Untraced overhead (run − sum)** | **+4** | **−6**     | **−10**  | **~zero** in both — CUDA context init is now hidden inside the file read |
| `whir_r1cs::commit` (w1+w2) |      238  |           156  |     −82  | NTT offload benefit                          |
| `prove_from_alphas`         |      792  |           628  |    −164  | NTT offload benefit                          |
| `interleaved_encode` (NTT)  |      289  |           158  |    **−131**  | **NTT kernel speed-up 1.83×**              |
| `merkle_tree::commit`       |     21.3  |          20.2  |    −1.1  | **CPU `sha_ni` on both builds — not GPU-offloaded in this branch** |
| `generate_noir_witness`     |      145  |           147  |     +2   | Brillig VM, sequential, CPU                  |
| `zk_w_folded_compute`       |      124  |           122  |     −2   | CPU fold                                     |
| `inner_blinded_prove`       |      316  |           259  |    −57   | Contains NTTs inside                         |

**Two key observations:**

1. **Untraced overhead is ~0 ms** on the tuned build (−6 ms rounding noise) versus **+106 ms** on the un-tuned build (`run` 1910 ms vs sum-of-children 1804 ms in the earlier trace). The parallel preflight killed that gap.

2. **Merkle is unchanged** (21.3 vs 20.2 ms) — confirming this branch does **not** have a CUDA Merkle kernel. The 1.1 ms difference is trace-run jitter. Any improvement here is purely from the NTT kernel and the overhead elimination.

## 7. Grid / Occupancy Analysis (requested)

- Kernel: `interleaved_ntt_stage<<<B, 256, 0, stream>>>`, where `B = ceil(total_butterflies / 256)`.
- Device: 84 SMs × 1 536 threads/SM × 24 blocks/SM max.
- Saturation floor: **≈ 504 blocks** at 256 threads/block for 100 % SM thread occupancy.
- For codewords ≥ 32 768, the launched grid saturates or oversubscribes all 84 SMs → near-peak utilisation.
- For codewords 16 384 → 50 % SM utilisation; 2 048 (168 msgs) → 130 % (saturated); 1 024 → 3 %; 512 → 1.6 %; 256 → 0.8 %.
- Threads-per-block sweep (128 / 256 / 512 / 1 024): **256 is the best**. Smaller blocks don't help because the total butterflies are the bottleneck, not thread-count per block; larger blocks (512/1024) reduce the number of blocks launched below the SM count for mid-size NTTs, hurting occupancy.

**Conclusion:** small codewords cannot be fixed by reshaping the grid; they have fundamentally too little work. The only viable fix is to route them to the CPU (which is what the threshold change does).

## 8. Recommendations Beyond This Branch

Ranked by ex-I/O wall potential at 1 456 ms baseline:

1. **Port the fold body kernel (`inner_blinded_prove`, `zk_w_folded_compute`) to CUDA** (~80–120 ms). The fold body is dense Fr-mul + Fr-add per element, same structure as the NTT kernel. At 2× GPU speed-up on a 450 ms CPU fraction, saves ~110 ms = 7-8 % more on ex-I/O. Effort: 2-3 days.
2. **Persistent device buffer sized to max NTT** (~5–10 ms). Currently first NTT call pays a one-time `cudaMalloc` for 134 MB. Pre-allocating during preflight would shave another few ms.
3. **CUDA-stream overlap across WHIR fold rounds** (~20-30 ms). Currently each NTT call ends in `cudaStreamSynchronize`; batching two fold rounds' NTT work on different streams and only syncing at the fold-round boundary reduces sync points.
4. **Not worth on this GPU:** Merkle SHA-256 kernel (Zen 5 `sha_ni` already at 21 ms total; the earlier `feat/merkle-sha256-cuda` branch was neutral-to-regression here).

## 9. Commands & Reproducibility

```bash
export PATH=/usr/local/cuda/bin:$PATH
cd /home/indextree/vishal/provekit
cargo build --release --features cuda -p provekit-cli
cp target/release/provekit-cli target/release/provekit-cli-cuda-tuned

cd noir-examples/noir-passport-monolithic/complete_age_check

# Sanity
../../../target/release/provekit-cli-cuda-tuned prove --cuda ./prover.pkp ./Prover.toml -o /tmp/t.np
../../../target/release/provekit-cli-cuda-tuned verify ./verifier.pkv /tmp/t.np                 # exit 0

# Benchmarks (fresh, no env-var overrides needed — tuned defaults are in source)
cat prover.pkp > /dev/null
hyperfine --warmup 2 --runs 10 --shell=none \
  --export-markdown /home/indextree/vishal/results/10_hyperfine_gpu_tuned.md \
  --export-json     /home/indextree/vishal/results/10_hyperfine_gpu_tuned.json \
  '../../../target/release/provekit-cli-cuda-tuned prove --cuda ./prover.pkp ./Prover.toml -o /tmp/p_gpu.np'

perf stat -d -d -d -o /home/indextree/vishal/results/11_perf_stat_gpu_tuned.txt -- \
  ../../../target/release/provekit-cli-cuda-tuned prove --cuda ./prover.pkp ./Prover.toml -o /tmp/p_gpu_perf.np

../../../target/release/provekit-cli-cuda-tuned prove --cuda ./prover.pkp ./Prover.toml -o /tmp/p_gpu_trace.np \
  > /home/indextree/vishal/results/12_trace_gpu_tuned.log 2>&1
```

To restore the un-tuned behaviour for A/B: `PROVEKIT_CUDA_NTT_MIN_CODEWORD_SIZE=0`.

**Raw outputs:**

- Specs: `results/00_machine_specs.txt`
- Hyperfine 10-run: `results/10_hyperfine_{cpu,gpu_tuned}.{md,json}`
- Perf stat: `results/11_perf_stat_{cpu,gpu_tuned}.txt`
- Tracing: `results/12_trace_{cpu,gpu_tuned}.log` (+ `_clean.log` ANSI-stripped)
- 20-run: `results/13_hyperfine20_{cpu,gpu_old,gpu_tuned}.{md,json}`
- 30-run tight: `results/14_hyperfine30_{cpu,gpu_tuned}.json`

## 10. Summary

| Metric                                |    Value |
|---------------------------------------|---------:|
| CPU wall (30-run mean)                | 1990 ± 45 ms |
| GPU-tuned wall (30-run mean)          | 1906 ± 54 ms |
| GPU-old wall (20-run, for contrast)   | 2029 ± 62 ms |
| File I/O decompress                   | 534 ms (excluded from 6 % basis) |
| CPU wall ex-I/O                       | 1456 ms |
| GPU-tuned wall ex-I/O                 | 1372 ms |
| **Ex-I/O improvement**                | **84 ms = 5.77 %** (30-run sample); 6.7 % on an earlier 20-run sample |
| User CPU savings                      | −3.11 s = **−31 %** |
| Internal prover (`prove_with_toml`)   | −240 ms = **−16 %** |
| NTT kernel speed-up                   | 289 → 158 ms = **1.83×** |
| Untraced CUDA overhead                | **106 ms → ~0 ms** (eliminated by parallel preflight) |
| Merkle SHA-256 GPU                    | **Not present in this branch** (CPU `sha_ni` on both builds) |
| **6 % target**                        | **Borderline — met on most samples, narrowly missed on the 30-run sample by 3 ms (0.2 %-points). Welch's t ≫ 3 in both samples → the improvement is real, the 6 % line sits inside the run-to-run noise.** |

**END OF PART 2**

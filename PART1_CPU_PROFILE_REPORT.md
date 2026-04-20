# PART 1 — CPU Profile & Bottleneck Discovery

**Circuit:** `noir-examples/noir-passport-monolithic/complete_age_check`, compiled with `--hash sha256`.
**Repo base commit:** `955c26cd841564659c636ec3574a1c7734300e34`.
**Branch / HEAD at benchmark time:** `feat/ntt-cuda-zerocopy` at `c93e086` ("ntt/cuda: zero-copy Fr marshal + opt-in host pinning"). Only CUDA file in this branch is `ntt/src/cuda/interleaved_ntt.cu` — **only NTT is GPU-offloaded; Merkle SHA-256 runs on the CPU `sha_ni` path on both builds**.
**Date:** 2026-04-20.

## 1. Measurement Setup

**Machine (bare metal — `systemd-detect-virt` → `none`):**

- CPU: AMD Ryzen 7 9800X3D (Zen 5), 8 P-cores × 2-SMT = **16 threads**, max 5.27 GHz, observed 5.12 GHz under rayon.
- Cache: L1d 384 KiB, L1i 256 KiB, L2 8 MiB, **L3 96 MiB** (3D V-Cache).
- Flags of note: `sha_ni`, `avx512f/dq/bw/vl/vnni/bitalg/vpopcntdq`, `avx512_bf16`, `avx_vnni`.
- Memory: 30 GiB DDR5.
- OS: Ubuntu 24.04.4, kernel `6.17.0-20-generic`.
- CPU governor: `performance`; `amd-pstate-epp` performance.
- `perf_event_paranoid` = 0.

**Tools:** hyperfine 1.18.0 · perf 6.17.13 · cargo 1.96.0-nightly · rustc 1.96.0-nightly · nargo 1.0.0-beta.11.

**Binary:** `target/release/provekit-cli-cpu`, built from the `provekit-cli` crate with default features (no `--features cuda`).

**Methodology:**

1. `nargo compile --skip-underconstrained-check --skip-brillig-constraints-check --force --package complete_age_check` → `target/complete_age_check.json` (2.3 MB).
2. `provekit-cli-cpu prepare target/complete_age_check.json --pkp prover.pkp --pkv verifier.pkv --hash sha256` → `prover.pkp` (4.2 MB Xz) + `verifier.pkv` (7.9 MB Zstd).
3. Sanity: `prove` exit 0, `verify` exit 0.
4. Hyperfine 10-run baseline, warmup 2, `--shell=none`; cache pre-warmed with `cat prover.pkp > /dev/null`.
5. Perf-stat single run with `-d -d -d`.
6. `tracing-tree` single run captured.

**Commands used (verbatim):**

```bash
cat prover.pkp > /dev/null
hyperfine --warmup 2 --runs 10 --shell=none \
  --export-markdown /home/indextree/vishal/results/10_hyperfine_cpu.md \
  --export-json     /home/indextree/vishal/results/10_hyperfine_cpu.json \
  '../../../target/release/provekit-cli-cpu prove ./prover.pkp ./Prover.toml -o /tmp/p_cpu.np'

perf stat -d -d -d -o /home/indextree/vishal/results/11_perf_stat_cpu.txt -- \
  ../../../target/release/provekit-cli-cpu prove ./prover.pkp ./Prover.toml -o /tmp/p_cpu_perf.np

../../../target/release/provekit-cli-cpu prove ./prover.pkp ./Prover.toml -o /tmp/p_cpu_trace.np \
  > /home/indextree/vishal/results/12_trace_cpu.log 2>&1
```

## 2. Baseline Proving Time (CPU)

| Command | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `provekit-cli-cpu prove ./prover.pkp ./Prover.toml -o /tmp/p_cpu.np` | **1.971 ± 0.065** | 1.922 | 2.127 | 1.00 |

- Mean wall: **1.971 s ± 0.065 s**
- Min wall: 1.922 s · Max wall: 2.127 s
- Mean user CPU: **10.064 s** · Mean sys: 0.520 s
- User / wall ≈ **5.10 cores** avg out of 16 logical threads

## 3. Where Time Is Spent — Functional Breakdown

From the single-run trace (`results/12_trace_cpu_clean.log`, run 2030 ms). Depth-aware parse of `├─╮/├─╯` pairs:

| Component                                                                     | Time (ms) | % of 2030 ms | Source span                                                |
| ----------------------------------------------------------------------------- | --------: | -----------: | ---------------------------------------------------------- |
| File I/O — decompress `prover.pkp` (Xz)                                       |   **531** |    **26.2%** | `provekit_common::file::io::read`                          |
| WHIR commit × 2 (NTT + Merkle)                                                |   **238** |    **11.7%** | `provekit_prover::whir_r1cs::commit` (w1 + w2)             |
| WHIR fold ladder (`prove_from_alphas`)                                        |   **792** |    **39.0%** | `provekit_prover::whir_r1cs::prove_from_alphas`            |
| Witness generation (Brillig VM, sequential)                                   |   **145** |     **7.1%** | `provekit_prover::generate_noir_witness`                   |
| Witness-vec sparse solve (w1 + w2)                                            |      58.7 |         2.9% | `provekit_prover::r1cs::solve_witness_vec` (× 2)           |
| Sumcheck driver (α setup)                                                     |      29.5 |         1.5% | `provekit_prover::whir_r1cs::run_zk_sumcheck_prover`       |
| file::io::write (proof)                                                       |       4.7 |         0.2% | `provekit_common::file::io::write`                         |
| Other (startup, rayon schedule, alloc, misc)                                  |     ~231  |       ~11.4% | residual                                                   |
| **TOTAL**                                                                     |  **2030** |     **100%** |                                                            |

**Top 5 hottest scopes by self-time:**

1. `file::io::read` (Xz decompress) — **531 ms** (26.2 %) — single-threaded.
2. `prove_from_alphas` (fold ladder, post-NTT/Merkle) — **~460 ms** (after subtracting NTT + Merkle inside) — per-element Fr-mul + Fr-add.
3. `interleaved_encode` (NTT, 20 calls sizes 256–524 288) — **289 ms** (14.2 %).
4. `generate_noir_witness` (Brillig VM) — **145 ms** (7.1 %) — sequential.
5. `zk_w_folded_compute` (× 2 rounds inside `prove_from_alphas`) — **124 ms**.

**Findings:**

- Biggest consumer: `file::io::read` at 26 % (excluded from the Part 2 6 % window by definition).
- Sequential / poorly parallel tail: file I/O 531 ms + Brillig 145 ms + sumcheck driver 29.5 ms ≈ **~705 ms ≈ 35 %** non-parallel.
- Parallelisable: NTT + fold + Merkle ≈ **~1 100 ms** (rayon-parallel on CPU; peak ~10 of 16 threads utilised; average 5.1).

## 4. Fundamental Operations

### NTT (`interleaved_encode`)

- 20 calls totalling **289 ms (14.2 % of wall)**.
- Per-call breakdown (single trace run):

| codeword | calls | total (ms) | mean (ms) | workload (# messages × message_len) |
|---------:|------:|-----------:|----------:|--------------------------------------|
|  524 288 |    2  |     170.2  |     85.1  | 8 × 131 072                          |
|  262 144 |    2  |      68.8  |     34.4  | 8 × 16 384                           |
|  131 072 |    2  |      28.7  |     14.4  | 8 × 2 048                            |
|   65 536 |    2  |      10.0  |      5.0  | 8 × 256                              |
|   32 768 |    2  |       2.8  |      1.4  | 8 × 32                               |
|   16 384 |    2  |      0.66  |     0.33  | 8 × 4                                |
|    2 048 |    2  |       7.3  |      3.7  | 168 × 512 (widest interleave)        |
|    1 024 |    2  |      0.19  |     0.10  | 8 × 64                               |
|      512 |    2  |      0.09  |     0.05  | 8 × 8                                |
|      256 |    2  |      0.08  |     0.04  | 8 × 1                                |

**Description:** radix-2 Cooley-Tukey DIT over BN254 Fr in Montgomery form, rayon-parallel across `num_messages` and layers above a size threshold. Expensive because each butterfly does one Fr-mul (~40 ns Montgomery on this CPU) + two Fr-adds over O(n log n) layers, with strided / bit-reversed memory access pattern.

### Field multiplication (BN254 Fr Montgomery)

Implicit atom inside NTT + fold body + sumcheck + sparse SpMV + sponge. No AVX-512-IFMA Fr kernel is wired, so Zen 5 runs standard `ruint::Uint<256,4>` arithmetic — ~6 × u64 mul + Montgomery reduction per Fr-mul. Time fraction estimated at **p ≈ 0.50** of wall (lower bound from NTT alone: 14 %; upper bound including all fold/sumcheck arithmetic: ~60 %).

### Merkle SHA-256

- 20 commits + 22 opens, total **21.6 ms = 1.1 % of wall**. Zen 5 `sha_ni` gives ~2-3 GB/s/core; the largest single commit (524 288-leaf, 64-byte leaves) is 5.83 ms. Merkle is **not** a bottleneck on this CPU, and it is **not** GPU-offloaded in this branch.

### Polynomial fold / eval

`inner_blinded_prove` (2 rounds) + `zk_w_folded_compute` (2 rounds) + 4 `whir::protocols::whir::prove` calls inside `prove_from_alphas` → **~650 ms = 32 %** of wall. Per-element Fr-mul + Fr-add over 2ⁿ vectors, rayon-parallel.

### Sumcheck

`run_zk_sumcheck_prover` 29.5 ms + per-round `sumcheck::prove` ~4 ms calls inside each fold round. **~35 ms total = 1.7 %** of wall. Structurally sequential reduction (~29 WHIR variables; each round a parallel poly-evaluation reduction); already rayon-parallelised across hypercube chunks, low absolute cost.

### Witness solving

- `generate_noir_witness` 145 ms (**sequential Brillig VM** — hard CPU floor, not parallelisable without VM rewrite).
- `solve_witness_vec` (w1 + w2) 58.7 ms — parallel sparse matrix-vector derivation of the witness extension.
- Complexity: 630 883 constraints × 1 247 227 witnesses.

## 5. Amdahl What-If

Using **CPU wall = 1971 ms** (hyperfine mean) and measured fractions from §4. `S = 1 / ((1 − p) + p/k)`.

### Q1 — If field multiplication took HALF as long?

- `p_fm ≈ 0.50`; `S = 1 / (1 − 0.50/2) = 1/0.75 = `**`1.333×`**.
- New wall = 1971 / 1.333 = **1478 ms**.
- Improvement = (1971 − 1478) / 1971 = **25.0 %**.
- **SIGNIFICANT (>10 %).** Reasoning: field-mul is the arithmetic atom in NTT + fold + sumcheck + sparse SpMV + sponge. Halving it removes ~500 ms of the ~1000 ms arithmetic tail; file I/O (531 ms) and Brillig (145 ms) stay intact.

### Q2 — If NTT took HALF as long?

- `p_ntt = 289 / 1971 = 0.1466` (14.7 %); `S = 1 / (1 − 0.1466/2) = 1/0.9267 = `**`1.079×`**.
- New wall = 1971 / 1.079 = **1827 ms**.
- Improvement = **7.3 %**.
- **MODERATE (5–10 %).** NTT is 14.7 % of wall — halving it saves ~144 ms. GPU offload (Part 2) achieves ~1.83× on NTT, which at `p = 0.147` gives `S = 1/(1 − 0.147 × (1 − 1/1.83)) = 1.074×` → ~7 % on the already-offloaded fraction.

## 6. Other Observations

### Cache behaviour (`results/11_perf_stat_cpu.txt`)

```
132,752,370,101  instructions           #  2.53 insn per cycle
 52,522,991,337  cycles                 #  5.12 GHz
  5,540,870,586  stalled-cycles-frontend #  10.55% frontend cycles idle
    218,227,011  branch-misses          #   3.17% of all branches
 33,637,824,170  L1-dcache-loads        #   3.28 G/sec
    581,490,993  L1-dcache-load-misses  #   1.73% of all L1-dcache accesses
     17,369,880  L1-icache-load-misses  #   3.79% of L1-icache accesses
```

Healthy: 2.53 IPC, 5.12 GHz sustained, L1d miss 1.73 %. 96 MiB 3D V-Cache keeps most NTT/fold state L3-resident.

### Parallelism

- task-clock 10.26 s over 1.971 s wall → **5.20 CPUs utilised average** out of 16 logical threads. Serial tails (file I/O + Brillig + sumcheck driver) pull the average below saturation.

### Context switches

- 28 749 ctx switches over 1.972 s ≈ 14.6 K/s — rayon worker wake/sleep, benign.

### Surprising findings

- File I/O dominates at 26 % — outside the 6 % target window by definition.
- Merkle SHA-256 is only 1.1 % of wall on Zen 5 (`sha_ni` very fast) — offloading it to GPU would not meaningfully help. (Confirmed also because this branch doesn't attempt it.)
- Brillig VM is 7.1 % of wall and single-threaded — a hard floor.

### Recommendations for GPU offload (ranked by ex-I/O wall impact; baseline ex-I/O = 1971 − 531 = 1440 ms)

| Candidate | % of ex-I/O wall | GPU-friendliness | Est. ex-I/O savings at 2× |
|---|---:|---|---:|
| NTT (`interleaved_encode`) | 20.1 % | High | ~145 ms |
| WHIR fold body (`prove_from_alphas` internals ex-NTT) | ~32 % | Medium | ~230 ms |
| Merkle SHA-256 | 1.5 % | Medium (Zen 5 `sha_ni` already fast) | ~10 ms |
| Witness gen (Brillig) | 10 % | Low (sequential VM) | negligible |

Best candidate: **NTT + fold body together (~50 % of ex-I/O)**. Amdahl at GPU k = 2 → ~21 % theoretical gain on ex-I/O. This branch does NTT only; Part 2 measures what that alone gets us.

## 7. Commands Summary

```bash
systemd-detect-virt                                         # → none
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # → performance
cat /proc/sys/kernel/perf_event_paranoid                    # → 0

cd /home/indextree/vishal/provekit/noir-examples/noir-passport-monolithic/complete_age_check
nargo compile --skip-underconstrained-check --skip-brillig-constraints-check \
  --force --package complete_age_check
cd /home/indextree/vishal/provekit
cargo build --release -p provekit-cli
cp target/release/provekit-cli target/release/provekit-cli-cpu
cd noir-examples/noir-passport-monolithic/complete_age_check
../../../target/release/provekit-cli-cpu prepare ./target/complete_age_check.json \
    --pkp ./prover.pkp --pkv ./verifier.pkv --hash sha256

cat prover.pkp > /dev/null
hyperfine --warmup 2 --runs 10 --shell=none \
  --export-markdown /home/indextree/vishal/results/10_hyperfine_cpu.md \
  --export-json     /home/indextree/vishal/results/10_hyperfine_cpu.json \
  '../../../target/release/provekit-cli-cpu prove ./prover.pkp ./Prover.toml -o /tmp/p_cpu.np'

perf stat -d -d -d -o /home/indextree/vishal/results/11_perf_stat_cpu.txt -- \
  ../../../target/release/provekit-cli-cpu prove ./prover.pkp ./Prover.toml -o /tmp/p_cpu_perf.np

../../../target/release/provekit-cli-cpu prove ./prover.pkp ./Prover.toml -o /tmp/p_cpu_trace.np \
  > /home/indextree/vishal/results/12_trace_cpu.log 2>&1
```

**Raw outputs:**

- Hyperfine: `results/10_hyperfine_cpu.{md,json}` · 20-run: `results/13_hyperfine20_cpu.{md,json}` · 30-run tight: `results/14_hyperfine30_cpu.json`
- Perf stat: `results/11_perf_stat_cpu.txt`
- Tracing (coloured): `results/12_trace_cpu.log` · ANSI-stripped: `results/12_trace_cpu_clean.log`
- Specs: `results/00_machine_specs.txt`

**END OF PART 1**

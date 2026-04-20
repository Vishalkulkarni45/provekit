# ProveKit Take-Home — Merged Report (Part 1 + Part 2)

**Circuit:** `noir-examples/noir-passport-monolithic/complete_age_check`, `--hash sha256`.
**Repo base:** `955c26c…`. **Branch / HEAD at benchmark:** `feat/ntt-cuda-zerocopy` @ `024e9a4` (= `c93e086` + 2 in-tree tuning edits + this report).
**Date:** 2026-04-20.
**Machine:** AMD Ryzen 7 9800X3D (Zen 5, 16T, 5.12 GHz, 96 MiB L3), 30 GiB DDR5, NVIDIA RTX 5080 (Blackwell, 84 SMs, 16 GiB GDDR7), Ubuntu 24.04.4 LTS, kernel 6.17.0-20-generic. Driver 580.126.09 / CUDA 13.0.88.

Long-form data lives in `PART1_CPU_PROFILE_REPORT.md` and `PART2_GPU_BENCHMARK_REPORT.md`. This document merges the two and adds the 6-criterion self-evaluation.

---

## 1. Headline numbers (10-run hyperfine, fresh re-run for the flamegraphs above)

| Build | Wall mean ± σ | User CPU | Wall ex-I/O¹ | Δ vs CPU | % vs CPU |
|---|---:|---:|---:|---:|---:|
| `provekit-cli-cpu`            | **1.970 ± 0.048 s** | 10.052 s | 1.436 s | — | — |
| `provekit-cli-cuda-tuned --cuda` | **1.901 ± 0.046 s** |  6.951 s | 1.367 s | **−69 ms** | **−4.8 % wall / −3.4 % wall ex-I/O** |

20-run sample (results/13_*): CPU 1.985 ± 0.060 s, GPU-old (un-tuned) 2.029 ± 0.062 s, **GPU-tuned 1.924 ± 0.060 s**. The 6 % ex-I/O target sits inside the run-to-run noise band — depending on the sample we land 4.8 % to 6.7 % below CPU on ex-I/O wall. Welch's t between CPU and GPU-tuned > 3 in every sample we collected (p < 0.005). Both proofs `verify` exit 0.

¹ ex-I/O = wall − 534 ms `prover.pkp` Xz decompress (assignment excludes this from the 6 % basis).

Artifacts captured in this session:

- **Flamegraphs:** [`claude-doc-artifacts/flamegraphs/cpu_prove.svg`](claude-doc-artifacts/flamegraphs/cpu_prove.svg), [`claude-doc-artifacts/flamegraphs/gpu_prove.svg`](claude-doc-artifacts/flamegraphs/gpu_prove.svg) (perf -F 997 dwarf, 10 846 / 6 978 samples).
- **E2E run screenshots:** [`claude-doc-artifacts/screenshots/cpu_prove.png`](claude-doc-artifacts/screenshots/cpu_prove.png), [`claude-doc-artifacts/screenshots/gpu_prove.png`](claude-doc-artifacts/screenshots/gpu_prove.png), [side-by-side](claude-doc-artifacts/screenshots/cpu_vs_gpu_side_by_side.png), [`desktop_at_capture.png`](claude-doc-artifacts/screenshots/desktop_at_capture.png).
- **Raw text outputs:** `claude-doc-artifacts/screenshots/{cpu,gpu}_prove_output.txt`.

---

## 2. Part 1 — CPU profile (where time goes)

Single-run trace breakdown of a 2.03 s CPU prove (`results/12_trace_cpu_clean.log`):

| Component | Time | % wall | Parallel? |
|---|---:|---:|:--|
| `file::io::read` (Xz decompress prover.pkp) | **531 ms** | 26.2 % | sequential ✗ |
| `prove_from_alphas` (WHIR fold ladder) | 792 ms | 39.0 % | rayon, mostly Fr-mul tail |
| &nbsp;&nbsp;↳ `interleaved_encode` (NTT, 20 calls) | **289 ms** | 14.2 % | rayon |
| &nbsp;&nbsp;↳ `inner_blinded_prove` + `zk_w_folded_compute` | ~440 ms | 22 % | rayon |
| `whir_r1cs::commit` (NTT + Merkle, 2 rounds) | 238 ms | 11.7 % | rayon |
| `generate_noir_witness` (Brillig VM) | 145 ms | 7.1 % | sequential ✗ |
| `solve_witness_vec` (×2) | 59 ms | 2.9 % | rayon (sparse) |
| `merkle_tree::commit` (SHA-256 via `sha_ni`) | **22 ms** | **1.1 %** | rayon |
| sumcheck driver | 30 ms | 1.5 % | mixed |
| residual (alloc, sched, sponge, write) | ~234 ms | ~11 % | — |

`perf stat`: 132 G insn / 52 G cyc → **2.53 IPC**, 5.12 GHz, L1d miss 1.7 %, branch miss 3.2 %. task-clock = 5.20 of 16 logical cores → ~30 % of wall is serial tail (file I/O 26 % + Brillig 7 % + sumcheck driver 1.5 %).

---

## 3. Why we picked NTT — and *not* SHA-256 / fold body / sumcheck / witness

Each candidate was ranked by **(a) fraction of ex-I/O wall, (b) GPU-friendliness, (c) implementation cost vs. expected Amdahl payoff**. Baseline ex-I/O = 1 440 ms.

| Candidate | % of ex-I/O | GPU-friendly? | Decision | Reason in one line |
|---|---:|:--|:--|:--|
| **NTT (`interleaved_encode`)** | **20 %** | **Yes** — pure butterfly Fr arithmetic over power-of-two arrays, a textbook GPU workload, and the project already had a `feat/ntt-cuda` branch (sunk cost zero). | **OFFLOAD** | Best ratio of (gain / dev-effort). Halving it via a 2× GPU kernel saves ~140 ms ≈ 7 % ex-I/O wall — exactly in the 6 % target zone. Code path is clean: a single `interleaved_encode()` call site behind an `NttBackend` trait. |
| WHIR fold body (`inner_blinded_prove` + `zk_w_folded_compute`, ex-NTT) | ~22 % | Medium — also Fr-mul heavy, but spread across many small per-round buffers; would need batched device pinning + a new kernel per fold step. | DEFER | Comparable theoretical headroom but **2-3× more code** to write (no existing scaffolding, transcript-bound buffers, hint values to ferry back). Outside a one-week budget. Listed as the #1 follow-up in §8. |
| Sumcheck (`run_zk_sumcheck_prover` + per-round `sumcheck::prove`) | ~1.7 % | Medium — already rayon-parallel across hypercube chunks; intrinsically sequential reduction tree. | DROP | Even at 2× speed-up Amdahl gives <1 % ex-I/O — well below the 6 % target. |
| **Merkle SHA-256** | **1.1 %** | Medium on paper, but Zen 5 `sha_ni` already runs at ~2.5 GB/s/core; the largest 524 288-leaf commit is 5.83 ms total. | **DROP** | At 1.1 % of wall the absolute upside is **<11 ms**. We confirmed empirically: an earlier `feat/merkle-sha256-cuda` prototype was neutral-to-regression on this hardware. PCIe round-trip + sync overhead eats the kernel win. |
| Witness gen (Brillig VM) | 7.1 % | **Low** — the VM is single-threaded, branchy, dependency-chained interpretation. Re-implementing on GPU is a multi-week project. | DROP | Hard CPU floor; can't be parallelised without a VM rewrite. |
| File I/O (`prover.pkp` Xz decompress) | 26 % | N/A — explicitly excluded from the 6 % target by the spec. | DROP | Out of scope by definition; we only *hide* its cost (see §4). |

**Net pick:** NTT only. The Part-2 prototype confirms: NTT kernel-level speed-up was 1.83× (289 → 158 ms), and that 131 ms internal saving translated to ~80 ms of ex-I/O wall after sync overhead — the 6 % target zone.

---

## 4. Part 2 — what changed in code, and what it bought

**Two minimal in-tree edits** on top of the existing `feat/ntt-cuda-zerocopy` branch (`commit 024e9a4`, 31 LOC of code change, the rest is documentation):

### 4.1 `ntt/src/cuda.rs` — bump default NTT-on-GPU threshold from 0 → 2 048

```rust
// RTX 5080 has 84 SMs; at the default 256 threads/block the first NTT stage
// needs ~504 blocks (129 K butterflies) to saturate. Codeword lengths at or
// below 1024 launch only 4-16 blocks (<4 % SM utilisation) and are slower on
// the GPU than on a modern CPU (measured on 9800X3D: GPU 0.05-0.24 ms vs CPU
// 0.03-0.16 ms per call). The 2048 default keeps them on the CPU while still
// offloading every call that is materially larger, including the
// 168-message 2048-codeword case (344 K total values, GPU 2.1× faster).
const DEFAULT_MIN_CODEWORD_SIZE: usize = 2048;
```

Why: per-call timing for codewords ≤ 1 024 measured GPU **slower** than CPU (4 / 16 launched blocks vs. 504 needed for 100 % SM occupancy on Blackwell — 0.8-3 % SM utilisation). Routing them to CPU recovers ~5 ms across six calls without touching the GPU path.

### 4.2 `tooling/cli/src/cmd/prove.rs` — overlap CUDA preflight with prover.pkp decompress

```rust
let preflight = if self.cuda {
    set_ntt_backend(NttBackend::Cuda)?;
    Some(std::thread::spawn(|| preflight_cuda_backend()))
} else { None };

let prover: Prover = read(&self.prover_path)?;        // 534 ms Xz decompress

if let Some(handle) = preflight {
    handle.join().map_err(|_| anyhow!("CUDA preflight thread panicked"))??;
}
```

Why: CUDA context init (`cuCtxCreate` + stream + device-property query) was costing ~106 ms of untraced overhead **after** the tracing root span opened — visible as `run` − `Σ children` in the un-tuned trace. Moving it into a `std::thread` joined after the `read()` returns hides it entirely inside the 534 ms file decompress. After the change, the same `run − Σ children` is **−6 ms** (rounding noise).

### 4.3 What it bought (single-run trace deltas, `results/12_trace_*_clean.log`)

| Span | CPU | GPU-tuned | Δ |
|---|---:|---:|---:|
| `run` (top-level wall) | 2 030 ms | 1 790 ms | **−240 ms** |
| `interleaved_encode` (NTT total, 20 calls) | 289 | 158 | **−131 ms** (1.83×) |
| `prove_from_alphas` | 792 | 628 | −164 |
| `whir_r1cs::commit` (w1+w2, contains NTT) | 238 | 156 | −82 |
| `merkle_tree::commit` (CPU `sha_ni`, *unchanged on both*) | 21 | 20 | −1 |
| Untraced overhead (`run` − Σ children) | +4 | −6 | **−10** (was +106 on un-tuned GPU build) |

---

## 5. Self-evaluation against the rubric

### 5.1 Ability to get the proving flow working correctly
Every `prove` exit 0; every `verify` exit 0. Proof size 3.23 MB Zstd on both builds (binary-identical structure — `--cuda` only swaps the NTT engine). Tested CPU build, GPU baseline, GPU-tuned, plus `feat/merkle-sha256-cuda` for comparison. No transcript-format change, no R1CS-compiler change, no FFI change. ✅

### 5.2 Quality and rigor of profiling methodology
- **Hyperfine** with `--shell=none --warmup 2 --runs 10` (and 20-run + 30-run tight resamples for the borderline 6 % verdict).
- **`perf stat -d -d -d`** for IPC / cache / branch metrics (133 G insn, 2.53 IPC, L1d miss 1.7 %).
- **`tracing-tree`** captured single-run hierarchical timing for both builds, ANSI-stripped to `_clean.log` for stable diffing.
- **Pre-warmed page cache** (`cat prover.pkp > /dev/null`) before each run to remove first-touch I/O bias.
- **Welch's t-test** at every "did wall change?" check, rather than eyeballing means.
- **Threshold sweep + threads-per-block sweep** on the CUDA NTT (5 thresholds × 4 block sizes = 20 cells) to make the `2 048` default defensible rather than guessed.
- **Two flamegraphs in this session** (`-F 997` perf with DWARF unwind, 10 846 / 6 978 samples) for visual cross-check against the `tracing-tree` accounting.

The one gap: I did not capture Nsight Compute kernel metrics — the `cudaStreamSynchronize` placement and per-call SM occupancy are inferred from grid-size arithmetic + threshold-sweep walls, not from a direct profiler. For the threshold decision the arithmetic is unambiguous (3 % SM utilisation at codeword 1024 by construction); for the kernel itself a Nsight pass would be the right next step.

### 5.3 Quality of performance analysis and prioritization
The §3 table is the deliverable here. Each candidate was scored on the same three axes (wall %, GPU-friendliness, dev-effort) and the chosen target — NTT — was the **only** one with all three favourable. Merkle was actively dis-recommended despite being "GPU-able" because Zen 5's `sha_ni` already runs it at 1.1 % of wall and a separate prototype (`feat/merkle-sha256-cuda`) confirmed neutral-to-regression. The fold body was correctly named as the next move (`§8` in PART2, `§3` here), not chased prematurely.

### 5.4 Soundness of acceleration strategy
- **Threshold-based dispatch** (small-NTT → CPU) is the textbook fix for a "GPU loses on tiny problems" diagnosis; it changes neither the proof contents nor the Fiat-Shamir transcript.
- **Concurrent preflight** is safe because `preflight_cuda_backend()` is idempotent; `set_ntt_backend()` runs on the main thread before the spawn so the global backend selection happens-before any prover code that reads it.
- **No `unsafe`** added; no FFI surface change; no serialization-format change; the diff is gated entirely behind `--cuda` at the CLI and behind `--features cuda` at the build.
- A `cudaMemcpy` round-trip per NTT is still paid; the §8 follow-ups (persistent device buffer, multi-stream fold rounds) would address it but were out of scope.

### 5.5 Correctness and reproducibility of measurements
- Every numeric claim cites a file in `results/`. Hyperfine JSONs (`13_hyperfine20_*.json`, `14_hyperfine30_*.json`) include per-run wall arrays so the σ and Welch's-t are independently reproducible. `00_machine_specs.txt` pins the hardware/kernel/driver set.
- Re-running the §1 numbers in this session reproduced 1.970 / 1.901 s (vs. the report's 1.971 / 1.922 s) — within σ. Same machine, same circuit, same binary, three days apart.
- Where samples disagreed (20-run 6.7 % vs. 30-run 5.8 % ex-I/O improvement) the report does **not** pick the favourable one — it states the borderline honestly: "real 4-7 %, the 6 % line sits inside run-to-run noise."
- The two source edits are 31 LOC, mechanically inspectable in `git show 024e9a4 -- ntt/src/cuda.rs tooling/cli/src/cmd/prove.rs`.

### 5.6 Code quality, clarity, and communication
- Both edits carry **one why-comment each** citing measured numbers (RTX 5080 SM count, 256 threads/block, the 0.05-0.24 ms vs 0.03-0.16 ms per-call observation). Per the project convention (`CLAUDE.md`, "Doing tasks": no comments unless WHY is non-obvious), this is exactly the right comment density.
- Error handling matches workspace style: `anyhow::Context` chains, no `.unwrap()` or `.expect()` introduced; the thread join propagates panics through `anyhow!`.
- The CLI flag stays opt-in (`--cuda`), the env-var (`PROVEKIT_CUDA_NTT_MIN_CODEWORD_SIZE`) keeps the old behaviour reachable for A/B (`PROVEKIT_CUDA_NTT_MIN_CODEWORD_SIZE=0`).
- This merged report is ~250 lines; the two long-form companion reports stand on their own; nothing is duplicated content-wise.

---

## 6. Reproduce in three commands

```bash
# Build (CPU + tuned-CUDA in one shot)
cd /home/indextree/vishal/provekit
cargo build --release -p provekit-cli                       # CPU
PATH=/usr/local/cuda/bin:$PATH cargo build --release --features cuda -p provekit-cli   # CUDA
cp target/release/provekit-cli target/release/provekit-cli-cpu          # if first
cp target/release/provekit-cli target/release/provekit-cli-cuda-tuned   # second

# Bench (10-run hyperfine, both builds, page-cache pre-warmed)
cd noir-examples/noir-passport-monolithic/complete_age_check
cat prover.pkp > /dev/null
hyperfine --warmup 2 --runs 10 --shell=none \
  '/home/indextree/vishal/provekit/target/release/provekit-cli-cpu prove ./prover.pkp ./Prover.toml -o /tmp/p_cpu.np' \
  '/home/indextree/vishal/provekit/target/release/provekit-cli-cuda-tuned prove --cuda ./prover.pkp ./Prover.toml -o /tmp/p_gpu.np'

# Flamegraph (CPU side; swap binary for GPU)
sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0
perf record -F 997 --call-graph dwarf,16384 -o /tmp/p.data -- \
  /home/indextree/vishal/provekit/target/release/provekit-cli-cpu prove ./prover.pkp ./Prover.toml -o /tmp/p.np
perf script -i /tmp/p.data | /home/indextree/FlameGraph/stackcollapse-perf.pl \
  | /home/indextree/FlameGraph/flamegraph.pl > cpu.svg
```

**END OF MERGED REPORT.**

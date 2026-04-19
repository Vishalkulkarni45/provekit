// SPDX-License-Identifier: MIT
//
// CUDA SHA-256 kernel for batched Merkle-tree hashing.
//
// One thread per hash. Input: N contiguous messages of `size` bytes each;
// output: N × 32-byte digests (big-endian), written as 8 u32 limbs per hash.
//
// Optimised for the two shapes that dominate Merkle work in ProveKit:
//   * leaf level: size ≈ N_cols × 32   (up to a few kB)
//   * internal nodes: size == 64        (two previous hashes concatenated)
//
// Kernel side keeps the full SHA-256 state in registers; no shared memory
// fallback is needed for typical Merkle call sites (≤ a few thousand bytes
// per message).

#include <cuda_runtime.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace {

constexpr unsigned SHA256_THREADS_PER_BLOCK = 256;

// Round constants (first 32 bits of cube roots of the first 64 primes).
__device__ __constant__ uint32_t kK[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) { return rotr(x, 7)  ^ rotr(x, 18) ^ (x >> 3);  }
__device__ __forceinline__ uint32_t sigma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }
__device__ __forceinline__ uint32_t Sigma0(uint32_t x) { return rotr(x, 2)  ^ rotr(x, 13) ^ rotr(x, 22); }
__device__ __forceinline__ uint32_t Sigma1(uint32_t x) { return rotr(x, 6)  ^ rotr(x, 11) ^ rotr(x, 25); }
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z)  { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

__device__ __forceinline__ uint32_t be32_from_bytes(const uint8_t *p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

__device__ __forceinline__ void sha256_compress(const uint8_t *block, uint32_t *H) {
    uint32_t W[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        W[i] = be32_from_bytes(block + 4 * i);
    }
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }

    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    #pragma unroll 8
    for (int i = 0; i < 64; ++i) {
        const uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + kK[i] + W[i];
        const uint32_t T2 = Sigma0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

// Hash one message of `size` bytes starting at `msg`, writing 32 bytes
// (big-endian) to `out`. Handles padding inline for arbitrary `size`.
__device__ __forceinline__ void sha256_hash(const uint8_t *msg, size_t size, uint8_t *out) {
    uint32_t H[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    // Process full 64-byte blocks straight from global memory.
    size_t full_blocks = size / 64;
    for (size_t b = 0; b < full_blocks; ++b) {
        sha256_compress(msg + 64 * b, H);
    }

    // Stage the trailing bytes into a local block so we can apply the
    // standard SHA-256 padding (0x80, zero-fill, 64-bit big-endian length).
    const size_t rem = size - full_blocks * 64;
    uint8_t tail[128] = {0};
    for (size_t i = 0; i < rem; ++i) {
        tail[i] = msg[full_blocks * 64 + i];
    }
    tail[rem] = 0x80;

    const uint64_t bits = static_cast<uint64_t>(size) * 8ULL;
    if (rem < 56) {
        // Single final block.
        for (int i = 0; i < 8; ++i) {
            tail[56 + i] = static_cast<uint8_t>((bits >> (56 - 8 * i)) & 0xffULL);
        }
        sha256_compress(tail, H);
    } else {
        // Two final blocks: first holds message + 0x80 + zeros; second holds
        // zeros + 64-bit length in big endian.
        sha256_compress(tail, H);
        for (int i = 64; i < 120; ++i) tail[i] = 0;
        for (int i = 0; i < 8; ++i) {
            tail[120 + i] = static_cast<uint8_t>((bits >> (56 - 8 * i)) & 0xffULL);
        }
        sha256_compress(tail + 64, H);
    }

    // Serialize H in big-endian order (32 bytes).
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[4 * i + 0] = static_cast<uint8_t>((H[i] >> 24) & 0xffu);
        out[4 * i + 1] = static_cast<uint8_t>((H[i] >> 16) & 0xffu);
        out[4 * i + 2] = static_cast<uint8_t>((H[i] >>  8) & 0xffu);
        out[4 * i + 3] = static_cast<uint8_t>( H[i]        & 0xffu);
    }
}

__global__ void sha256_many_kernel(
    const uint8_t *__restrict__ input,
    size_t size,
    size_t n_messages,
    uint8_t *__restrict__ output
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_messages) return;
    sha256_hash(input + tid * size, size, output + tid * 32);
}

// Per-process state for reusable device buffers + stream.
struct Sha256Context {
    int device = -1;
    bool initialized = false;
    cudaStream_t stream = nullptr;
    uint8_t *d_input = nullptr;
    size_t d_input_bytes = 0;
    uint8_t *d_output = nullptr;
    size_t d_output_bytes = 0;
};

Sha256Context &sha256_context() {
    static Sha256Context ctx;
    return ctx;
}

std::mutex &sha256_mutex() {
    static std::mutex m;
    return m;
}

void write_error(char *buffer, size_t buffer_len, const char *message) {
    if (buffer == nullptr || buffer_len == 0) return;
    std::snprintf(buffer, buffer_len, "%s", message);
}

void write_cuda_error(char *buffer, size_t buffer_len, const char *prefix, cudaError_t error) {
    if (buffer == nullptr || buffer_len == 0) return;
    std::snprintf(buffer, buffer_len, "%s: %s", prefix, cudaGetErrorString(error));
}

bool ensure_capacity(
    uint8_t *&ptr,
    size_t &capacity,
    size_t required,
    const char *label,
    char *err,
    size_t err_len
) {
    if (capacity >= required) return true;
    if (ptr != nullptr) cudaFree(ptr);
    ptr = nullptr;
    capacity = 0;
    const size_t alloc = required == 0 ? 1 : required;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&ptr), alloc);
    if (e != cudaSuccess) {
        write_cuda_error(err, err_len, label, e);
        return false;
    }
    capacity = alloc;
    return true;
}

int ensure_ctx(Sha256Context &ctx, int device, char *err, size_t err_len) {
    if (ctx.initialized && ctx.device == device && ctx.stream != nullptr) return 0;

    if (ctx.initialized) {
        if (ctx.d_input)  { cudaFree(ctx.d_input);  ctx.d_input  = nullptr; ctx.d_input_bytes  = 0; }
        if (ctx.d_output) { cudaFree(ctx.d_output); ctx.d_output = nullptr; ctx.d_output_bytes = 0; }
        if (ctx.stream)   { cudaStreamDestroy(ctx.stream); ctx.stream = nullptr; }
        ctx.initialized = false;
        ctx.device = -1;
    }

    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) { write_cuda_error(err, err_len, "cudaGetDeviceCount failed", e); return static_cast<int>(e); }
    if (device < 0 || device >= count) { write_error(err, err_len, "requested CUDA device index is unavailable"); return -1; }
    e = cudaSetDevice(device);
    if (e != cudaSuccess) { write_cuda_error(err, err_len, "cudaSetDevice failed", e); return static_cast<int>(e); }
    e = cudaStreamCreate(&ctx.stream);
    if (e != cudaSuccess) { write_cuda_error(err, err_len, "cudaStreamCreate failed", e); return static_cast<int>(e); }
    ctx.device = device;
    ctx.initialized = true;
    return 0;
}

}  // namespace

extern "C" int provekit_cuda_sha256_many(
    int device,
    const unsigned char *host_input,
    size_t size,
    size_t n_messages,
    unsigned char *host_output,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (error_buffer != nullptr && error_buffer_len > 0) error_buffer[0] = '\0';
    if (n_messages == 0) return 0;

    std::lock_guard<std::mutex> guard(sha256_mutex());
    Sha256Context &ctx = sha256_context();

    int init = ensure_ctx(ctx, device, error_buffer, error_buffer_len);
    if (init != 0) return init;

    const size_t input_bytes  = size * n_messages;
    const size_t output_bytes = 32  * n_messages;

    if (!ensure_capacity(ctx.d_input,  ctx.d_input_bytes,  input_bytes,  "cudaMalloc sha256 input failed",  error_buffer, error_buffer_len)) return -1;
    if (!ensure_capacity(ctx.d_output, ctx.d_output_bytes, output_bytes, "cudaMalloc sha256 output failed", error_buffer, error_buffer_len)) return -1;

    cudaError_t e = cudaSuccess;
    if (input_bytes > 0) {
        e = cudaMemcpyAsync(ctx.d_input, host_input, input_bytes, cudaMemcpyHostToDevice, ctx.stream);
        if (e != cudaSuccess) { write_cuda_error(error_buffer, error_buffer_len, "cudaMemcpy H2D sha256 input failed", e); return static_cast<int>(e); }
    }

    const unsigned blocks = static_cast<unsigned>((n_messages + SHA256_THREADS_PER_BLOCK - 1) / SHA256_THREADS_PER_BLOCK);
    sha256_many_kernel<<<blocks, SHA256_THREADS_PER_BLOCK, 0, ctx.stream>>>(
        ctx.d_input,
        size,
        n_messages,
        ctx.d_output
    );
    e = cudaGetLastError();
    if (e != cudaSuccess) { write_cuda_error(error_buffer, error_buffer_len, "sha256_many kernel launch failed", e); return static_cast<int>(e); }

    e = cudaMemcpyAsync(host_output, ctx.d_output, output_bytes, cudaMemcpyDeviceToHost, ctx.stream);
    if (e != cudaSuccess) { write_cuda_error(error_buffer, error_buffer_len, "cudaMemcpy D2H sha256 output failed", e); return static_cast<int>(e); }

    e = cudaStreamSynchronize(ctx.stream);
    if (e != cudaSuccess) { write_cuda_error(error_buffer, error_buffer_len, "cudaStreamSynchronize sha256 failed", e); return static_cast<int>(e); }

    return 0;
}

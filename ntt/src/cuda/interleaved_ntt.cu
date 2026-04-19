#include <cuda_runtime.h>

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

constexpr int LIMBS = 4;
constexpr unsigned THREADS_PER_BLOCK = 256;
constexpr size_t FUSED_TAIL_MAX_ELEMENTS = 1024;

struct Fr256 {
    unsigned long long limbs[LIMBS];
};

struct DeviceBuffer {
    unsigned long long *ptr = nullptr;
    size_t capacity_bytes = 0;
};

struct HostRegistration {
    unsigned long long *ptr = nullptr;
    size_t registered_bytes = 0;
    // Default: off. The zero-copy path (see ntt/src/cuda.rs) hands us a new
    // host buffer on every call, so pinning would re-pin 128 MB per big NTT
    // call (~15 ms wasted). Pageable cudaMemcpyAsync on CUDA 13 staging
    // buffers reaches ~12 GB/s which is within a few ms of pinned for our
    // 128 MB transfers. Opt-in via PROVEKIT_CUDA_NTT_PIN_HOST_BUFFER=1 for
    // workloads that reuse the same host buffer many times.
    bool registration_enabled = false;
};

struct CudaNttContext {
    int device = -1;
    bool initialized = false;
    cudaStream_t stream = nullptr;
    unsigned threads_per_block = THREADS_PER_BLOCK;
    size_t fused_tail_max_elements = FUSED_TAIL_MAX_ELEMENTS;
    DeviceBuffer values;
    DeviceBuffer modulus;
    HostRegistration host_values;
    std::unordered_map<size_t, DeviceBuffer> roots;
};

CudaNttContext &cuda_ntt_context() {
    static CudaNttContext context;
    return context;
}

std::mutex &cuda_ntt_context_mutex() {
    static std::mutex mutex;
    return mutex;
}

void free_device_buffer(DeviceBuffer &buffer) {
    if (buffer.ptr != nullptr) {
        cudaFree(buffer.ptr);
        buffer.ptr = nullptr;
    }
    buffer.capacity_bytes = 0;
}

void unregister_host_buffer(HostRegistration &registration) {
    if (registration.ptr != nullptr) {
        cudaHostUnregister(registration.ptr);
        registration.ptr = nullptr;
    }
    registration.registered_bytes = 0;
}

void destroy_context(CudaNttContext &context) {
    if (context.device >= 0) {
        cudaSetDevice(context.device);
    }

    for (auto &entry : context.roots) {
        free_device_buffer(entry.second);
    }
    context.roots.clear();

    unregister_host_buffer(context.host_values);
    free_device_buffer(context.modulus);
    free_device_buffer(context.values);

    if (context.stream != nullptr) {
        cudaStreamDestroy(context.stream);
        context.stream = nullptr;
    }

    context.device = -1;
    context.initialized = false;
}

void write_error(char *buffer, size_t buffer_len, const char *message) {
    if (buffer == nullptr || buffer_len == 0) {
        return;
    }
    std::snprintf(buffer, buffer_len, "%s", message);
}

void write_cuda_error(char *buffer, size_t buffer_len, const char *prefix, cudaError_t error) {
    if (buffer == nullptr || buffer_len == 0) {
        return;
    }
    std::snprintf(buffer, buffer_len, "%s: %s", prefix, cudaGetErrorString(error));
}

bool parse_env_size(const char *name, size_t *out_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }

    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || (end != nullptr && *end != '\0')) {
        *out_value = static_cast<size_t>(-1);
        return true;
    }

    *out_value = static_cast<size_t>(parsed);
    return true;
}

size_t round_down_power_of_two(size_t value) {
    if (value <= 1) {
        return value;
    }

    size_t power = 1;
    while (power <= value / 2) {
        power <<= 1;
    }
    return power;
}

unsigned default_threads_per_block(const cudaDeviceProp &properties) {
    if (properties.maxThreadsPerBlock <= 0) {
        return THREADS_PER_BLOCK;
    }

    size_t threads = static_cast<size_t>(properties.maxThreadsPerBlock);
    threads = threads >= THREADS_PER_BLOCK ? THREADS_PER_BLOCK : threads;
    if (threads >= 32) {
        threads = (threads / 32) * 32;
    }
    if (threads == 0) {
        threads = static_cast<size_t>(properties.maxThreadsPerBlock);
    }
    return static_cast<unsigned>(threads);
}

unsigned clamp_threads_per_block(size_t requested, const cudaDeviceProp &properties) {
    const unsigned fallback = default_threads_per_block(properties);
    if (requested == 0) {
        return fallback;
    }

    size_t threads = requested;
    const size_t max_threads = properties.maxThreadsPerBlock > 0
        ? static_cast<size_t>(properties.maxThreadsPerBlock)
        : static_cast<size_t>(fallback);
    if (threads > max_threads) {
        threads = max_threads;
    }
    if (threads >= 32) {
        threads = (threads / 32) * 32;
    }
    if (threads == 0) {
        threads = max_threads >= 32 ? 32 : max_threads;
    }
    return static_cast<unsigned>(threads);
}

size_t default_fused_tail_max_elements(const cudaDeviceProp &properties) {
    const size_t shared_capacity = properties.sharedMemPerBlock > 0
        ? static_cast<size_t>(properties.sharedMemPerBlock) / sizeof(Fr256)
        : 0;
    const size_t rounded_capacity = round_down_power_of_two(shared_capacity);
    if (rounded_capacity < 2) {
        return 0;
    }

    return rounded_capacity > 2048 ? 2048 : rounded_capacity;
}

size_t clamp_fused_tail_max_elements(size_t requested, const cudaDeviceProp &properties) {
    if (requested == 0) {
        return 0;
    }

    const size_t device_limit = default_fused_tail_max_elements(properties);
    if (device_limit == 0) {
        return 0;
    }

    size_t elements = round_down_power_of_two(requested);
    if (elements < 2) {
        elements = 2;
    }
    if (elements > device_limit) {
        elements = device_limit;
    }
    return elements;
}

int configure_context_tuning(
    CudaNttContext &context,
    int device,
    char *error_buffer,
    size_t error_buffer_len
) {
    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) {
        write_cuda_error(error_buffer, error_buffer_len, "cudaGetDeviceProperties failed", error);
        return static_cast<int>(error);
    }

    context.threads_per_block = default_threads_per_block(properties);
    context.fused_tail_max_elements = default_fused_tail_max_elements(properties);

    // Opt-in host buffer pinning. See HostRegistration.registration_enabled
    // for the default-off rationale.
    const char *pin_raw = std::getenv("PROVEKIT_CUDA_NTT_PIN_HOST_BUFFER");
    if (pin_raw != nullptr && pin_raw[0] != '\0' &&
        (pin_raw[0] == '1' || pin_raw[0] == 't' || pin_raw[0] == 'T' ||
         pin_raw[0] == 'y' || pin_raw[0] == 'Y')) {
        context.host_values.registration_enabled = true;
    }

    size_t env_threads = 0;
    if (parse_env_size("PROVEKIT_CUDA_NTT_THREADS_PER_BLOCK", &env_threads)) {
        if (env_threads == static_cast<size_t>(-1)) {
            write_error(
                error_buffer,
                error_buffer_len,
                "PROVEKIT_CUDA_NTT_THREADS_PER_BLOCK must be a non-negative integer"
            );
            return -1;
        }
        context.threads_per_block = clamp_threads_per_block(env_threads, properties);
    }

    size_t env_tail = 0;
    if (parse_env_size("PROVEKIT_CUDA_NTT_FUSED_TAIL_MAX_ELEMENTS", &env_tail)) {
        if (env_tail == static_cast<size_t>(-1)) {
            write_error(
                error_buffer,
                error_buffer_len,
                "PROVEKIT_CUDA_NTT_FUSED_TAIL_MAX_ELEMENTS must be a non-negative integer"
            );
            return -1;
        }
        context.fused_tail_max_elements = clamp_fused_tail_max_elements(env_tail, properties);
    }

    return 0;
}

__device__ __forceinline__ Fr256 load_fr(const unsigned long long *src, size_t index) {
    Fr256 out;
    const size_t base = index * LIMBS;
    #pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) {
        out.limbs[limb] = src[base + limb];
    }
    return out;
}

__device__ __forceinline__ void store_fr(unsigned long long *dst, size_t index, const Fr256 &value) {
    const size_t base = index * LIMBS;
    #pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) {
        dst[base + limb] = value.limbs[limb];
    }
}

__device__ __forceinline__ bool geq_modulus(const Fr256 &value, const unsigned long long *modulus) {
    for (int limb = LIMBS - 1; limb >= 0; --limb) {
        if (value.limbs[limb] > modulus[limb]) {
            return true;
        }
        if (value.limbs[limb] < modulus[limb]) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ Fr256 sub_modulus(const Fr256 &value, const unsigned long long *modulus) {
    Fr256 out{};
    unsigned long long borrow = 0;

    #pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) {
        const unsigned __int128 minuend = static_cast<unsigned __int128>(value.limbs[limb]);
        const unsigned __int128 subtrahend =
            static_cast<unsigned __int128>(modulus[limb]) + borrow;
        out.limbs[limb] = static_cast<unsigned long long>(minuend - subtrahend);
        borrow = minuend < subtrahend;
    }

    return out;
}

__device__ __forceinline__ Fr256 add_mod(const Fr256 &lhs, const Fr256 &rhs, const unsigned long long *modulus) {
    Fr256 out{};
    unsigned long long carry = 0;

    #pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) {
        const unsigned __int128 sum =
            static_cast<unsigned __int128>(lhs.limbs[limb]) +
            rhs.limbs[limb] +
            carry;
        out.limbs[limb] = static_cast<unsigned long long>(sum);
        carry = static_cast<unsigned long long>(sum >> 64);
    }

    if (carry != 0 || geq_modulus(out, modulus)) {
        return sub_modulus(out, modulus);
    }

    return out;
}

__device__ __forceinline__ Fr256 sub_mod(const Fr256 &lhs, const Fr256 &rhs, const unsigned long long *modulus) {
    Fr256 out{};
    unsigned long long borrow = 0;

    #pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) {
        const unsigned __int128 minuend = static_cast<unsigned __int128>(lhs.limbs[limb]);
        const unsigned __int128 subtrahend =
            static_cast<unsigned __int128>(rhs.limbs[limb]) + borrow;
        out.limbs[limb] = static_cast<unsigned long long>(minuend - subtrahend);
        borrow = minuend < subtrahend;
    }

    if (borrow == 0) {
        return out;
    }

    unsigned long long carry = 0;
    #pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) {
        const unsigned __int128 sum =
            static_cast<unsigned __int128>(out.limbs[limb]) +
            modulus[limb] +
            carry;
        out.limbs[limb] = static_cast<unsigned long long>(sum);
        carry = static_cast<unsigned long long>(sum >> 64);
    }

    return out;
}

__device__ __forceinline__ Fr256 mont_mul(
    const Fr256 &lhs,
    const Fr256 &rhs,
    const unsigned long long *modulus,
    unsigned long long montgomery_inv
) {
    unsigned long long t[2 * LIMBS + 1] = {};

    #pragma unroll
    for (int i = 0; i < LIMBS; ++i) {
        unsigned __int128 carry = 0;

        #pragma unroll
        for (int j = 0; j < LIMBS; ++j) {
            const unsigned __int128 acc =
                static_cast<unsigned __int128>(lhs.limbs[i]) * rhs.limbs[j] +
                t[i + j] +
                carry;
            t[i + j] = static_cast<unsigned long long>(acc);
            carry = acc >> 64;
        }

        int index = i + LIMBS;
        while (carry != 0) {
            const unsigned __int128 acc = static_cast<unsigned __int128>(t[index]) + carry;
            t[index] = static_cast<unsigned long long>(acc);
            carry = acc >> 64;
            ++index;
        }
    }

    #pragma unroll
    for (int i = 0; i < LIMBS; ++i) {
        const unsigned long long m = t[i] * montgomery_inv;
        unsigned __int128 carry = 0;

        #pragma unroll
        for (int j = 0; j < LIMBS; ++j) {
            const unsigned __int128 acc =
                static_cast<unsigned __int128>(m) * modulus[j] +
                t[i + j] +
                carry;
            t[i + j] = static_cast<unsigned long long>(acc);
            carry = acc >> 64;
        }

        int index = i + LIMBS;
        while (carry != 0) {
            const unsigned __int128 acc = static_cast<unsigned __int128>(t[index]) + carry;
            t[index] = static_cast<unsigned long long>(acc);
            carry = acc >> 64;
            ++index;
        }
    }

    Fr256 out{};
    #pragma unroll
    for (int limb = 0; limb < LIMBS; ++limb) {
        out.limbs[limb] = t[limb + LIMBS];
    }

    if (geq_modulus(out, modulus)) {
        return sub_modulus(out, modulus);
    }

    return out;
}

__global__ void interleaved_ntt_stage(
    unsigned long long *values,
    const unsigned long long *roots,
    size_t elements_in_group,
    size_t num_groups,
    const unsigned long long *modulus,
    unsigned long long montgomery_inv
) {
    const size_t butterflies_per_group = elements_in_group >> 1;
    const size_t total_butterflies = butterflies_per_group * num_groups;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_butterflies) {
        return;
    }

    const size_t group = idx / butterflies_per_group;
    const size_t offset = idx % butterflies_per_group;
    const size_t even_index = group * elements_in_group + offset;
    const size_t odd_index = even_index + butterflies_per_group;

    const Fr256 even = load_fr(values, even_index);
    const Fr256 odd = load_fr(values, odd_index);
    const Fr256 omega = load_fr(roots, group);
    const Fr256 twiddled = mont_mul(omega, odd, modulus, montgomery_inv);

    store_fr(values, even_index, add_mod(even, twiddled, modulus));
    store_fr(values, odd_index, sub_mod(even, twiddled, modulus));
}

__global__ void replicate_first_segment(
    unsigned long long *values,
    size_t segment_len,
    size_t segment_count
) {
    const size_t replicated_values = segment_len * (segment_count - 1);
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= replicated_values) {
        return;
    }

    const size_t segment_offset = idx % segment_len;
    const size_t target_segment = idx / segment_len + 1;
    const Fr256 value = load_fr(values, segment_offset);
    store_fr(values, target_segment * segment_len + segment_offset, value);
}

__global__ void interleaved_ntt_tail(
    unsigned long long *values,
    const unsigned long long *roots,
    size_t segment_len,
    size_t tail_ntt_size,
    const unsigned long long *modulus,
    unsigned long long montgomery_inv
) {
    extern __shared__ Fr256 segment[];

    const size_t segment_index = blockIdx.x;
    const size_t segment_base = segment_index * segment_len;

    for (size_t idx = threadIdx.x; idx < segment_len; idx += blockDim.x) {
        segment[idx] = load_fr(values, segment_base + idx);
    }
    __syncthreads();

    size_t elements_in_group = segment_len;
    size_t num_groups = 1;
    while (num_groups < tail_ntt_size) {
        const size_t butterflies_per_group = elements_in_group >> 1;
        const size_t total_butterflies = segment_len >> 1;

        for (size_t idx = threadIdx.x; idx < total_butterflies; idx += blockDim.x) {
            const size_t group = idx / butterflies_per_group;
            const size_t offset = idx % butterflies_per_group;
            const size_t even_index = group * elements_in_group + offset;
            const size_t odd_index = even_index + butterflies_per_group;
            const Fr256 omega = load_fr(roots, segment_index * num_groups + group);
            const Fr256 even = segment[even_index];
            const Fr256 odd = segment[odd_index];
            const Fr256 twiddled = mont_mul(omega, odd, modulus, montgomery_inv);

            segment[even_index] = add_mod(even, twiddled, modulus);
            segment[odd_index] = sub_mod(even, twiddled, modulus);
        }

        __syncthreads();
        elements_in_group >>= 1;
        num_groups <<= 1;
    }

    for (size_t idx = threadIdx.x; idx < segment_len; idx += blockDim.x) {
        store_fr(values, segment_base + idx, segment[idx]);
    }
}

int ensure_context_initialized(
    CudaNttContext &context,
    int device,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (context.initialized && context.device == device && context.stream != nullptr) {
        return 0;
    }

    if (context.initialized) {
        destroy_context(context);
    }

    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) {
        write_cuda_error(error_buffer, error_buffer_len, "cudaGetDeviceCount failed", error);
        destroy_context(context);
        return static_cast<int>(error);
    }

    if (device < 0 || device >= count) {
        write_error(error_buffer, error_buffer_len, "requested CUDA device index is unavailable");
        destroy_context(context);
        return -1;
    }

    error = cudaSetDevice(device);
    if (error != cudaSuccess) {
        write_cuda_error(error_buffer, error_buffer_len, "cudaSetDevice failed", error);
        destroy_context(context);
        return static_cast<int>(error);
    }

    error = cudaFree(nullptr);
    if (error != cudaSuccess) {
        write_cuda_error(error_buffer, error_buffer_len, "CUDA context initialization failed", error);
        destroy_context(context);
        return static_cast<int>(error);
    }

    error = cudaStreamCreate(&context.stream);
    if (error != cudaSuccess) {
        write_cuda_error(error_buffer, error_buffer_len, "cudaStreamCreate failed", error);
        destroy_context(context);
        return static_cast<int>(error);
    }

    const int tuning = configure_context_tuning(context, device, error_buffer, error_buffer_len);
    if (tuning != 0) {
        destroy_context(context);
        return tuning;
    }

    context.device = device;
    context.initialized = true;
    return 0;
}

bool ensure_active_device(
    const CudaNttContext &context,
    char *error_buffer,
    size_t error_buffer_len
) {
    cudaError_t error = cudaSetDevice(context.device);
    if (error != cudaSuccess) {
        write_cuda_error(error_buffer, error_buffer_len, "cudaSetDevice failed", error);
        return false;
    }

    return true;
}

bool ensure_buffer_capacity(
    DeviceBuffer &buffer,
    size_t required_bytes,
    const char *label,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (buffer.capacity_bytes >= required_bytes) {
        return true;
    }

    free_device_buffer(buffer);

    const size_t allocation_bytes = required_bytes == 0 ? sizeof(unsigned long long) : required_bytes;
    cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&buffer.ptr), allocation_bytes);
    if (error != cudaSuccess) {
        write_cuda_error(error_buffer, error_buffer_len, label, error);
        free_device_buffer(buffer);
        return false;
    }

    buffer.capacity_bytes = allocation_bytes;
    return true;
}

bool ensure_host_values_registration(
    CudaNttContext &context,
    unsigned long long *host_values,
    size_t values_bytes,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (host_values == nullptr || values_bytes == 0) {
        return true;
    }

    HostRegistration &registration = context.host_values;
    if (!registration.registration_enabled) {
        return true;
    }

    if (registration.ptr == host_values && registration.registered_bytes >= values_bytes) {
        return true;
    }

    unregister_host_buffer(registration);

    cudaError_t error = cudaHostRegister(host_values, values_bytes, cudaHostRegisterDefault);
    if (error != cudaSuccess) {
        registration.registration_enabled = false;
        write_cuda_error(
            error_buffer,
            error_buffer_len,
            "registering reusable CUDA host buffer failed; falling back to pageable transfers",
            error
        );
        if (error_buffer != nullptr && error_buffer_len > 0) {
            error_buffer[0] = '\0';
        }
        return true;
    }

    registration.ptr = host_values;
    registration.registered_bytes = values_bytes;
    return true;
}

bool ensure_constants(
    CudaNttContext &context,
    const unsigned long long *modulus,
    char *error_buffer,
    size_t error_buffer_len
) {
    constexpr size_t constant_bytes = LIMBS * sizeof(unsigned long long);
    const bool modulus_missing = context.modulus.ptr == nullptr;

    if (!ensure_buffer_capacity(
            context.modulus,
            constant_bytes,
            "cudaMalloc for modulus failed",
            error_buffer,
            error_buffer_len
        )) {
        return false;
    }

    cudaError_t error = cudaSuccess;
    if (modulus_missing) {
        error = cudaMemcpyAsync(
            context.modulus.ptr,
            modulus,
            constant_bytes,
            cudaMemcpyHostToDevice,
            context.stream
        );
        if (error != cudaSuccess) {
            write_cuda_error(
                error_buffer,
                error_buffer_len,
                "copying the BN254 modulus to CUDA memory failed",
                error
            );
            return false;
        }
    }

    return true;
}

bool ensure_roots(
    CudaNttContext &context,
    size_t num_roots,
    const unsigned long long *host_roots,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (context.roots.find(num_roots) != context.roots.end()) {
        return true;
    }

    DeviceBuffer roots;
    const size_t roots_bytes = num_roots * LIMBS * sizeof(unsigned long long);
    if (!ensure_buffer_capacity(
            roots,
            roots_bytes,
            "cudaMalloc for roots failed",
            error_buffer,
            error_buffer_len
        )) {
        return false;
    }

    cudaError_t error = cudaMemcpyAsync(
        roots.ptr,
        host_roots,
        roots_bytes,
        cudaMemcpyHostToDevice,
        context.stream
    );
    if (error != cudaSuccess) {
        write_cuda_error(
            error_buffer,
            error_buffer_len,
            "copying roots to CUDA memory failed",
            error
        );
        free_device_buffer(roots);
        return false;
    }

    context.roots.emplace(num_roots, roots);
    return true;
}

}  // namespace

extern "C" int provekit_cuda_ntt_preflight(
    int device,
    char *error_buffer,
    size_t error_buffer_len
) {
    std::lock_guard<std::mutex> guard(cuda_ntt_context_mutex());

    if (error_buffer != nullptr && error_buffer_len > 0) {
        error_buffer[0] = '\0';
    }

    return ensure_context_initialized(cuda_ntt_context(), device, error_buffer, error_buffer_len);
}

extern "C" int provekit_cuda_interleaved_ntt(
    int device,
    const unsigned long long *host_values,
    size_t input_elements,
    size_t num_values,
    const unsigned long long *host_roots,
    size_t num_roots,
    size_t elements_in_group_start,
    size_t num_groups_start,
    const unsigned long long *modulus,
    unsigned long long montgomery_inv,
    unsigned long long *host_output,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (error_buffer != nullptr && error_buffer_len > 0) {
        error_buffer[0] = '\0';
    }

    if (input_elements == 0 || input_elements > num_values) {
        write_error(error_buffer, error_buffer_len, "input_elements must be in the range 1..=num_values");
        return -1;
    }

    if (num_groups_start == 0 || elements_in_group_start == 0) {
        write_error(error_buffer, error_buffer_len, "elements_in_group_start and num_groups_start must be non-zero");
        return -1;
    }

    std::lock_guard<std::mutex> guard(cuda_ntt_context_mutex());
    CudaNttContext &context = cuda_ntt_context();

    const int init = ensure_context_initialized(context, device, error_buffer, error_buffer_len);
    if (init != 0) {
        return init;
    }

    if (!ensure_active_device(context, error_buffer, error_buffer_len)) {
        destroy_context(context);
        return -1;
    }

    const size_t values_bytes = num_values * LIMBS * sizeof(unsigned long long);
    const size_t input_bytes = input_elements * LIMBS * sizeof(unsigned long long);
    if (!ensure_constants(context, modulus, error_buffer, error_buffer_len) ||
        !ensure_roots(context, num_roots, host_roots, error_buffer, error_buffer_len) ||
        !ensure_host_values_registration(
            context,
            const_cast<unsigned long long *>(host_values),
            values_bytes,
            error_buffer,
            error_buffer_len
        ) ||
        !ensure_buffer_capacity(
            context.values,
            values_bytes,
            "cudaMalloc for values failed",
            error_buffer,
            error_buffer_len
        )) {
        destroy_context(context);
        return -1;
    }

    cudaError_t error = cudaMemcpyAsync(
        context.values.ptr,
        host_values,
        input_bytes,
        cudaMemcpyHostToDevice,
        context.stream
    );
    if (error != cudaSuccess) {
        write_cuda_error(
            error_buffer,
            error_buffer_len,
            "copying values to CUDA memory failed",
            error
        );
        destroy_context(context);
        return static_cast<int>(error);
    }

    if (input_elements < num_values) {
        if (input_elements != elements_in_group_start) {
            write_error(
                error_buffer,
                error_buffer_len,
                "input_elements must match the first-stage segment length when using on-device replication"
            );
            destroy_context(context);
            return -1;
        }

        if (num_values != input_elements * num_groups_start) {
            write_error(
                error_buffer,
                error_buffer_len,
                "num_values must equal input_elements * num_groups_start for on-device replication"
            );
            destroy_context(context);
            return -1;
        }

        const size_t replicated_values = num_values - input_elements;
        const unsigned blocks = static_cast<unsigned>(
            (replicated_values + context.threads_per_block - 1) / context.threads_per_block
        );
        replicate_first_segment<<<blocks == 0 ? 1 : blocks,
                                  context.threads_per_block,
                                  0,
                                  context.stream>>>(
            context.values.ptr,
            input_elements,
            num_groups_start
        );
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            write_cuda_error(
                error_buffer,
                error_buffer_len,
                "launching the CUDA first-segment replication kernel failed",
                error
            );
            destroy_context(context);
            return static_cast<int>(error);
        }
    }

    size_t elements_in_group = elements_in_group_start;
    size_t num_groups = num_groups_start;
    const size_t codeword_size = num_roots * 2;
    const unsigned long long *d_roots = context.roots.at(num_roots).ptr;
    while (num_groups < codeword_size) {
        const size_t tail_ntt_size = codeword_size / num_groups;
        if (context.fused_tail_max_elements >= 2 &&
            elements_in_group <= context.fused_tail_max_elements &&
            tail_ntt_size > 1) {
            const size_t shared_bytes = elements_in_group * sizeof(Fr256);
            interleaved_ntt_tail<<<num_groups == 0 ? 1 : static_cast<unsigned>(num_groups),
                                   context.threads_per_block,
                                   shared_bytes,
                                   context.stream>>>(
                context.values.ptr,
                d_roots,
                elements_in_group,
                tail_ntt_size,
                context.modulus.ptr,
                montgomery_inv
            );
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                write_cuda_error(
                    error_buffer,
                    error_buffer_len,
                    "launching the fused CUDA interleaved NTT tail kernel failed",
                    error
                );
                destroy_context(context);
                return static_cast<int>(error);
            }
            break;
        }

        const size_t butterflies = num_groups * (elements_in_group / 2);
        const unsigned blocks = static_cast<unsigned>(
            (butterflies + context.threads_per_block - 1) / context.threads_per_block
        );
        interleaved_ntt_stage<<<blocks == 0 ? 1 : blocks,
                                context.threads_per_block,
                                0,
                                context.stream>>>(
            context.values.ptr,
            d_roots,
            elements_in_group,
            num_groups,
            context.modulus.ptr,
            montgomery_inv
        );
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            write_cuda_error(
                error_buffer,
                error_buffer_len,
                "launching the CUDA interleaved NTT stage kernel failed",
                error
            );
            destroy_context(context);
            return static_cast<int>(error);
        }

        elements_in_group /= 2;
        num_groups *= 2;
    }

    error = cudaMemcpyAsync(
        host_output,
        context.values.ptr,
        values_bytes,
        cudaMemcpyDeviceToHost,
        context.stream
    );
    if (error != cudaSuccess) {
        write_cuda_error(
            error_buffer,
            error_buffer_len,
            "copying CUDA NTT output back to host memory failed",
            error
        );
        destroy_context(context);
        return static_cast<int>(error);
    }

    error = cudaStreamSynchronize(context.stream);
    if (error != cudaSuccess) {
        write_cuda_error(
            error_buffer,
            error_buffer_len,
            "waiting for the CUDA NTT stream failed",
            error
        );
        destroy_context(context);
        return static_cast<int>(error);
    }

    return 0;
}

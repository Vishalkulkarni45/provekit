// SPDX-License-Identifier: MIT / Apache-2.0
//
// Thin FFI shim around sppark's NTT for BN254, callable from Rust.
//
// The key observation is that sppark's `alt_bn128::fr_t` is a 4 × u64
// Montgomery representation with the same modulus constants and the same
// Montgomery R as arkworks `ark_bn254::Fr` (which is `#[repr(transparent)]`
// over `BigInt<4> = [u64; 4]`). So the Rust wrapper in `ntt/src/sppark_ntt.rs`
// passes `values.as_mut_ptr() as *mut fr_t` and the kernel reads arkworks
// buffers directly — zero host-side conversion.

#define FEATURE_BN254
#include <ff/alt_bn128.hpp>
#include <ntt/ntt.cuh>

#include <cstdint>

// Expose a very thin C entrypoint. We do not want to pull in all of sppark's
// RustError/Slice types into the provekit crate, so we only propagate success
// (0) / cudaError code. The error message ends up inside a fixed-size buffer
// the caller provides.
extern "C" int provekit_sppark_bn254_ntt(
    size_t device_id,
    void*  inout,
    uint32_t lg_domain_size,
    unsigned ntt_order,      // 0=NN, 1=NR, 2=RN, 3=RR
    unsigned ntt_direction,  // 0=Forward, 1=Inverse
    unsigned ntt_type,       // 0=Standard, 1=Coset
    char*  error_buffer,
    size_t error_buffer_len
)
{
    if (error_buffer != nullptr && error_buffer_len > 0) {
        error_buffer[0] = '\0';
    }

    auto order = static_cast<NTT::InputOutputOrder>(ntt_order);
    auto dir   = static_cast<NTT::Direction>(ntt_direction);
    auto typ   = static_cast<NTT::Type>(ntt_type);

    try {
        auto& gpu = select_gpu(device_id);
        RustError err = NTT::Base(
            gpu,
            static_cast<fr_t*>(inout),
            lg_domain_size,
            order,
            dir,
            typ
        );
        if (err.code != 0) {
            if (error_buffer != nullptr && error_buffer_len > 0 && err.message != nullptr) {
                std::snprintf(error_buffer, error_buffer_len, "%s", err.message);
                free(err.message);
            }
            return err.code;
        }
        return 0;
    } catch (const cuda_error& e) {
        if (error_buffer != nullptr && error_buffer_len > 0) {
            std::snprintf(error_buffer, error_buffer_len,
                          "sppark NTT cuda_error: %s", e.what());
        }
        return -1;
    } catch (const std::exception& e) {
        if (error_buffer != nullptr && error_buffer_len > 0) {
            std::snprintf(error_buffer, error_buffer_len,
                          "sppark NTT exception: %s", e.what());
        }
        return -1;
    }
}

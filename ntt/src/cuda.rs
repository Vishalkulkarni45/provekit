use {
    anyhow::{anyhow, ensure, Context, Result},
    ark_bn254::Fr,
    ark_ff::{AdditiveGroup, PrimeField},
    rayon::prelude::{
        IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
    },
    std::{
        collections::HashMap,
        env,
        ffi::{c_char, c_int, CStr},
        sync::{Arc, Mutex, OnceLock},
    },
};

const LIMBS_PER_FR: usize = 4;
const CUDA_SUCCESS: c_int = 0;
const ERROR_BUFFER_LEN: usize = 1024;
const PARALLEL_MARSHAL_THRESHOLD: usize = 1 << 14;
const MAX_CODEWORD_ORDER: usize = 1 << 28;

// Static assertions: `Fr` must be layout-compatible with `[u64; LIMBS_PER_FR]`
// so we can reinterpret `&[Fr]` as `&[u64]` without copying.
// `ark-ff`'s `Fp<P, N>` wraps a single `BigInt<N>` (plus a zero-sized PhantomData);
// `BigInt<N>` wraps `[u64; N]`. The compiler is free to lay this out, but in
// every version of ark-ff through 0.5 the non-ZST field is the only real
// storage, so size and alignment must match `[u64; 4]`. We enforce that at
// compile time below; if a future ark-ff release changes the layout this
// will refuse to build, forcing a revisit of the zero-copy path.
const _: () = {
    assert!(std::mem::size_of::<Fr>() == LIMBS_PER_FR * std::mem::size_of::<u64>());
    assert!(std::mem::align_of::<Fr>() >= std::mem::align_of::<u64>());
};

/// Reinterpret `&[Fr]` as `&[u64]` without copying. Safe given the layout
/// assertions above: each `Fr` is exactly `LIMBS_PER_FR` u64 limbs in
/// Montgomery form, which is the representation the CUDA kernel expects.
#[inline]
fn fr_slice_as_u64(values: &[Fr]) -> &[u64] {
    // SAFETY: const-asserted size/alignment match. `Fr` is `Copy`, plain old
    // data (no niches), and aligned to at least 8 bytes.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u64, values.len() * LIMBS_PER_FR)
    }
}

#[inline]
fn fr_slice_as_u64_mut(values: &mut [Fr]) -> &mut [u64] {
    // SAFETY: same as `fr_slice_as_u64`.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr() as *mut u64,
            values.len() * LIMBS_PER_FR,
        )
    }
}

unsafe extern "C" {
    fn provekit_cuda_ntt_preflight(
        device: c_int,
        error_buffer: *mut c_char,
        error_buffer_len: usize,
    ) -> c_int;

    fn provekit_cuda_interleaved_ntt(
        device: c_int,
        host_values: *const u64,
        input_elements: usize,
        num_values: usize,
        host_roots: *const u64,
        num_roots: usize,
        elements_in_group_start: usize,
        num_groups_start: usize,
        modulus: *const u64,
        montgomery_inv: u64,
        host_output: *mut u64,
        error_buffer: *mut c_char,
        error_buffer_len: usize,
    ) -> c_int;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaNttPolicy {
    Auto,
    SizeOnly,
    Always,
    Never,
}

#[derive(Debug)]
struct CudaRuntimeState {
    device:          usize,
    modulus:         [u64; LIMBS_PER_FR],
    montgomery_inv:  u64,
    flattened_roots: Mutex<HashMap<usize, Arc<[u64]>>>,
}

fn parse_cuda_device_index() -> Result<usize> {
    match env::var("PROVEKIT_CUDA_DEVICE") {
        Ok(value) => value.parse::<usize>().with_context(|| {
            format!("PROVEKIT_CUDA_DEVICE must be a non-negative integer, got `{value}` instead")
        }),
        Err(env::VarError::NotPresent) => Ok(0),
        Err(err) => Err(anyhow!(
            "failed to read PROVEKIT_CUDA_DEVICE for CUDA NTT selection: {err}"
        )),
    }
}

fn cuda_device_index() -> Result<usize> {
    static CUDA_DEVICE_INDEX: OnceLock<Result<usize, String>> = OnceLock::new();

    CUDA_DEVICE_INDEX
        .get_or_init(|| parse_cuda_device_index().map_err(|err| format!("{err:#}")))
        .clone()
        .map_err(anyhow::Error::msg)
}

fn cuda_min_codeword_size() -> Result<usize> {
    static CUDA_MIN_CODEWORD_SIZE: OnceLock<Result<usize, String>> = OnceLock::new();

    CUDA_MIN_CODEWORD_SIZE
        .get_or_init(|| match env::var("PROVEKIT_CUDA_NTT_MIN_CODEWORD_SIZE") {
            Ok(value) => value.parse::<usize>().map_err(|err| {
                format!(
                    "PROVEKIT_CUDA_NTT_MIN_CODEWORD_SIZE must be a non-negative integer, got \
                     `{value}` instead: {err}"
                )
            }),
            Err(env::VarError::NotPresent) => Ok(0),
            Err(err) => Err(format!(
                "failed to read PROVEKIT_CUDA_NTT_MIN_CODEWORD_SIZE for CUDA NTT tuning: {err}"
            )),
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

pub fn cuda_should_offload_codeword_size(codeword_size: usize) -> Result<bool> {
    Ok(codeword_size >= cuda_min_codeword_size()?)
}

pub fn cuda_ntt_policy() -> Result<CudaNttPolicy> {
    static CUDA_NTT_POLICY: OnceLock<Result<CudaNttPolicy, String>> = OnceLock::new();

    CUDA_NTT_POLICY
        .get_or_init(|| match env::var("PROVEKIT_CUDA_NTT_POLICY") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "auto" => Ok(CudaNttPolicy::Auto),
                "size-only" | "size_only" | "size" => Ok(CudaNttPolicy::SizeOnly),
                "always" => Ok(CudaNttPolicy::Always),
                "never" => Ok(CudaNttPolicy::Never),
                _ => Err(format!(
                    "PROVEKIT_CUDA_NTT_POLICY must be one of `auto`, `size-only`, `always`, or \
                     `never`, got `{value}` instead"
                )),
            },
            Err(env::VarError::NotPresent) => Ok(CudaNttPolicy::Auto),
            Err(err) => Err(format!(
                "failed to read PROVEKIT_CUDA_NTT_POLICY for CUDA NTT tuning: {err}"
            )),
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

fn cuda_release_guard_stride() -> Result<usize> {
    static CUDA_RELEASE_GUARD_STRIDE: OnceLock<Result<usize, String>> = OnceLock::new();

    CUDA_RELEASE_GUARD_STRIDE
        .get_or_init(
            || match env::var("PROVEKIT_CUDA_NTT_RELEASE_GUARD_STRIDE") {
                Ok(value) => value.parse::<usize>().map_err(|err| {
                    format!(
                        "PROVEKIT_CUDA_NTT_RELEASE_GUARD_STRIDE must be a non-negative integer, \
                         got `{value}` instead: {err}"
                    )
                }),
                Err(env::VarError::NotPresent) => Ok(0),
                Err(err) => Err(format!(
                    "failed to read PROVEKIT_CUDA_NTT_RELEASE_GUARD_STRIDE for CUDA NTT safety \
                     tuning: {err}"
                )),
            },
        )
        .clone()
        .map_err(anyhow::Error::msg)
}

fn error_buffer_to_string(buffer: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn call_cuda_preflight(device: usize) -> Result<()> {
    let mut error_buffer = [0_i8; ERROR_BUFFER_LEN];
    let status = unsafe {
        provekit_cuda_ntt_preflight(
            device as c_int,
            error_buffer.as_mut_ptr(),
            error_buffer.len(),
        )
    };

    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        let message = error_buffer_to_string(&error_buffer);
        Err(anyhow!(if message.is_empty() {
            format!("CUDA preflight failed with status code {status}")
        } else {
            message
        }))
    }
}

fn montgomery_inv(modulus_limb_0: u64) -> u64 {
    let mut inv = 1_u64;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2_u64.wrapping_sub(modulus_limb_0.wrapping_mul(inv)));
    }
    inv.wrapping_neg()
}

/// Flatten the precomputed twiddle roots into a u64 limb vector. Only called
/// once per unique codeword size (cached in `flattened_roots`); the main
/// per-NTT flatten/hydrate copies have been removed in favour of zero-copy
/// reinterpretation of `[Fr]` as `[u64]` — see `fr_slice_as_u64_mut`.
fn flatten_roots_into(roots: &[Fr], flat: &mut Vec<u64>) {
    flat.resize(roots.len() * LIMBS_PER_FR, 0);
    if roots.len() >= PARALLEL_MARSHAL_THRESHOLD {
        flat.par_chunks_exact_mut(LIMBS_PER_FR)
            .zip(roots.par_iter())
            .for_each(|(limbs, value)| limbs.copy_from_slice(&value.0 .0));
    } else {
        for (limbs, value) in flat.chunks_exact_mut(LIMBS_PER_FR).zip(roots) {
            limbs.copy_from_slice(&value.0 .0);
        }
    }
}

fn next_order_impl(size: usize) -> Option<usize> {
    let order = size.next_power_of_two();
    (order <= MAX_CODEWORD_ORDER).then_some(order)
}

fn build_interleaved_host_values(
    messages: &[&[Fr]],
    masks: &[Fr],
    message_length: usize,
    total_values: usize,
    chunk_size: usize,
    num_cosets: usize,
) -> Vec<Fr> {
    let num_messages = messages.len();
    let mut result = vec![Fr::ZERO; total_values];

    result[..message_length * num_messages]
        .par_chunks_mut(num_messages)
        .enumerate()
        .for_each(|(column, chunk)| {
            for row in 0..num_messages {
                chunk[row] = messages[row][column];
            }
        });

    result[message_length * num_messages..message_length * num_messages + masks.len()]
        .copy_from_slice(masks);

    if num_cosets > 1 {
        let (source, remaining) = result.split_at_mut(chunk_size);
        remaining
            .par_chunks_mut(chunk_size)
            .take(num_cosets - 1)
            .for_each(|chunk| chunk.copy_from_slice(source));
    }

    result
}

fn flattened_roots(codeword_size: usize) -> Arc<[u64]> {
    let runtime = cuda_runtime_state();
    let mut flattened_roots = runtime.flattened_roots.lock().unwrap();

    if let Some(cached) = flattened_roots.get(&codeword_size) {
        return Arc::clone(cached);
    }

    let mut flat = Vec::new();
    flatten_roots_into(&crate::ntt::reverse_ordered_roots(codeword_size), &mut flat);
    let flattened: Arc<[u64]> = flat.into();
    flattened_roots.insert(codeword_size, Arc::clone(&flattened));
    flattened
}

fn should_check_hydrated_value(index: usize, len: usize, stride: usize) -> bool {
    stride == 1 || (stride > 1 && (index == 0 || index + 1 == len || index.is_multiple_of(stride)))
}

fn cuda_runtime_state() -> &'static CudaRuntimeState {
    static CUDA_RUNTIME_STATE: OnceLock<CudaRuntimeState> = OnceLock::new();

    CUDA_RUNTIME_STATE.get_or_init(|| {
        let modulus = Fr::MODULUS.0;
        CudaRuntimeState {
            device: cuda_device_index().expect("CUDA device selection must succeed"),
            modulus,
            montgomery_inv: montgomery_inv(modulus[0]),
            flattened_roots: Mutex::new(HashMap::new()),
        }
    })
}

pub fn preflight_cuda() -> Result<()> {
    static CUDA_PREFLIGHT: OnceLock<Result<(), String>> = OnceLock::new();

    CUDA_PREFLIGHT
        .get_or_init(|| {
            let device = cuda_device_index().map_err(|err| format!("{err:#}"))?;
            call_cuda_preflight(device).map_err(|err| format!("{err:#}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Zero-copy entry point: caller supplies raw limb pointers. Used by
/// `ntt_nr_cuda` and `interleaved_encode_cuda` after reinterpreting their
/// `Fr` buffers as `u64` limbs.
fn ntt_nr_cuda_raw(
    host_input: *const u64,
    host_output: *mut u64,
    input_elements: usize,
    total_values: usize,
    codeword_size: usize,
    num_groups: usize,
) -> Result<()> {
    ensure!(num_groups > 0, "num_groups must be non-zero");
    ensure!(input_elements > 0, "input_elements must be non-zero");
    ensure!(
        input_elements <= total_values,
        "input_elements must be <= total_values"
    );
    ensure!(
        total_values.is_multiple_of(num_groups),
        "total_values must be divisible by num_groups"
    );
    ensure!(
        codeword_size.is_power_of_two(),
        "codeword_size must be a power of two"
    );

    preflight_cuda()?;

    let runtime = cuda_runtime_state();
    let roots_flat = flattened_roots(codeword_size);
    let mut error_buffer = [0_i8; ERROR_BUFFER_LEN];

    let status = unsafe {
        provekit_cuda_interleaved_ntt(
            runtime.device as c_int,
            host_input,
            input_elements,
            total_values,
            roots_flat.as_ptr(),
            codeword_size / 2,
            total_values / num_groups,
            num_groups,
            runtime.modulus.as_ptr(),
            runtime.montgomery_inv,
            host_output,
            error_buffer.as_mut_ptr(),
            error_buffer.len(),
        )
    };

    if status != CUDA_SUCCESS {
        let message = error_buffer_to_string(&error_buffer);
        return Err(anyhow!(if message.is_empty() {
            format!("CUDA NTT execution failed with status code {status}")
        } else {
            message
        }));
    }

    Ok(())
}

pub fn ntt_nr_cuda(values: &mut [Fr], codeword_size: usize, num_groups: usize) -> Result<()> {
    if codeword_size <= 1 {
        return Ok(());
    }

    ensure!(num_groups > 0, "num_groups must be non-zero");
    ensure!(
        values.len() % num_groups == 0,
        "values.len() must be divisible by num_groups"
    );
    ensure!(
        codeword_size.is_power_of_two(),
        "codeword_size must be a power of two"
    );

    preflight_cuda()?;

    if !cuda_should_offload_codeword_size(codeword_size)? {
        crate::ntt::ntt_nr(values, codeword_size, num_groups);
        return Ok(());
    }

    // Zero-copy: reinterpret the Fr slice as u64 limbs and call CUDA in-place.
    let len = values.len();
    let limbs = fr_slice_as_u64_mut(values);
    ntt_nr_cuda_raw(
        limbs.as_ptr(),
        limbs.as_mut_ptr(),
        len,
        len,
        codeword_size,
        num_groups,
    )?;

    if cfg!(debug_assertions) {
        // Cheap sanity check: re-validate every result limb lies inside the
        // BN254 field. In release this is handled by the
        // PROVEKIT_CUDA_NTT_RELEASE_GUARD_STRIDE heuristic instead.
        for value in values.iter() {
            ensure!(
                value.0 < Fr::MODULUS,
                "CUDA kernel returned a value outside the BN254 field"
            );
        }
    } else {
        release_guard_check(values)?;
    }

    Ok(())
}

/// Run the release-mode striped range check for CUDA NTT outputs without
/// allocating a scratch buffer. Mirrors the semantics of the removed
/// `hydrate_field_elements` guard.
fn release_guard_check(values: &[Fr]) -> Result<()> {
    let stride = cuda_release_guard_stride()?;
    if stride == 0 {
        return Ok(());
    }
    let len = values.len();
    if len == 0 {
        return Ok(());
    }
    for (index, value) in values.iter().enumerate() {
        if should_check_hydrated_value(index, len, stride) {
            ensure!(
                value.0 < Fr::MODULUS,
                "CUDA kernel returned a value outside the BN254 field"
            );
        }
    }
    Ok(())
}

pub fn interleaved_encode_cuda(
    messages: &[&[Fr]],
    masks: &[Fr],
    codeword_length: usize,
    coset_size: usize,
    num_cosets: usize,
) -> Result<Vec<Fr>> {
    if messages.is_empty() {
        return Ok(vec![]);
    }

    let num_messages = messages.len();
    let message_length = messages[0].len();
    let mask_length = masks.len() / num_messages;
    let masked_message_length = message_length + mask_length;
    for message in messages {
        ensure!(
            message.len() == message_length,
            "all messages must have the same length"
        );
    }
    ensure!(
        masks.len().is_multiple_of(num_messages),
        "masks.len() must be divisible by the number of messages"
    );
    ensure!(
        codeword_length.is_power_of_two(),
        "codeword_length must be a power of two"
    );
    ensure!(
        codeword_length.is_multiple_of(coset_size),
        "codeword_length must be divisible by coset_size"
    );
    ensure!(
        num_cosets == codeword_length / coset_size,
        "num_cosets must match codeword_length / coset_size"
    );
    ensure!(
        next_order_impl(masked_message_length).is_some_and(|min_coset| min_coset <= coset_size),
        "coset_size must be large enough for the masked message length"
    );
    let total_values = num_messages * codeword_length;
    if !cuda_should_offload_codeword_size(codeword_length)? {
        let mut result = build_interleaved_host_values(
            messages,
            masks,
            message_length,
            total_values,
            coset_size * num_messages,
            num_cosets,
        );

        crate::ntt::ntt_nr(&mut result, codeword_length, num_cosets);
        return Ok(result);
    }

    // Zero-copy: build the interleaved layout directly in the result buffer,
    // reinterpret its Fr slice as u64 limbs, and hand that pointer to CUDA as
    // both input and output. Removes two host-side 128 MB copies (Fr→u64
    // flatten and u64→Fr hydrate) and one extra Vec<Fr> allocation per call.
    let mut values = build_interleaved_host_values(
        messages,
        masks,
        message_length,
        total_values,
        coset_size * num_messages,
        num_cosets,
    );

    {
        let limbs = fr_slice_as_u64_mut(&mut values);
        ntt_nr_cuda_raw(
            limbs.as_ptr(),
            limbs.as_mut_ptr(),
            total_values,
            total_values,
            codeword_length,
            num_cosets,
        )?;
    }

    release_guard_check(&values)?;
    Ok(values)
}

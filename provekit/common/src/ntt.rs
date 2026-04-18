use {
    ark_bn254::Fr,
    ark_ff::{AdditiveGroup, FftField, Field},
    ntt::ntt_nr,
    rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut},
    tracing::instrument,
    whir::algebra::ntt::ReedSolomon,
};

#[derive(Debug)]
pub struct RSFr;

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct RSFrCuda;

#[derive(Clone, Copy, Debug)]
struct InterleavedLayout {
    coset_size: usize,
    num_cosets: usize,
    total_size: usize,
    chunk_size: usize,
}

fn next_order_impl(size: usize) -> Option<usize> {
    let order = size.next_power_of_two();
    if order <= 1 << 28 {
        Some(order)
    } else {
        None
    }
}

fn generator_impl(codeword_length: usize) -> Fr {
    Fr::get_root_of_unity(codeword_length as u64).unwrap()
}

fn evaluation_points_impl(codeword_length: usize, indices: &[usize]) -> Vec<Fr> {
    indices
        .iter()
        .map(|i| {
            let bits = usize::BITS - (codeword_length - 1).leading_zeros();
            let k = if bits == 0 {
                *i
            } else {
                i.reverse_bits() >> (usize::BITS - bits)
            };

            generator_impl(codeword_length).pow([k as u64])
        })
        .collect()
}

fn interleaved_layout(
    num_messages: usize,
    message_length: usize,
    masks_len: usize,
    codeword_length: usize,
) -> InterleavedLayout {
    let masked_message_length = message_length + masks_len / num_messages;

    let mut coset_size = next_order_impl(masked_message_length).unwrap();
    while !codeword_length.is_multiple_of(coset_size) {
        coset_size = next_order_impl(coset_size + 1).unwrap();
    }
    let num_cosets = codeword_length / coset_size;

    InterleavedLayout {
        coset_size,
        num_cosets,
        total_size: num_messages * codeword_length,
        chunk_size: coset_size * num_messages,
    }
}

fn interleaved_encode_impl<F>(
    messages: &[&[Fr]],
    masks: &[Fr],
    codeword_length: usize,
    mut ntt_impl: F,
) -> Vec<Fr>
where
    F: FnMut(&mut [Fr], usize, usize),
{
    if messages.is_empty() {
        return vec![];
    }

    let num_messages = messages.len();
    let message_length = messages[0].len();
    for message in messages {
        assert_eq!(message_length, message.len());
    }

    let layout = interleaved_layout(num_messages, message_length, masks.len(), codeword_length);
    let mut result = vec![Fr::ZERO; layout.total_size];

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

    if layout.num_cosets > 1 {
        let (source, remaining) = result.split_at_mut(layout.chunk_size);
        remaining
            .par_chunks_mut(layout.chunk_size)
            .take(layout.num_cosets - 1)
            .for_each(|chunk| chunk.copy_from_slice(source));
    }

    ntt_impl(&mut result, codeword_length, layout.num_cosets);
    result
}

#[cfg(feature = "cuda")]
fn should_use_cuda_for_shape(codeword_length: usize, _num_cosets: usize) -> bool {
    let size_allows_cuda =
        ntt::cuda_should_offload_codeword_size(codeword_length).unwrap_or_else(|err| {
            panic!("CUDA NTT size policy parsing failed before backend execution: {err:#}")
        });

    match ntt::cuda_ntt_policy().unwrap_or_else(|err| {
        panic!("CUDA NTT policy parsing failed before backend execution: {err:#}")
    }) {
        ntt::CudaNttPolicy::Always => true,
        ntt::CudaNttPolicy::SizeOnly => size_allows_cuda,
        ntt::CudaNttPolicy::Never => false,
        ntt::CudaNttPolicy::Auto => size_allows_cuda,
    }
}

#[cfg(feature = "cuda")]
fn interleaved_encode_cuda_impl(
    messages: &[&[Fr]],
    masks: &[Fr],
    codeword_length: usize,
) -> Vec<Fr> {
    if messages.is_empty() {
        return vec![];
    }

    let num_messages = messages.len();
    let message_length = messages[0].len();
    let layout = interleaved_layout(num_messages, message_length, masks.len(), codeword_length);

    if !should_use_cuda_for_shape(codeword_length, layout.num_cosets) {
        return interleaved_encode_impl(messages, masks, codeword_length, ntt_nr);
    }

    ntt::interleaved_encode_cuda(
        messages,
        masks,
        codeword_length,
        layout.coset_size,
        layout.num_cosets,
    )
    .unwrap_or_else(|err| {
        panic!("CUDA NTT execution failed after successful backend preflight: {err:#}")
    })
}

impl ReedSolomon<Fr> for RSFr {
    fn next_order(&self, size: usize) -> Option<usize> {
        next_order_impl(size)
    }

    fn evaluation_points(
        &self,
        _masked_message_length: usize,
        codeword_length: usize,
        indices: &[usize],
    ) -> Vec<Fr> {
        evaluation_points_impl(codeword_length, indices)
    }

    #[instrument(skip(self, messages, masks), fields(
        num_messages = messages.len(),
        message_len = messages.first().map(|c| c.len()),
        codeword_length = codeword_length,
        mask_len = masks.len().checked_div(messages.len())

    ))]
    fn interleaved_encode(
        &self,
        messages: &[&[Fr]],
        masks: &[Fr],
        codeword_length: usize,
    ) -> Vec<Fr> {
        interleaved_encode_impl(messages, masks, codeword_length, ntt_nr)
    }

    fn generator(&self, codeword_length: usize) -> Fr {
        generator_impl(codeword_length)
    }
}

#[cfg(feature = "cuda")]
impl ReedSolomon<Fr> for RSFrCuda {
    fn next_order(&self, size: usize) -> Option<usize> {
        next_order_impl(size)
    }

    fn evaluation_points(
        &self,
        _masked_message_length: usize,
        codeword_length: usize,
        indices: &[usize],
    ) -> Vec<Fr> {
        evaluation_points_impl(codeword_length, indices)
    }

    #[instrument(skip(self, messages, masks), fields(
        num_messages = messages.len(),
        message_len = messages.first().map(|c| c.len()),
        codeword_length = codeword_length,
        mask_len = masks.len().checked_div(messages.len())

    ))]
    fn interleaved_encode(
        &self,
        messages: &[&[Fr]],
        masks: &[Fr],
        codeword_length: usize,
    ) -> Vec<Fr> {
        interleaved_encode_cuda_impl(messages, masks, codeword_length)
    }

    fn generator(&self, codeword_length: usize) -> Fr {
        generator_impl(codeword_length)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use std::sync::OnceLock;
    use {
        super::*,
        ark_ff::{BigInt, PrimeField},
        proptest::{collection, prelude::*},
        whir::algebra::ntt::NttEngine,
    };

    fn fr() -> impl Strategy<Value = Fr> + Clone {
        proptest::array::uniform4(0u64..).prop_map(|val| Fr::new(BigInt(val)))
    }

    proptest! {
        #[test]
        fn interleaved_encode_matches_whir_reference(
            log_msg in 0_usize..=4,
            log_extra in 0_usize..=3,
            num_messages in 1_usize..=4,
            log_mask in 0_usize..=3,
            messages_flat in collection::vec(fr(), 0..=64),
            masks_flat in collection::vec(fr(), 0..=64),
        ) {
            let message_length = 1 << log_msg;
            let mask_length: usize = 1 << log_mask;
            let masked_message_length = message_length + mask_length;
            let codeword_length = masked_message_length.next_power_of_two() << log_extra;

            let total = num_messages * message_length;
            let mut data = messages_flat;
            data.resize(total, Fr::ZERO);

            let messages: Vec<&[Fr]> = data.chunks(message_length).collect();

            // Our masks are interleaved: num_messages x mask_length in row-major order
            // i.e. [m0_c0, m1_c0, m0_c1, m1_c1, ...]
            let mask_total = num_messages * mask_length;
            let mut masks = masks_flat;
            masks.resize(mask_total, Fr::ZERO);

            // Whir expects masks per-message (column-major from our perspective):
            // [m0_c0, m0_c1, ..., m1_c0, m1_c1, ...]
            // Transpose the num_messages x mask_length matrix.
            let mut masks_transposed = vec![Fr::ZERO; mask_total];
            for row in 0..num_messages {
                for col in 0..mask_length {
                    masks_transposed[row * mask_length + col] = masks[col * num_messages + row];
                }
            }

            let indices: Vec<usize> = (0..codeword_length).collect();

            let reference = NttEngine::<Fr>::new_from_fftfield();
            let our_codeword = RSFr.interleaved_encode(&messages, &masks, codeword_length);
            let ref_codeword = reference.interleaved_encode(&messages, &masks_transposed, codeword_length);

            let our_points = RSFr.evaluation_points(message_length, codeword_length, &indices);
            let ref_points = reference.evaluation_points(message_length, codeword_length, &indices);

            // Pair each evaluation point with its num_messages-wide slice, then sort
            // by point so that ordering differences between implementations don't matter.
            let mut our_rows: Vec<_> = our_points.iter().enumerate()
                .map(|(i, pt)| (pt.into_bigint(), &our_codeword[i * num_messages..(i + 1) * num_messages]))
                .collect();
            our_rows.sort_by_key(|(k, _)| *k);

            let mut ref_rows: Vec<_> = ref_points.iter().enumerate()
                .map(|(i, pt)| (pt.into_bigint(), &ref_codeword[i * num_messages..(i + 1) * num_messages]))
                .collect();
            ref_rows.sort_by_key(|(k, _)| *k);

            prop_assert_eq!(our_rows, ref_rows);
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_ready() -> bool {
        static CUDA_READY: OnceLock<bool> = OnceLock::new();

        *CUDA_READY.get_or_init(|| match ntt::preflight_cuda() {
            Ok(()) => true,
            Err(err) => {
                eprintln!("skipping CUDA parity tests because CUDA is unavailable: {err:#}");
                false
            }
        })
    }

    #[cfg(feature = "cuda")]
    proptest! {
        #[test]
        fn cuda_matches_cpu_next_order(size in 0_usize..=(1 << 20)) {
            prop_assert_eq!(RSFr.next_order(size), RSFrCuda.next_order(size));
        }

        #[test]
        fn cuda_matches_cpu_evaluation_points(
            log_msg in 0_usize..=4,
            log_extra in 0_usize..=3,
            log_mask in 0_usize..=3,
        ) {
            if !cuda_ready() {
                return Ok(());
            }

            let message_length = 1 << log_msg;
            let mask_length: usize = 1 << log_mask;
            let masked_message_length = message_length + mask_length;
            let codeword_length = masked_message_length.next_power_of_two() << log_extra;
            let indices: Vec<usize> = (0..codeword_length).collect();

            prop_assert_eq!(
                RSFr.evaluation_points(message_length, codeword_length, &indices),
                RSFrCuda.evaluation_points(message_length, codeword_length, &indices),
            );
        }

        #[test]
        fn cuda_interleaved_encode_matches_cpu(
            log_msg in 0_usize..=4,
            log_extra in 0_usize..=3,
            num_messages in 1_usize..=4,
            log_mask in 0_usize..=3,
            messages_flat in collection::vec(fr(), 0..=64),
            masks_flat in collection::vec(fr(), 0..=64),
        ) {
            if !cuda_ready() {
                return Ok(());
            }

            let message_length = 1 << log_msg;
            let mask_length: usize = 1 << log_mask;
            let masked_message_length = message_length + mask_length;
            let codeword_length = masked_message_length.next_power_of_two() << log_extra;

            let total = num_messages * message_length;
            let mut data = messages_flat;
            data.resize(total, Fr::ZERO);
            let messages: Vec<&[Fr]> = data.chunks(message_length).collect();

            let mask_total = num_messages * mask_length;
            let mut masks = masks_flat;
            masks.resize(mask_total, Fr::ZERO);

            prop_assert_eq!(
                RSFr.interleaved_encode(&messages, &masks, codeword_length),
                RSFrCuda.interleaved_encode(&messages, &masks, codeword_length),
            );
        }
    }
}

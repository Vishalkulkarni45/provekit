#[cfg(feature = "cuda")]
use ntt::ntt_nr_cuda;
use {
    ark_bn254::Fr,
    ark_ff::UniformRand,
    ark_std::rand::{rngs::StdRng, SeedableRng},
    divan::{black_box, Bencher},
    ntt::ntt_nr,
};

#[derive(Clone, Copy, Debug)]
struct Case {
    num_messages:    usize,
    message_length:  usize,
    codeword_length: usize,
}

const TEST_CASES: &[Case] = &[
    Case {
        num_messages:    8,
        message_length:  1,
        codeword_length: 256,
    },
    Case {
        num_messages:    8,
        message_length:  8,
        codeword_length: 512,
    },
    Case {
        num_messages:    8,
        message_length:  64,
        codeword_length: 1024,
    },
    Case {
        num_messages:    168,
        message_length:  512,
        codeword_length: 2048,
    },
    Case {
        num_messages:    8,
        message_length:  256,
        codeword_length: 65536,
    },
    Case {
        num_messages:    8,
        message_length:  2048,
        codeword_length: 131072,
    },
    Case {
        num_messages:    8,
        message_length:  16384,
        codeword_length: 262144,
    },
    Case {
        num_messages:    8,
        message_length:  131072,
        codeword_length: 524288,
    },
];

fn num_groups(case: Case) -> usize {
    case.codeword_length / case.message_length.next_power_of_two()
}

fn make_values(case: Case) -> Vec<Fr> {
    let mut rng = StdRng::seed_from_u64(
        (case.num_messages as u64)
            ^ ((case.message_length as u64) << 16)
            ^ (case.codeword_length as u64),
    );
    (0..case.num_messages * case.codeword_length)
        .map(|_| Fr::rand(&mut rng))
        .collect()
}

#[divan::bench(args = TEST_CASES)]
fn cpu_ntt(bencher: Bencher, case: &Case) {
    bencher
        .with_inputs(|| make_values(*case))
        .bench_values(|mut values| {
            ntt_nr(&mut values, case.codeword_length, num_groups(*case));
            black_box(values);
        });
}

#[cfg(feature = "cuda")]
#[divan::bench(args = TEST_CASES)]
fn cuda_ntt(bencher: Bencher, case: &Case) {
    bencher
        .with_inputs(|| make_values(*case))
        .bench_values(|mut values| {
            ntt_nr_cuda(&mut values, case.codeword_length, num_groups(*case)).unwrap();
            black_box(values);
        });
}

fn main() {
    rayon::ThreadPoolBuilder::new().build_global().unwrap();
    divan::main();
}

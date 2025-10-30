use {
    ark_bn254::Fr,
    ark_std::UniformRand,
    divan,
    ntt::{transpose, transpose_tiled},
};

fn main() {
    rayon::ThreadPoolBuilder::new().build_global().unwrap();

    divan::main();
}

const MATRIX_SIZES: [usize; 4] = [6, 8, 10, 12];

#[divan::bench(args = MATRIX_SIZES)]
fn bench_transpose(bencher: divan::Bencher, log_size: &usize) {
    let size = 1 << log_size;
    bencher
        .with_inputs(|| {
            let mut rng = ark_std::test_rng();
            let matrix: Vec<Fr> = (0..(size * size)).map(|_| Fr::rand(&mut rng)).collect();
            matrix
        })
        .bench_refs(|matrix| {
            let _transposed = transpose(matrix, size, size);
        });
}

#[divan::bench(args = MATRIX_SIZES)]
fn bench_transpose_parallel(bencher: divan::Bencher, log_size: &usize) {
    let size = 1 << log_size;
    bencher
        .with_inputs(|| {
            let mut rng = ark_std::test_rng();
            let matrix: Vec<Fr> = (0..(size * size)).map(|_| Fr::rand(&mut rng)).collect();
            matrix
        })
        .bench_refs(|matrix| {
            let _transposed = transpose_tiled(matrix, size, size);
        });
}

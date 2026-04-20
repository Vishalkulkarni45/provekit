//! Cost of arkworks `Fr` ↔ icicle `ScalarField` round-trip conversion.
//! This is the tax we'd pay per NTT call if we replaced the in-tree kernel.

#![cfg(feature = "cuda-icicle")]

use {
    ark_bn254::Fr,
    ark_ff::{BigInteger, PrimeField, UniformRand},
    icicle_bn254::curve::ScalarField as IcicleScalar,
    icicle_core::traits::FieldImpl,
    rayon::prelude::{IntoParallelRefIterator, ParallelIterator},
    std::time::Instant,
};

fn ark_to_icicle(src: &[Fr]) -> Vec<IcicleScalar> {
    src.par_iter()
        .map(|fr| {
            let bytes = fr.into_bigint().to_bytes_le();
            IcicleScalar::from_bytes_le(&bytes)
        })
        .collect()
}

fn icicle_to_ark(src: &[IcicleScalar]) -> Vec<Fr> {
    src.par_iter()
        .map(|sf| {
            let bytes = sf.to_bytes_le();
            Fr::from_le_bytes_mod_order(&bytes)
        })
        .collect()
}

fn main() {
    println!("# Fr ↔ ScalarField round-trip conversion cost\n# mean of 5 runs, ms\n");
    println!("{:>12}  {:>12}  {:>14}  {:>14}", "elements", "MB", "ark→icicle", "icicle→ark");
    println!("{}", "-".repeat(60));

    let mut rng = ark_std::test_rng();
    for &n in &[1 << 10, 1 << 14, 1 << 17, 1 << 20, 1 << 22, 1 << 23] {
        let bytes_mb = (n * 32) as f64 / (1024.0 * 1024.0);
        let ark: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        // warmup
        let iciclev = ark_to_icicle(&ark);
        let _back = icicle_to_ark(&iciclev);

        let runs = 5;
        let mut t1 = 0.0;
        for _ in 0..runs {
            let start = Instant::now();
            let _: Vec<_> = ark_to_icicle(&ark);
            t1 += start.elapsed().as_secs_f64();
        }
        let mut t2 = 0.0;
        for _ in 0..runs {
            let start = Instant::now();
            let _: Vec<_> = icicle_to_ark(&iciclev);
            t2 += start.elapsed().as_secs_f64();
        }
        println!(
            "{:>12}  {:>12.1}  {:>12.3} ms  {:>12.3} ms",
            n,
            bytes_mb,
            t1 * 1000.0 / runs as f64,
            t2 * 1000.0 / runs as f64
        );
    }
}

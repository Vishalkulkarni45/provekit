use {
    super::Command,
    anyhow::{Context, Result},
    argh::FromArgs,
    provekit_common::{
        file::{read, write},
        preflight_cuda_backend, set_ntt_backend, NttBackend, Prover,
    },
    provekit_prover::Prove,
    std::path::PathBuf,
    tracing::{info, instrument},
};
#[cfg(test)]
use {provekit_common::Verifier, provekit_verifier::Verify};

/// Prove a prepared Noir program
#[derive(FromArgs, PartialEq, Eq, Debug)]
#[argh(subcommand, name = "prove")]
pub struct Args {
    /// use the CUDA NTT backend for proving; errors if CUDA support is
    /// unavailable
    #[argh(switch)]
    cuda: bool,

    /// path to the prepared proof scheme
    #[argh(positional)]
    prover_path: PathBuf,

    #[cfg(test)]
    /// path to the verifier
    #[argh(positional)]
    verifier_path: PathBuf,

    /// path to the input values
    #[argh(positional)]
    input_path: PathBuf,

    /// path to store proof file
    #[argh(
        option,
        long = "out",
        short = 'o',
        default = "PathBuf::from(\"./proof.np\")"
    )]
    proof_path: PathBuf,
}

impl Command for Args {
    #[instrument(skip_all)]
    fn run(&self) -> Result<()> {
        // Overlap CUDA context init with the (~500 ms) Xz decompression of
        // prover.pkp. `preflight_cuda_backend` is idempotent; `set_ntt_backend`
        // only updates the requested backend state, so it is safe to run on
        // the main thread before spawning.
        let preflight = if self.cuda {
            set_ntt_backend(NttBackend::Cuda)
                .context("while selecting the CUDA NTT backend for `prove`")?;
            Some(std::thread::spawn(|| preflight_cuda_backend()))
        } else {
            None
        };

        // Read the scheme
        let prover: Prover = read(&self.prover_path).context("while reading Provekit Prover")?;
        let (constraints, witnesses) = prover.size();
        info!(constraints, witnesses, "Read Noir proof scheme");

        if let Some(handle) = preflight {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("CUDA preflight thread panicked"))?
                .context("while preflighting the CUDA NTT backend for `prove --cuda`")?;
        }

        // Generate the proof
        let proof = prover
            .prove_with_toml(&self.input_path)
            .context("While proving Noir program statement")?;

        // Store the proof to file
        write(&proof, &self.proof_path).context("while writing proof")?;

        // Verify the proof (test-only; runs after write so we can move `proof`)
        #[cfg(test)]
        {
            let mut verifier: Verifier =
                read(&self.verifier_path).context("while reading Provekit Verifier")?;
            verifier
                .verify(&proof)
                .context("While verifying Noir proof")?;
        }

        Ok(())
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

    #[test]
    fn cuda_requires_cuda_feature() {
        let args = Args {
            cuda:          true,
            prover_path:   std::env::temp_dir().join("unused-prover.pkp"),
            verifier_path: std::env::temp_dir().join("unused-verifier.pkv"),
            input_path:    std::env::temp_dir().join("unused-prover.toml"),
            proof_path:    std::env::temp_dir().join("unused-proof.np"),
        };

        let err = args
            .run()
            .expect_err("`prove --cuda` should fail without the `cuda` feature");
        assert!(
            err.to_string().contains("built without the `cuda` feature"),
            "unexpected error: {err:#}"
        );
    }
}

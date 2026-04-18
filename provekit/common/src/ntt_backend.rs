#[cfg(feature = "cuda")]
use anyhow::Context;
use {
    crate::{skyscraper, FieldElement},
    anyhow::{ensure, Result},
    std::sync::{Arc, LazyLock, Mutex},
    whir::algebra::ntt::ReedSolomon,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NttBackend {
    Cpu,
    Cuda,
}

impl std::fmt::Display for NttBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda => f.write_str("cuda"),
        }
    }
}

#[derive(Debug)]
struct NttBackendState {
    requested:  NttBackend,
    registered: Option<NttBackend>,
}

static NTT_BACKEND_STATE: LazyLock<Mutex<NttBackendState>> = LazyLock::new(|| {
    Mutex::new(NttBackendState {
        requested:  NttBackend::Cpu,
        registered: None,
    })
});

pub fn requested_ntt_backend() -> NttBackend {
    NTT_BACKEND_STATE.lock().unwrap().requested
}

pub fn selected_ntt_backend() -> NttBackend {
    let state = NTT_BACKEND_STATE.lock().unwrap();
    state.registered.unwrap_or(state.requested)
}

pub fn set_ntt_backend(backend: NttBackend) -> Result<()> {
    let mut state = NTT_BACKEND_STATE.lock().unwrap();

    if let Some(registered) = state.registered {
        ensure!(
            registered == backend,
            "the NTT backend was already initialized as `{registered}`; it cannot be switched to \
             `{backend}` in the same process"
        );
        return Ok(());
    }

    state.requested = backend;
    Ok(())
}

pub fn preflight_cuda_backend() -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        ntt::preflight_cuda().context("CUDA NTT requested, but CUDA initialization failed")
    }

    #[cfg(not(feature = "cuda"))]
    {
        anyhow::bail!("CUDA NTT requested, but this binary was built without the `cuda` feature")
    }
}

fn build_registered_ntt_backend(backend: NttBackend) -> Result<Arc<dyn ReedSolomon<FieldElement>>> {
    match backend {
        NttBackend::Cpu => {
            #[cfg(not(feature = "provekit_ntt"))]
            let ntt: Arc<dyn ReedSolomon<FieldElement>> =
                Arc::new(whir::algebra::ntt::NttEngine::<FieldElement>::new_from_fftfield());

            #[cfg(feature = "provekit_ntt")]
            let ntt: Arc<dyn ReedSolomon<FieldElement>> = Arc::new(crate::ntt::RSFr);

            Ok(ntt)
        }
        NttBackend::Cuda => {
            preflight_cuda_backend()?;

            #[cfg(feature = "cuda")]
            {
                Ok(Arc::new(crate::ntt::RSFrCuda))
            }

            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("preflight_cuda_backend already returns an error without `cuda`")
            }
        }
    }
}

pub(crate) fn register_ntt() -> Result<()> {
    let mut state = NTT_BACKEND_STATE.lock().unwrap();
    if state.registered.is_some() {
        return Ok(());
    }

    let backend = state.requested;
    let ntt = build_registered_ntt_backend(backend)?;
    whir::algebra::ntt::NTT.insert(ntt);

    // Register Skyscraper (ProveKit-specific); WHIR's built-in engines
    // (SHA2, Keccak, Blake3, etc.) are pre-registered via whir::hash::ENGINES.
    whir::hash::ENGINES.register(Arc::new(skyscraper::SkyscraperHashEngine));
    state.registered = Some(backend);

    Ok(())
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

    #[test]
    fn preflight_cuda_backend_errors_without_cuda_feature() {
        let err =
            preflight_cuda_backend().expect_err("CUDA preflight should fail without the feature");
        assert!(
            err.to_string().contains("built without the `cuda` feature"),
            "unexpected error: {err:#}"
        );
    }
}

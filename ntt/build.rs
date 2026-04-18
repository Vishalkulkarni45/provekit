use std::{
    env,
    path::PathBuf,
    process::{Command, Stdio},
};

fn configured_cuda_arch() -> Option<String> {
    env::var("PROVEKIT_CUDA_ARCH")
        .ok()
        .or_else(|| env::var("CUDA_ARCH").ok())
        .map(|arch| arch.trim().replace('.', ""))
        .filter(|arch| !arch.is_empty())
}

fn detected_cuda_arch() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .stderr(Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let arch = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .replace('.', "");

    (!arch.is_empty()).then_some(arch)
}

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/interleaved_ntt.cu");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let nvcc = env::var_os("NVCC")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("CUDA_HOME")
                .map(PathBuf::from)
                .map(|p| p.join("bin/nvcc"))
        })
        .or_else(|| {
            env::var_os("CUDA_PATH")
                .map(PathBuf::from)
                .map(|p| p.join("bin/nvcc"))
        })
        .unwrap_or_else(|| PathBuf::from("nvcc"));

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let object = out_dir.join("interleaved_ntt.o");
    let archive = out_dir.join("libprovekit_cuda_ntt.a");
    let cuda_arch = configured_cuda_arch()
        .or_else(detected_cuda_arch)
        .unwrap_or_else(|| "60".to_string());

    let compute = format!("compute_{cuda_arch}");
    let sm = format!("sm_{cuda_arch}");

    let compile_status = Command::new(&nvcc)
        .args(["-c", "-O3", "--std=c++17", "-Xcompiler", "-fPIC"])
        .args([
            "-gencode",
            &format!("arch={compute},code={sm}"),
            "-gencode",
            &format!("arch={compute},code={compute}"),
            "src/cuda/interleaved_ntt.cu",
            "-o",
        ])
        .arg(&object)
        .status()
        .unwrap_or_else(|err| panic!("failed to invoke `{}`: {err}", nvcc.display()));

    if !compile_status.success() {
        panic!(
            "nvcc failed while compiling src/cuda/interleaved_ntt.cu; ensure the CUDA toolkit is \
             installed and `nvcc` is on PATH"
        );
    }

    let archive_status = Command::new("ar")
        .args(["crus"])
        .arg(&archive)
        .arg(&object)
        .status()
        .expect("failed to invoke `ar` while archiving the CUDA object file");

    if !archive_status.success() {
        panic!("`ar` failed while archiving the CUDA object file");
    }

    let cuda_lib_dir = env::var_os("CUDA_LIB_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("CUDA_HOME")
                .map(PathBuf::from)
                .map(|p| p.join("lib64"))
        })
        .or_else(|| {
            env::var_os("CUDA_PATH")
                .map(PathBuf::from)
                .map(|p| p.join("lib/x64"))
        })
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda/lib64"));

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    println!("cargo:rustc-link-lib=static=provekit_cuda_ntt");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:warning=building CUDA NTT kernels for compute capability {cuda_arch}");
}

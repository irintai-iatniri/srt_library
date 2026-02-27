//! Build script for syntonic-core
//!
//! Compiles CUDA kernels to a static library and links them.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile CUDA if the cuda feature is enabled
    if !cfg!(feature = "cuda") {
        return;
    }

    // Detect CUDA installation
    let cuda_path = detect_cuda_path();
    if cuda_path.is_none() {
        println!("cargo:warning=CUDA not found, skipping static kernel compilation");
        println!("cargo:warning=PTX-based kernels will still work at runtime");
        return;
    }
    let cuda_path = cuda_path.unwrap();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels");

    // Detect GPU architectures to compile for
    let architectures = vec!["75", "80", "86", "90"];

    // List of CUDA source files to compile
    // srt_constants.cu must be first as it defines shared constant memory
    let cuda_sources = vec![
        "srt_constants.cu",
        "elementwise.cu",  // Contains sin_toroidal_f64, cos_toroidal_f64, etc.
        "dhsr.cu",
        "gnosis.cu",
        "attractor.cu",
        "golden_ops.cu",
        "matmul.cu",
        "scatter_gather_srt.cu",
        "trilinear.cu",
        "complex_ops.cu",
        "host_wrappers.cu",
    ];

    // Compile each source file to object files
    let mut objects = Vec::new();
    for source in &cuda_sources {
        let source_path = kernel_dir.join(source);
        if !source_path.exists() {
            println!("cargo:warning=CUDA source not found: {}", source_path.display());
            continue;
        }

        let obj_name = source.replace(".cu", ".o");
        let obj_path = out_dir.join(&obj_name);

        // Build nvcc command
        let nvcc = cuda_path.join("bin").join("nvcc");
        let mut cmd = Command::new(&nvcc);

        cmd.arg("-c")
            .arg(&source_path)
            .arg("-o")
            .arg(&obj_path)
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-I")
            .arg(&kernel_dir);

        // Add architecture flags
        for arch in &architectures {
            cmd.arg(format!("-gencode=arch=compute_{},code=sm_{}", arch, arch));
        }

        // Enable device code linking for cross-file kernel calls
        cmd.arg("-dc");

        println!("cargo:rerun-if-changed={}", source_path.display());

        let output = cmd.output().expect("Failed to run nvcc");
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("cargo:warning=nvcc compilation failed for {}: {}", source, stderr);
            // Continue with other files, PTX fallback will handle missing kernels
            continue;
        }

        objects.push(obj_path);
    }

    if objects.is_empty() {
        println!("cargo:warning=No CUDA objects compiled, using PTX-only mode");
        return;
    }

    // Link device code
    let dlink_path = out_dir.join("dlink.o");
    let nvcc = cuda_path.join("bin").join("nvcc");
    let mut dlink_cmd = Command::new(&nvcc);

    dlink_cmd.arg("-dlink");
    for obj in &objects {
        dlink_cmd.arg(obj);
    }
    dlink_cmd.arg("-o").arg(&dlink_path);

    for arch in &architectures {
        dlink_cmd.arg(format!("-gencode=arch=compute_{},code=sm_{}", arch, arch));
    }

    let dlink_output = dlink_cmd.output().expect("Failed to run nvcc dlink");
    if !dlink_output.status.success() {
        let stderr = String::from_utf8_lossy(&dlink_output.stderr);
        println!("cargo:warning=Device link failed: {}", stderr);
        return;
    }

    // Create static library
    let lib_path = out_dir.join("libsrt_cuda_kernels.a");
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("rcs").arg(&lib_path);
    for obj in &objects {
        ar_cmd.arg(obj);
    }
    ar_cmd.arg(&dlink_path);

    let ar_output = ar_cmd.output().expect("Failed to run ar");
    if !ar_output.status.success() {
        let stderr = String::from_utf8_lossy(&ar_output.stderr);
        panic!("ar failed: {}", stderr);
    }

    // Link against the static library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=srt_cuda_kernels");

    // Link CUDA runtime
    let cuda_lib = cuda_path.join("lib64");
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=cudart");

    // Also link against stdc++ for CUDA runtime
    println!("cargo:rustc-link-lib=stdc++");

    println!("cargo:rerun-if-changed=kernels/host_wrappers.cu");
    println!("cargo:rerun-if-changed=kernels/srt_constants.cuh");
}

/// Detect CUDA installation path
fn detect_cuda_path() -> Option<PathBuf> {
    // Check CUDA_PATH environment variable first
    if let Ok(path) = env::var("CUDA_PATH") {
        let p = PathBuf::from(path);
        if p.join("bin").join("nvcc").exists() {
            return Some(p);
        }
    }

    // Check CUDA_HOME
    if let Ok(path) = env::var("CUDA_HOME") {
        let p = PathBuf::from(path);
        if p.join("bin").join("nvcc").exists() {
            return Some(p);
        }
    }

    // Check common installation paths
    let common_paths = vec![
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-11",
        "/opt/cuda",
    ];

    for path in common_paths {
        let p = PathBuf::from(path);
        if p.join("bin").join("nvcc").exists() {
            return Some(p);
        }
    }

    // Try to find nvcc in PATH
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let nvcc_path = String::from_utf8_lossy(&output.stdout);
            let nvcc_path = nvcc_path.trim();
            // nvcc is in cuda/bin/nvcc, so go up two directories
            if let Some(parent) = PathBuf::from(nvcc_path).parent() {
                if let Some(cuda_root) = parent.parent() {
                    return Some(cuda_root.to_path_buf());
                }
            }
        }
    }

    None
}

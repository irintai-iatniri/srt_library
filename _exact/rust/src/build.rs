fn main() {
    println!("cargo:rerun-if-changed=src/exact/");
    println!("cargo:rerun-if-changed=scripts/generate_srt_constants.rs");

    // Generate constants header
    let status = std::process::Command::new("cargo")
        .args(["run", "--bin", "generate_constants", "--release"])
        .status()
        .expect("Failed to generate constants");

    assert!(status.success(), "Constant generation failed");
}
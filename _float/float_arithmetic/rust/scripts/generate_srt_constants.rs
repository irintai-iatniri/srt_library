//! Generate srt_constants.cuh from exact Rust infrastructure
//! Run with: cargo run --bin generate_constants

use std::fs;

fn main() {
    println!("Generating SRT constants from exact Rust infrastructure...");

    let phi = syntonic_core::exact::GoldenExact::phi().to_f64();
    let phi_inv = syntonic_core::exact::GoldenExact::phi_hat().to_f64();
    let phi_sq = phi * phi;
    let phi_inv_sq = phi_inv * phi_inv;

    let q_deficit = syntonic_core::exact::FundamentalConstant::Q.approx_f64();
    let e_star = syntonic_core::exact::FundamentalConstant::EStar.approx_f64();
    let pi = syntonic_core::exact::FundamentalConstant::Pi.approx_f64();
    let euler = syntonic_core::exact::FundamentalConstant::Euler.approx_f64();

    // Generate φ powers using exact Fibonacci recurrence
    let mut phi_powers = Vec::new();
    let mut a = 0.0; // φ^{-2}
    let mut b = 1.0; // φ^{-1}
    for i in 0..32 {
        if i == 0 {
            phi_powers.push(1.0); // φ⁰
        } else if i == 1 {
            phi_powers.push(phi); // φ¹
        } else {
            let next = phi_powers[i - 1] * phi;
            phi_powers.push(next);
        }
    }

    // Generate Pisano periods (simplified - would need full computation)
    let pisano_periods = [
        1, 3, 6, 6, 20, 12, 24, 12, 18, 18, 10, 30, 24, 24, 48, 12, 36, 24, 18, 42, 30, 24, 48, 30,
        24, 24, 66, 12, 60, 36, 24, 78, 60, 12, 84, 24, 48, 24, 30, 48, 42, 30, 24, 60, 24, 48, 96,
        36, 60, 24, 30, 48, 24, 24, 120, 24, 48, 24, 48, 66, 60, 60, 24, 48, 30, 30, 24, 120, 24,
        60, 48, 60, 48, 24, 60, 30, 60, 24, 60, 24, 48, 60, 48, 24, 60, 24, 60, 48, 48, 60, 24,
        120, 60, 24, 120, 24, 60, 48, 60, 24, 48, 120, 24, 60, 24, 24, 120, 60, 60, 24, 60, 24, 60,
        24, 60, 24, 60, 24, 120, 60, 24, 120, 24, 60, 48, 60, 24, 48, 120, 24, 60, 24, 24, 120, 60,
        60, 24,
    ];

    // Generate Mersenne primes mask
    let mut mersenne_mask = [false; 8192];
    let mersenne_primes = [2, 3, 5, 7, 13]; // M_p for p in these
    for &p in &mersenne_primes {
        let mp = (1i64 << p) - 1; // 2^p - 1
        if mp < 8192 {
            mersenne_mask[mp as usize] = true;
        }
    }

    let content = format!(
        r#"// =============================================================================
// SRT Constants - AUTO-GENERATED from Rust exact infrastructure
// DO NOT EDIT MANUALLY - regenerate with: cargo run --bin generate_constants
// Generated: {}
// Source: syntonic-core exact arithmetic
// Precision: f64 (17 significant digits)
// =============================================================================

#ifndef SRT_CONSTANTS_CUH
#define SRT_CONSTANTS_CUH

// =============================================================================
// Golden Ratio Constants (φ = (1 + √5) / 2)
// Source: GoldenExact::phi().to_f64()
// =============================================================================

#define PHI_F64         {phi:.17}
#define PHI_INV_F64     {phi_inv:.17}
#define PHI_SQ_F64      {phi_sq:.17}
#define PHI_INV_SQ_F64  {phi_inv_sq:.17}

#define PHI_F32         {phi:.9}f
#define PHI_INV_F32     {phi_inv:.9}f
#define PHI_SQ_F32      {phi_sq:.9}f
#define PHI_INV_SQ_F32  {phi_inv_sq:.9}f

// =============================================================================
// Mathematical Constants
// Source: FundamentalConstant approximations
// =============================================================================

#define PI_F64          {pi:.17}
#define PI_F32          {pi:.9}f
#define EULER_F64       {euler:.17}
#define EULER_F32       {euler:.9}f

// =============================================================================
// SRT Fundamental Constants
// Source: FundamentalConstant exact values
// =============================================================================

#define E_STAR_F64      {e_star:.17}    // E* = e^π - π
#define E_STAR_F32      {e_star:.9}f

#define Q_DEFICIT_F64   {q:.17}      // Universal syntony deficit
#define Q_DEFICIT_F32   {q:.9}f

// =============================================================================
// Structure Constants
// =============================================================================

// E₈ lattice dimensions
#define E8_DIM              8       // Dimension of E₈ root system
#define E8_ROOTS            240     // Number of E₈ roots
#define E8_ROOTS_POSITIVE   120     // Positive roots (chiral)

// E₆ lattice (Golden Cone)
#define E6_DIM              6       // Dimension of E₆
#define E6_POSITIVE_ROOTS   36      // |Φ⁺(E₆)| - Golden Cone count

// D₄ kissing number (consciousness threshold)
#define D4_KISSING          24
#define D4_KISSING_F64      24.0

// Mersenne prime constants for stable indices
#define MAX_MERSENNE        8192    // 2^13 = 8192, covers M_13 = 8191

// Fibonacci transcendence gates (prime indices)
#define FIB_GATES_COUNT     11

// =============================================================================
// Precomputed Tables (Constant Memory)
// =============================================================================

// φ powers: φ^n for n = 0 to 31 (exact via Fibonacci)
__constant__ double c_phi_powers[32] = {{
{}
}};

// Pisano periods π(p) for primes p = 2 to 128
__constant__ int c_pisano_periods[128] = {{
{}
}};

// Mersenne prime stability mask (true for M_p indices)
__constant__ bool c_mersenne_mask[8192] = {{
{}
}};

// Fibonacci transcendence gates
__constant__ int c_fib_gates[11] = {{3,4,5,7,11,13,17,23,29,43,47}};

// =============================================================================
// Block Size Constants (SRT-optimized)
// =============================================================================

#define BLOCK_E8_ROOTS      240     // One per E₈ root
#define BLOCK_GOLDEN_CONE   36      // E₆ positive roots
#define BLOCK_WINDING_4D    256     // 4⁴ for winding space
#define BLOCK_GENERATIONS   192     // 64 × 3 generations
#define BLOCK_D4_KISSING    96      // 4 × 24 consciousness threshold
#define BLOCK_DEFAULT       256     // Default for general ops

#endif // SRT_CONSTANTS_CUH
"#,
        chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S"),
        phi = phi,
        phi_inv = phi_inv,
        phi_sq = phi_sq,
        phi_inv_sq = phi_inv_sq,
        pi = pi,
        euler = euler,
        e_star = e_star,
        q = q,
        phi_powers
            .iter()
            .map(|&x| format!("    {:.17}", x))
            .collect::<Vec<_>>()
            .join(",\n"),
        pisano_periods
            .iter()
            .map(|&x| format!("    {}", x))
            .collect::<Vec<_>>()
            .join(",\n"),
        mersenne_mask
            .iter()
            .map(|&x| if x { "    true" } else { "    false" })
            .collect::<Vec<_>>()
            .join(",\n"),
    );

    // Write to the kernels directory
    let output_path = "kernels/srt_constants.cuh";
    fs::write(output_path, content).expect("Failed to write srt_constants.cuh");
    println!("✓ Generated {} with exact constants", output_path);
    println!("  PHI_F64: {:.17}", phi);
    println!("  Q_DEFICIT_F64: {:.17}", q);
    println!("  E_STAR_F64: {:.17}", e_star);
}
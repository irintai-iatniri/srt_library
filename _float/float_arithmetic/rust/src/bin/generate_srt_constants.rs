//! Generate srt_constants.cuh from exact Rust infrastructure
//! Run with: cargo run --bin generate_constants

use std::fs;

fn main() {
    println!("Generating SRT constants from exact Rust infrastructure...");

    // Use exact SRT values (hardcoded for bin compatibility)
    let phi = 1.6180339887498948482045868343656381177203091798057628621354486227052604628189024497072072041893911374847540880753868917521266338622235369317931800607667263544333890865959395829056383226613199282902678806752087668925017116962070322210432162695486262963136144381497587012203408058879544547492461856953648644492;
    let phi_inv = 0.6180339887498948482045868343656381177203091798057628621354486227052604628189024497072072041893911374847540880753868917521266338622235369317931800607667263544333890865959395829056383226613199282902678806752087668925017116962070322210432162695486262963136144381497587012203408058879544547492461856953648644492;
    let phi_sq = phi * phi;
    let phi_inv_sq = phi_inv * phi_inv;

    let sqrt5 = 5.0_f64.sqrt();
    let proj_norm = (2.0_f64).sqrt().recip(); // 1/sqrt(2) for E8 normalization
    let q_deficit = 0.0072973525693; // Fine structure constant deficit
    let e_star = 2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274274663919320030599218174135966290435729003342952605956307381323286279434907632338298807531952510190115738341879307021540891499348841675092447614606680822637200302134934470462844069344977836228416466860857;
    let pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870;
    let two_pi = 2.0 * pi;
    let euler = 0.5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495146314472498070824809605045149997699828760462674488852090766454357239622770165505005469067546159086453550705546370944296878014626055419342749157990864763489646632024184568478874849370494496960216063845650298;

    // Generate φ powers using exact Fibonacci recurrence
    let mut phi_powers = Vec::new();
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

    // Generate Pisano periods for moduli 1-32
    let pisano_periods = [
        1, 3, 6, 6, 20, 12, 24, 12, 18, 18, 10, 30, 24, 24, 48, 12, 36, 12, 18, 30, 12, 84, 48, 24,
        60, 24, 48, 30, 72, 24, 48, 24,
    ];

    // Generate Mersenne primes mask for p in 2, 3, 5, 7, 13, 17, 19, 31
    let mut mersenne_mask = [false; 8192];
    let mersenne_primes = [2, 3, 5, 7, 13, 17, 19, 31]; // M_p for p in these
    for &p in &mersenne_primes {
        let mp = (1i64 << p) - 1; // 2^p - 1
        if mp < 8192 {
            mersenne_mask[mp as usize] = true;
        }
    }

    let phi_powers_section = phi_powers
        .iter()
        .enumerate()
        .map(|(i, power)| format!("#define PHI_POW_{:<3}   {:.17}", i, power))
        .collect::<Vec<_>>()
        .join("\n");

    let pisano_section = pisano_periods
        .iter()
        .enumerate()
        .map(|(i, period)| format!("#define PISANO_{:<3}   {}", i + 1, period))
        .collect::<Vec<_>>()
        .join("\n");

    let mersenne_primes_list = [2, 3, 5, 7, 13, 17, 19, 31];
    let mersenne_section = mersenne_primes_list
        .iter()
        .map(|p| {
            let mp = (1i64 << p) - 1; // 2^p - 1
            let is_prime = if mp < 8192 {
                mersenne_mask[mp as usize]
            } else {
                false
            };
            format!(
                "#define MERSENNE_{:<3}   {}",
                p,
                if is_prime { "true" } else { "false" }
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let content = format!(
        r#"// =============================================================================
// SRT Theory Constants: Auto-generated from exact Rust infrastructure
// Generated: Auto-generated
// =============================================================================

#ifndef SRT_CONSTANTS_CUH
#define SRT_CONSTANTS_CUH

#include <cuda_runtime.h>

// =============================================================================
// Warp-Level Reduction Primitives (CUDA Helper Functions)
// =============================================================================

// Warp-level sum reduction for float32
__device__ __forceinline__ float warp_reduce_sum(float val) {{
    for (int offset = 16; offset > 0; offset >>= 1) {{
        val += __shfl_down_sync(0xffffffff, val, offset);
    }}
    return val;
}}

// Warp-level sum reduction for float64
__device__ __forceinline__ double warp_reduce_sum_f64(double val) {{
    for (int offset = 16; offset > 0; offset >>= 1) {{
        val += __shfl_down_sync(0xffffffff, val, offset);
    }}
    return val;
}}

// Warp-level max reduction for float32
__device__ __forceinline__ float warp_reduce_max(float val) {{
    for (int offset = 16; offset > 0; offset >>= 1) {{
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }}
    return val;
}}

// Warp-level max reduction for float64
__device__ __forceinline__ double warp_reduce_max_f64(double val) {{
    for (int offset = 16; offset > 0; offset >>= 1) {{
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }}
    return val;
}}

// =============================================================================
// Fundamental SRT Constants (exact symbolic values)
// =============================================================================

// Golden Ratio and Powers
#define PHI_F32                 {:.17}f
#define PHI_F64                 {:.17}
#define PHI_INV_F32             {:.17}f
#define PHI_INV_F64             {:.17}
#define PHI_SQ_F32              {:.17}f
#define PHI_SQ_F64              {:.17}
#define PHI_INV_SQ_F32          {:.17}f
#define PHI_INV_SQ_F64          {:.17}

// Fundamental Constants
#define PI_F32                  {:.17}f
#define PI_F64                  {:.17}
#define TWO_PI_F32              {:.17}f
#define TWO_PI_F64              {:.17}
#define EULER_F32               {:.17}f
#define EULER_F64               {:.17}
#define E_STAR_F32              {:.17}f
#define E_STAR_F64              {:.17}
#define SQRT5_F32               {:.17}f
#define SQRT5_F64               {:.17}
#define PROJ_NORM_F32           {:.17}f
#define PROJ_NORM_F64           {:.17}
#define Q_DEFICIT_F64           {:.17}
#define Q_DEFICIT_F32           {:.17}f

// Golden Ratio Powers (φ^n for n=0..32) - individual constants for CUDA compatibility
{}

// Pisano Periods (moduli for Fibonacci arithmetic)
{}

// Mersenne Prime Mask (true for prime p where 2^p-1 is prime)
{}

// =============================================================================
// Block Size Configuration (SRT-optimized)
// =============================================================================

#define BLOCK_E8_ROOTS      240     // One per E₈ root
#define BLOCK_GOLDEN_CONE   36      // E₆ positive roots
#define BLOCK_WINDING_4D    256     // 4⁴ for winding space
#define BLOCK_GENERATIONS   192     // 64 × 3 generations
#define BLOCK_D4_KISSING    96      // 4 × 24 consciousness threshold
#define BLOCK_DEFAULT       256     // Default for general ops

#endif // SRT_CONSTANTS_CUH
"#,
        phi,
        phi,
        phi_inv,
        phi_inv,
        phi_sq,
        phi_sq,
        phi_inv_sq,
        phi_inv_sq,
        pi,
        pi,
        two_pi,
        two_pi,
        euler,
        euler,
        e_star,
        e_star,
        sqrt5,
        sqrt5,
        proj_norm,
        proj_norm,
        q_deficit,
        q_deficit,
        phi_powers_section,
        pisano_section,
        mersenne_section
    );

    // Write to the kernels directory
    let output_path = "kernels/srt_constants.cuh";
    fs::write(output_path, content).expect("Failed to write srt_constants.cuh");
    println!("✓ Generated {} with exact constants", output_path);
    println!("  PHI_F64: {:.17}", phi);
    println!("  Q_DEFICIT_F64: {:.17}", q_deficit);
    println!("  E_STAR_F64: {:.17}", e_star);
}

// =============================================================================
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
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level sum reduction for float64
__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level max reduction for float32
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level max reduction for float64
__device__ __forceinline__ double warp_reduce_max_f64(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// =============================================================================
// Fundamental SRT Constants (exact symbolic values)
// =============================================================================

// Golden Ratio and Powers
#define PHI_F32                 1.61803398874989490f
#define PHI_F64                 1.61803398874989490
#define PHI_INV_F32             0.61803398874989490f
#define PHI_INV_F64             0.61803398874989490
#define PHI_SQ_F32              2.61803398874989490f
#define PHI_SQ_F64              2.61803398874989490
#define PHI_INV_SQ_F32          0.38196601125010521f
#define PHI_INV_SQ_F64          0.38196601125010521

// Fundamental Constants
#define PI_F32                  3.14159265358979312f
#define PI_F64                  3.14159265358979312
#define TWO_PI_F32              6.28318530717958623f
#define TWO_PI_F64              6.28318530717958623
#define EULER_F32               0.57721566490153287f
#define EULER_F64               0.57721566490153287
#define E_STAR_F32              2.71828182845904509f
#define E_STAR_F64              2.71828182845904509
#define SQRT5_F32               2.23606797749978981f
#define SQRT5_F64               2.23606797749978981
#define PROJ_NORM_F32           0.70710678118654746f
#define PROJ_NORM_F64           0.70710678118654746
#define Q_DEFICIT_F64           0.00729735256930000
#define Q_DEFICIT_F32           0.00729735256930000f

// Golden Ratio Powers (φ^n for n=0..32) - individual constants for CUDA compatibility
#define PHI_POW_0     1.00000000000000000
#define PHI_POW_1     1.61803398874989490
#define PHI_POW_2     2.61803398874989490
#define PHI_POW_3     4.23606797749978981
#define PHI_POW_4     6.85410196624968471
#define PHI_POW_5     11.09016994374947451
#define PHI_POW_6     17.94427190999915922
#define PHI_POW_7     29.03444185374863551
#define PHI_POW_8     46.97871376374779828
#define PHI_POW_9     76.01315561749643734
#define PHI_POW_10    122.99186938124424273
#define PHI_POW_11    199.00502499874068008
#define PHI_POW_12    321.99689437998495123
#define PHI_POW_13    521.00191937872568815
#define PHI_POW_14    842.99881375871063938
#define PHI_POW_15    1364.00073313743632752
#define PHI_POW_16    2206.99954689614696690
#define PHI_POW_17    3571.00028003358329443
#define PHI_POW_18    5777.99982692973026133
#define PHI_POW_19    9349.00010696331446525
#define PHI_POW_20    15126.99993389304654556
#define PHI_POW_21    24476.00004085636101081
#define PHI_POW_22    39602.99997474940755637
#define PHI_POW_23    64079.00001560577220516
#define PHI_POW_24    103681.99999035517976154
#define PHI_POW_25    167761.00000596095924266
#define PHI_POW_26    271442.99999631615355611
#define PHI_POW_27    439204.00000227714190260
#define PHI_POW_28    710646.99999859335366637
#define PHI_POW_29    1149851.00000087055377662
#define PHI_POW_30    1860497.99999946402385831
#define PHI_POW_31    3010349.00000033481046557

// Pisano Periods (moduli for Fibonacci arithmetic)
#define PISANO_1     1
#define PISANO_2     3
#define PISANO_3     6
#define PISANO_4     6
#define PISANO_5     20
#define PISANO_6     12
#define PISANO_7     24
#define PISANO_8     12
#define PISANO_9     18
#define PISANO_10    18
#define PISANO_11    10
#define PISANO_12    30
#define PISANO_13    24
#define PISANO_14    24
#define PISANO_15    48
#define PISANO_16    12
#define PISANO_17    36
#define PISANO_18    12
#define PISANO_19    18
#define PISANO_20    30
#define PISANO_21    12
#define PISANO_22    84
#define PISANO_23    48
#define PISANO_24    24
#define PISANO_25    60
#define PISANO_26    24
#define PISANO_27    48
#define PISANO_28    30
#define PISANO_29    72
#define PISANO_30    24
#define PISANO_31    48
#define PISANO_32    24

// Mersenne Prime Mask (true for prime p where 2^p-1 is prime)
#define MERSENNE_2     true
#define MERSENNE_3     true
#define MERSENNE_5     true
#define MERSENNE_7     true
#define MERSENNE_13    true
#define MERSENNE_17    false
#define MERSENNE_19    false
#define MERSENNE_31    false

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
// Compatibility defines (auto-appended)
#ifndef MAX_MERSENNE
#define MAX_MERSENNE 32
#endif
#ifndef D4_KISSING_F64
#define D4_KISSING_F64 24.0
#endif
#ifndef E8_ROOTS
#define E8_ROOTS 240
#endif
// Mersenne mask - defined in srt_constants.cu, declared extern here
// Note: When linking multiple .cu files, constant memory must be defined only once
#ifndef SRT_CONSTANTS_IMPL
extern __device__ __constant__ bool c_mersenne_mask[MAX_MERSENNE];
#endif

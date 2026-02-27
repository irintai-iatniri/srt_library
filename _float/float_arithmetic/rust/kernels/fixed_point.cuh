#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// =============================================================================
// Exact Arithmetic Type Definition
// =============================================================================

// 64-bit Fixed Point: Q32.32
// Range: +/- 2.14 billion
// Precision: 2^-32 (approx 2.3e-10)
typedef int64_t fp64;

// Constants (must match Rust FixedPoint64 / spectral.rs exactly)
#define FP_SCALE 4294967296LL // 2^32

// Fixed-point constants (Q32.32)
#define SRT_FP_ZERO 0LL
#define SRT_FP_ONE (1LL << 32)
#define SRT_FP_HALF (1LL << 31)
#define SRT_FP_PHI 6949633539LL // 1.61803398875 * 2^32

// Aliases for compatibility
#define FP_ONE SRT_FP_ONE
#define FP_ZERO SRT_FP_ZERO
#define FP_HALF SRT_FP_HALF

// Constants from spectral.rs / libexact
// PI  = 3.1415926535... * 2^32 = 13493037704
#define FP_PI 13493037704LL
#define FP_PHI_INV 2654435769LL // 1/PHI * 2^32 = 0.6180339887 * 2^32

// CORDIC Constants
#define CORDIC_GAIN_INV 2608146603LL // 0.607252935 * 2^32

// =============================================================================
// Core Arithmetic (Device Functions)
// =============================================================================

__device__ __forceinline__ fp64 fp_add(fp64 a, fp64 b) { return a + b; }

__device__ __forceinline__ fp64 fp_sub(fp64 a, fp64 b) { return a - b; }

__device__ __forceinline__ fp64 fp_neg(fp64 a) { return -a; }

__device__ __forceinline__ fp64 fp_mul(fp64 a, fp64 b) {
  // 128-bit multiplication: (a * b) >> 32
  // We can use CUDA intrinsic __mul64hi for the high 64 bits of (a*b)
  // Low 64 bits: a*b
  // The result we want is (high << 32) | (low >> 32)

  // Note: __mul64hi(x, y) returns high 64 bits of x*y
  long long high = __mul64hi(a, b);
  unsigned long long low = (unsigned long long)a * (unsigned long long)b;

  // Result = (high << 32) + (low >> 32)
  // This assumes standard 2s complement behavior for signed multiply
  // Handling sign explicitly might be safer for cross-checking but intrinsics
  // usually handle it. However, __mul64hi is signed.

  // Shift logic:
  // We want the middle 64 bits of the 128-bit product (Q32.32 * Q32.32 ->
  // Q64.64, take bits [32:95])

  unsigned long long result = ((unsigned long long)high << 32) | (low >> 32);
  return (fp64)result;
}

__device__ __forceinline__ fp64 fp_div(fp64 a, fp64 b) {
  if (b == 0)
    return 0; // Check

// Division: (a << 32) / b
// We need 128-bit numerator.
// Use manual 128-bit construction or double-cast hack (unsafe) -> NO.
// Must be exact.

// Construct 128 bit a << 32
// Since __int128_t is supported in CUDA > 9 on device code usually
#if defined(__CUDA_ARCH__)
  __int128_t num = (__int128_t)a << 32;
  __int128_t den = (__int128_t)b;
  return (fp64)(num / den);
#else
  // CPU fallback (if compiled for host for checking)
  // Just return 0 or approx
  return 0;
#endif
}

// =============================================================================
// Conversion
// =============================================================================

__device__ __forceinline__ fp64 float_to_fp(float v) {
  return (fp64)(v * 4294967296.0f);
}

__device__ __forceinline__ fp64 double_to_fp(double v) {
  return (fp64)(v * 4294967296.0);
}

__device__ __forceinline__ float fp_to_float(fp64 v) {
  return (float)v / 4294967296.0f;
}

__device__ __forceinline__ double fp_to_double(fp64 v) {
  return (double)v / 4294967296.0;
}

// =============================================================================
// Transcendental Stub / Implementation
// =============================================================================

// Basic exp implementation (Taylor Series) for negative inputs (Heat kernels)
__device__ __forceinline__ static fp64 fp_exp(fp64 x) {
  if (x == 0)
    return FP_ONE;
  if (x > 0) {
    // e^x = 1 / e^(-x)
    fp64 neg = fp_exp(-x);
    return fp_div(FP_ONE, neg);
  }

  // x is negative
  // Range reduction: if x is large negative, return 0
  if (x < -20 * FP_ONE)
    return 0;

  // Taylor series: 1 + x + x^2/2 + ...
  // To make it converge fast, reduce x
  fp64 val = x;
  int k = 0;
  while (val < -FP_ONE) {
    val = val >> 1; // div by 2
    k++;
  }

  fp64 res = FP_ONE;
  fp64 term = FP_ONE;

  for (int i = 1; i < 15; i++) {
    fp64 i_fp = i * FP_ONE;
    term = fp_div(fp_mul(term, val), i_fp);
    res = fp_add(res, term);
    if (term > -100 && term < 100)
      break;
  }

  // Squaring
  for (int i = 0; i < k; i++) {
    res = fp_mul(res, res);
  }

  return res;
}

__device__ __forceinline__ static fp64 fp_sqrt(fp64 x) {
  if (x <= 0)
    return 0;

  // Newton Raphson
  // Initial guess
  fp64 y = (x >> 1) + FP_ONE;
  if (y > x)
    y = x;

  for (int i = 0; i < 10; i++) {
    fp64 div = fp_div(x, y);
    y = (y + div) >> 1;
  }
  return y;
}

// CORDIC Constants
#define FP_HALF_PI 6746518852LL
#define FP_TWO_PI 26986075409LL

__device__ const fp64 CORDIC_TABLE[] = {
    3373259426LL, 1991351317LL, 1052175346LL, 534100634LL, 268086747LL,
    134174062LL,  67103403LL,   33553749LL,   16777130LL,  8388597LL,
    4194302LL,    2097151LL,    1048575LL,    524287LL,    262143LL,
    131071LL,     65535LL,      32767LL,      16383LL,     8191LL,
    4095LL,       2047LL,       1023LL,       511LL,       255LL,
    127LL,        63LL,         32LL,         16LL,        8LL,
    4LL,          2LL};

__device__ __forceinline__ static void fp_sin_cos(fp64 angle, fp64 *sin_out, fp64 *cos_out) {
  fp64 z = angle;
  // Range reduction to [-PI, PI]
  while (z > FP_PI)
    z -= FP_TWO_PI;
  while (z < -FP_PI)
    z += FP_TWO_PI;

  // Quadrant mapping
  int quadrant = 0;
  if (z > FP_HALF_PI) {
    z = FP_PI - z;
    quadrant = 1;
  } else if (z < -FP_HALF_PI) {
    z = -FP_PI - z;
    quadrant = 1;
  }

  fp64 x = CORDIC_GAIN_INV;
  fp64 y = 0;

  for (int i = 0; i < 32; i++) {
    fp64 x_shift = x >> i;
    fp64 y_shift = y >> i;
    fp64 angle_val = CORDIC_TABLE[i];

    if (z >= 0) {
      x -= y_shift;
      y += x_shift;
      z -= angle_val;
    } else {
      x += y_shift;
      y -= x_shift;
      z += angle_val;
    }
  }

  if (quadrant == 1) {
    x = -x;
  }

  *sin_out = y;
  *cos_out = x;
}

__device__ __forceinline__ static fp64 fp_sin(fp64 angle) {
  fp64 s, c;
  fp_sin_cos(angle, &s, &c);
  return s;
}

__device__ __forceinline__ static fp64 fp_cos(fp64 angle) {
  fp64 s, c;
  fp_sin_cos(angle, &s, &c);
  return c;
}

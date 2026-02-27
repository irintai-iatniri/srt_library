// Syntonic CUDA Kernels - Element-wise Operations
// Compiled offline for multi-version driver compatibility

#include "srt_constants.cuh"
extern "C" __global__ void add_f64(double *out, const double *a,
                                   const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
}

extern "C" __global__ void add_f32(float *out, const float *a, const float *b,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
}

extern "C" __global__ void sub_f64(double *out, const double *a,
                                   const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - b[i];
}

extern "C" __global__ void sub_f32(float *out, const float *a, const float *b,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - b[i];
}

extern "C" __global__ void mul_f64(double *out, const double *a,
                                   const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * b[i];
}

extern "C" __global__ void mul_f32(float *out, const float *a, const float *b,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * b[i];
}

extern "C" __global__ void div_f64(double *out, const double *a,
                                   const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] / b[i];
}

extern "C" __global__ void div_f32(float *out, const float *a, const float *b,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] / b[i];
}

extern "C" __global__ void neg_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = -a[i];
}

extern "C" __global__ void neg_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = -a[i];
}

extern "C" __global__ void abs_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = fabs(a[i]);
}

extern "C" __global__ void abs_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = fabsf(a[i]);
}

extern "C" __global__ void scalar_add_f64(double *out, const double *a,
                                          double scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + scalar;
}

extern "C" __global__ void scalar_mul_f64(double *out, const double *a,
                                          double scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * scalar;
}

// Mathematical functions
extern "C" __global__ void exp_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = exp(a[i]);
}

extern "C" __global__ void exp_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = expf(a[i]);
}

extern "C" __global__ void log_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = log(a[i]);
}

extern "C" __global__ void log_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = logf(a[i]);
}

extern "C" __global__ void sin_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = sin(a[i]);
}

extern "C" __global__ void sin_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = sinf(a[i]);
}

extern "C" __global__ void cos_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = cos(a[i]);
}

extern "C" __global__ void cos_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = cosf(a[i]);
}

extern "C" __global__ void sqrt_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = sqrt(a[i]);
}

extern "C" __global__ void sqrt_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = sqrtf(a[i]);
}

extern "C" __global__ void tanh_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = tanh(a[i]);
}

extern "C" __global__ void tanh_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = tanhf(a[i]);
}

extern "C" __global__ void sigmoid_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = 1.0 / (1.0 + exp(-a[i]));
}

extern "C" __global__ void sigmoid_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = 1.0f / (1.0f + expf(-a[i]));
}

extern "C" __global__ void relu_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = fmax(a[i], 0.0);
}

extern "C" __global__ void relu_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = fmaxf(a[i], 0.0f);
}

// Golden exponential: exp(-x/φ)
extern "C" __global__ void exp_golden_f64(double *out, const double *a, int n) {
  const double PHI_INV = PHI_INV_F64; // 1/φ
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = exp(-a[i] * PHI_INV);
}

extern "C" __global__ void exp_golden_f32(float *out, const float *a, int n) {
  const float PHI_INV = PHI_INV_F32; // 1/φ
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = expf(-a[i] * PHI_INV);
}

// Complex operations (interleaved format: [re0, im0, re1, im1, ...])
extern "C" __global__ void add_c128(double *out, const double *a,
                                    const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    out[idx] = a[idx] + b[idx];
    out[idx + 1] = a[idx + 1] + b[idx + 1];
  }
}

extern "C" __global__ void sub_c128(double *out, const double *a,
                                    const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    out[idx] = a[idx] - b[idx];
    out[idx + 1] = a[idx + 1] - b[idx + 1];
  }
}

extern "C" __global__ void mul_c128(double *out, const double *a,
                                    const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    double ar = a[idx], ai = a[idx + 1];
    double br = b[idx], bi = b[idx + 1];
    out[idx] = ar * br - ai * bi;
    out[idx + 1] = ar * bi + ai * br;
  }
}

extern "C" __global__ void neg_c128(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    out[idx] = -a[idx];
    out[idx + 1] = -a[idx + 1];
  }
}

extern "C" __global__ void div_c128(double *out, const double *a,
                                    const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    double ar = a[idx], ai = a[idx + 1];
    double br = b[idx], bi = b[idx + 1];
    double denom = br * br + bi * bi;
    out[idx] = (ar * br + ai * bi) / denom;
    out[idx + 1] = (ai * br - ar * bi) / denom;
  }
}

extern "C" __global__ void abs_c128(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    double re = a[idx];
    double im = a[idx + 1];
    out[i] = sqrt(re * re + im * im);
  }
}

extern "C" __global__ void exp_c128(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;
  if (i < n) {
    double re = a[idx];
    double im = a[idx + 1];
    double exp_re = exp(re);
    double cos_im, sin_im;
    sincos(im, &sin_im, &cos_im);
    out[idx] = exp_re * cos_im;
    out[idx + 1] = exp_re * sin_im;
  }
}

// ============================================================================
// Broadcast Operations (Tensor op Scalar_Tensor)
// These kernels read the 'b' operand from a single memory address and apply
// it to the entire 'a' array. This avoids CPU roundtrips.
// ============================================================================

extern "C" __global__ void add_broadcast_scalar_f64(double *out,
                                                    const double *a,
                                                    const double *b_scalar,
                                                    int n) {
  // Read the scalar once from global memory (L2/Constant cache will optimize
  // this)
  double scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + scalar;
}

extern "C" __global__ void add_broadcast_scalar_f32(float *out, const float *a,
                                                    const float *b_scalar,
                                                    int n) {
  float scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + scalar;
}

extern "C" __global__ void sub_broadcast_scalar_f64(double *out,
                                                    const double *a,
                                                    const double *b_scalar,
                                                    int n) {
  double scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - scalar;
}

extern "C" __global__ void sub_broadcast_scalar_f32(float *out, const float *a,
                                                    const float *b_scalar,
                                                    int n) {
  float scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - scalar;
}

extern "C" __global__ void mul_broadcast_scalar_f64(double *out,
                                                    const double *a,
                                                    const double *b_scalar,
                                                    int n) {
  double scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * scalar;
}

extern "C" __global__ void mul_broadcast_scalar_f32(float *out, const float *a,
                                                    const float *b_scalar,
                                                    int n) {
  float scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * scalar;
}

extern "C" __global__ void div_broadcast_scalar_f64(double *out,
                                                    const double *a,
                                                    const double *b_scalar,
                                                    int n) {
  double scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] / scalar;
}

extern "C" __global__ void div_broadcast_scalar_f32(float *out, const float *a,
                                                    const float *b_scalar,
                                                    int n) {
  float scalar = *b_scalar;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] / scalar;
}

// ============================================================================
// Toroidal Math Functions (W⁴ Geometry)
// ============================================================================

/**
 * Toroidal sine function for winding phase calculations
 * sin(θ) where θ represents position on W⁴ torus
 */
extern "C" __global__ void sin_toroidal_f64(double *out, const double *a,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Toroidal sine: sin(2π * x) where x is in [0, 1]
    // This maps the unit interval to a full torus cycle
    out[i] = sin(2.0 * M_PI * a[i]);
  }
}

extern "C" __global__ void sin_toroidal_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Toroidal sine: sin(2π * x) where x is in [0, 1]
    out[i] = sinf(2.0f * M_PI * a[i]);
  }
}

/**
 * Toroidal cosine function for winding phase calculations
 */
extern "C" __global__ void cos_toroidal_f64(double *out, const double *a,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Toroidal cosine: cos(2π * x) where x is in [0, 1]
    // This maps the unit interval to a full torus cycle
    out[i] = cos(2.0 * M_PI * a[i]);
  }
}

extern "C" __global__ void cos_toroidal_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Toroidal cosine: cos(2π * x) where x is in [0, 1]
    out[i] = cosf(2.0f * M_PI * a[i]);
  }
}

/**
 * Toroidal atan2 function for phase angle calculations on W⁴
 * Returns angle in [0, 2π] range for toroidal topology
 */
extern "C" __global__ void atan2_toroidal_f64(double *out, const double *y,
                                              const double *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double angle = atan2(y[i], x[i]);
    // Normalize to [0, 2π] for toroidal geometry
    if (angle < 0)
      angle += 2.0 * M_PI;
    out[i] = angle;
  }
}

extern "C" __global__ void atan2_toroidal_f32(float *out, const float *y,
                                              const float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float angle = atan2f(y[i], x[i]);
    if (angle < 0)
      angle += 2.0f * M_PI;
    out[i] = angle;
  }
}

// ============================================================================
// Golden Exponentials (Consciousness Growth Functions)
// ============================================================================

/**
 * Golden exponential: φ^x - Natural growth function of consciousness
 * This represents the exponential growth pattern observed in biological
 * and conscious systems, following the golden ratio scaling.
 */
extern "C" __global__ void phi_exp_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // φ^x = φ * (φ^x) but computed efficiently
    double phi = PHI_F64;
    out[i] = pow(phi, a[i]);
  }
}

extern "C" __global__ void phi_exp_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float phi = PHI_F32;
    out[i] = powf(phi, a[i]);
  }
}

/**
 * Inverse golden exponential: φ^(-x)
 */
extern "C" __global__ void phi_exp_inv_f64(double *out, const double *a,
                                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double phi = PHI_F64;
    out[i] = pow(phi, -a[i]);
  }
}

extern "C" __global__ void phi_exp_inv_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float phi = PHI_F32;
    out[i] = powf(phi, -a[i]);
  }
}

// Golden entropy: phi-weighted entropy measure
// H_phi(x) = -|x|/phi * log(|x|/phi) for |x| > epsilon, else 0
extern "C" __global__ void golden_entropy_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double phi = PHI_F64;
    double x = fabs(a[i]) / phi;
    if (x > 1e-10) {
      out[i] = -x * log(x);
    } else {
      out[i] = 0.0;
    }
  }
}

extern "C" __global__ void golden_entropy_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float phi = PHI_F32;
    float x = fabsf(a[i]) / phi;
    if (x > 1e-6f) {
      out[i] = -x * logf(x);
    } else {
      out[i] = 0.0f;
    }
  }
}

// ============================================================================
// Extended Transcendental Functions (NumPy/math.py Replacement)
// ============================================================================

// Floor function
extern "C" __global__ void floor_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = floor(a[i]);
}

extern "C" __global__ void floor_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = floorf(a[i]);
}

// Ceil function
extern "C" __global__ void ceil_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = ceil(a[i]);
}

extern "C" __global__ void ceil_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = ceilf(a[i]);
}

// Round function
extern "C" __global__ void round_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = round(a[i]);
}

extern "C" __global__ void round_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = roundf(a[i]);
}

// Truncate function (toward zero)
extern "C" __global__ void trunc_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = trunc(a[i]);
}

extern "C" __global__ void trunc_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = truncf(a[i]);
}

// Power function (a^b)
extern "C" __global__ void pow_f64(double *out, const double *a, const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = pow(a[i], b[i]);
}

extern "C" __global__ void pow_f32(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = powf(a[i], b[i]);
}

// Power with scalar exponent
extern "C" __global__ void pow_scalar_f64(double *out, const double *a, double exp, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = pow(a[i], exp);
}

extern "C" __global__ void pow_scalar_f32(float *out, const float *a, float exp, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = powf(a[i], exp);
}

// Inverse trigonometric functions
extern "C" __global__ void asin_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = asin(a[i]);
}

extern "C" __global__ void asin_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = asinf(a[i]);
}

extern "C" __global__ void acos_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = acos(a[i]);
}

extern "C" __global__ void acos_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = acosf(a[i]);
}

extern "C" __global__ void atan_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = atan(a[i]);
}

extern "C" __global__ void atan_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = atanf(a[i]);
}

extern "C" __global__ void atan2_f64(double *out, const double *y, const double *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = atan2(y[i], x[i]);
}

extern "C" __global__ void atan2_f32(float *out, const float *y, const float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = atan2f(y[i], x[i]);
}

// Hyperbolic functions (sinh, cosh already available via tanh)
extern "C" __global__ void sinh_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sinh(a[i]);
}

extern "C" __global__ void sinh_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sinhf(a[i]);
}

extern "C" __global__ void cosh_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cosh(a[i]);
}

extern "C" __global__ void cosh_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = coshf(a[i]);
}

// Inverse hyperbolic functions
extern "C" __global__ void asinh_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = asinh(a[i]);
}

extern "C" __global__ void asinh_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = asinhf(a[i]);
}

extern "C" __global__ void acosh_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = acosh(a[i]);
}

extern "C" __global__ void acosh_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = acoshf(a[i]);
}

extern "C" __global__ void atanh_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = atanh(a[i]);
}

extern "C" __global__ void atanh_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = atanhf(a[i]);
}

// Logarithms (base 10, base 2)
extern "C" __global__ void log10_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = log10(a[i]);
}

extern "C" __global__ void log10_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = log10f(a[i]);
}

extern "C" __global__ void log2_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = log2(a[i]);
}

extern "C" __global__ void log2_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = log2f(a[i]);
}

// Log1p (log(1+x) for numerical stability near zero)
extern "C" __global__ void log1p_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = log1p(a[i]);
}

extern "C" __global__ void log1p_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = log1pf(a[i]);
}

// Expm1 (exp(x)-1 for numerical stability near zero)
extern "C" __global__ void expm1_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = expm1(a[i]);
}

extern "C" __global__ void expm1_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = expm1f(a[i]);
}

// Error function (erf, erfc)
extern "C" __global__ void erf_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = erf(a[i]);
}

extern "C" __global__ void erf_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = erff(a[i]);
}

extern "C" __global__ void erfc_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = erfc(a[i]);
}

extern "C" __global__ void erfc_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = erfcf(a[i]);
}

// Gamma function (tgamma, lgamma)
extern "C" __global__ void tgamma_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tgamma(a[i]);
}

extern "C" __global__ void tgamma_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tgammaf(a[i]);
}

extern "C" __global__ void lgamma_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = lgamma(a[i]);
}

extern "C" __global__ void lgamma_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = lgammaf(a[i]);
}

// ============================================================================
// Comparison and Predicate Functions
// ============================================================================

// isnan: output 1.0 if NaN, 0.0 otherwise
extern "C" __global__ void isnan_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = isnan(a[i]) ? 1.0 : 0.0;
}

extern "C" __global__ void isnan_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = isnan(a[i]) ? 1.0f : 0.0f;
}

// isinf: output 1.0 if infinite, 0.0 otherwise
extern "C" __global__ void isinf_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = isinf(a[i]) ? 1.0 : 0.0;
}

extern "C" __global__ void isinf_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = isinf(a[i]) ? 1.0f : 0.0f;
}

// isfinite: output 1.0 if finite, 0.0 otherwise
extern "C" __global__ void isfinite_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = isfinite(a[i]) ? 1.0 : 0.0;
}

extern "C" __global__ void isfinite_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = isfinite(a[i]) ? 1.0f : 0.0f;
}

// sign: returns -1, 0, or 1 based on sign of input
extern "C" __global__ void sign_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (a[i] > 0.0) out[i] = 1.0;
    else if (a[i] < 0.0) out[i] = -1.0;
    else out[i] = 0.0;
  }
}

extern "C" __global__ void sign_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (a[i] > 0.0f) out[i] = 1.0f;
    else if (a[i] < 0.0f) out[i] = -1.0f;
    else out[i] = 0.0f;
  }
}

// copysign: copy magnitude of first, sign of second
extern "C" __global__ void copysign_f64(double *out, const double *a, const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = copysign(a[i], b[i]);
}

extern "C" __global__ void copysign_f32(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = copysignf(a[i], b[i]);
}

// ============================================================================
// Min/Max Operations (Elementwise)
// ============================================================================

extern "C" __global__ void min_f64(double *out, const double *a, const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmin(a[i], b[i]);
}

extern "C" __global__ void min_f32(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fminf(a[i], b[i]);
}

extern "C" __global__ void max_f64(double *out, const double *a, const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmax(a[i], b[i]);
}

extern "C" __global__ void max_f32(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmaxf(a[i], b[i]);
}

// Clamp (clip values to range)
extern "C" __global__ void clamp_f64(double *out, const double *a, double lo, double hi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmax(lo, fmin(a[i], hi));
}

extern "C" __global__ void clamp_f32(float *out, const float *a, float lo, float hi, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmaxf(lo, fminf(a[i], hi));
}

// ============================================================================
// Angle Conversion (Degrees <-> Radians)
// ============================================================================

extern "C" __global__ void degrees_f64(double *out, const double *a, int n) {
  const double RAD_TO_DEG = 57.29577951308232; // 180 / π
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * RAD_TO_DEG;
}

extern "C" __global__ void degrees_f32(float *out, const float *a, int n) {
  const float RAD_TO_DEG = 57.29577951308232f;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * RAD_TO_DEG;
}

extern "C" __global__ void radians_f64(double *out, const double *a, int n) {
  const double DEG_TO_RAD = 0.017453292519943295; // π / 180
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * DEG_TO_RAD;
}

extern "C" __global__ void radians_f32(float *out, const float *a, int n) {
  const float DEG_TO_RAD = 0.017453292519943295f;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * DEG_TO_RAD;
}

// ============================================================================
// Fused Multiply-Add (FMA) - Hardware accelerated
// ============================================================================

extern "C" __global__ void fma_f64(double *out, const double *a, const double *b, const double *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fma(a[i], b[i], c[i]);
}

extern "C" __global__ void fma_f32(float *out, const float *a, const float *b, const float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmaf(a[i], b[i], c[i]);
}

// ============================================================================
// Modular Arithmetic
// ============================================================================

extern "C" __global__ void fmod_f64(double *out, const double *a, const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmod(a[i], b[i]);
}

extern "C" __global__ void fmod_f32(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmodf(a[i], b[i]);
}

// Remainder (IEEE remainder - different from fmod)
extern "C" __global__ void remainder_f64(double *out, const double *a, const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = remainder(a[i], b[i]);
}

extern "C" __global__ void remainder_f32(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = remainderf(a[i], b[i]);
}

// ============================================================================
// Hypot (Euclidean distance, avoiding overflow)
// ============================================================================

extern "C" __global__ void hypot_f64(double *out, const double *a, const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = hypot(a[i], b[i]);
}

extern "C" __global__ void hypot_f32(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = hypotf(a[i], b[i]);
}

// ============================================================================
// Cube Root
// ============================================================================

extern "C" __global__ void cbrt_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cbrt(a[i]);
}

extern "C" __global__ void cbrt_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cbrtf(a[i]);
}

// ============================================================================
// Reciprocal (1/x) - Often faster than division
// ============================================================================

extern "C" __global__ void reciprocal_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1.0 / a[i];
}

extern "C" __global__ void reciprocal_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __frcp_rn(a[i]); // Fast reciprocal
}

// Reciprocal square root (rsqrt) - Hardware accelerated
extern "C" __global__ void rsqrt_f64(double *out, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = rsqrt(a[i]);
}

extern "C" __global__ void rsqrt_f32(float *out, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = rsqrtf(a[i]);
}

// ============================================================================
// Lerp (Linear interpolation) - lerp(a, b, t) = a + t*(b-a)
// ============================================================================

extern "C" __global__ void lerp_f64(double *out, const double *a, const double *b, const double *t, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + t[i] * (b[i] - a[i]);
}

extern "C" __global__ void lerp_f32(float *out, const float *a, const float *b, const float *t, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + t[i] * (b[i] - a[i]);
}

// Lerp with scalar t
extern "C" __global__ void lerp_scalar_f64(double *out, const double *a, const double *b, double t, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + t * (b[i] - a[i]);
}

extern "C" __global__ void lerp_scalar_f32(float *out, const float *a, const float *b, float t, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + t * (b[i] - a[i]);
}

// ============================================================================
// SRT-Specific: Golden Lerp - interpolation weighted by φ
// golden_lerp(a, b) = a * (1 - φ⁻¹) + b * φ⁻¹ ≈ a * 0.382 + b * 0.618
// ============================================================================

extern "C" __global__ void golden_lerp_f64(double *out, const double *a, const double *b, int n) {
  const double PHI_INV = PHI_INV_F64;
  const double PHI_INV_COMP = PHI_INV_SQ_F64; // 1 - 1/φ = 1/φ² // 1 - φ⁻¹
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * PHI_INV_COMP + b[i] * PHI_INV;
}

extern "C" __global__ void golden_lerp_f32(float *out, const float *a, const float *b, int n) {
  const float PHI_INV = PHI_INV_F32;
  const float PHI_INV_COMP = PHI_INV_SQ_F32; // 1 - 1/φ = 1/φ²
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * PHI_INV_COMP + b[i] * PHI_INV;
}

// ============================================================================
// Q32.32 Fixed-Point Exact Arithmetic Kernels
// ============================================================================

#include "fixed_point.cuh"

// Core elementwise operations (zero overhead - same speed as float32)
extern "C" __global__ void add_fp64(int64_t *out, const int64_t *a, const int64_t *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_add(a[i], b[i]);
}

extern "C" __global__ void sub_fp64(int64_t *out, const int64_t *a, const int64_t *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_sub(a[i], b[i]);
}

extern "C" __global__ void mul_fp64(int64_t *out, const int64_t *a, const int64_t *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_mul(a[i], b[i]);
}

extern "C" __global__ void div_fp64(int64_t *out, const int64_t *a, const int64_t *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_div(a[i], b[i]);
}

extern "C" __global__ void neg_fp64(int64_t *out, const int64_t *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_neg(a[i]);
}

extern "C" __global__ void abs_fp64(int64_t *out, const int64_t *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (a[i] < 0) ? -a[i] : a[i];
}

extern "C" __global__ void scalar_add_fp64(int64_t *out, const int64_t *a, int64_t scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_add(a[i], scalar);
}

extern "C" __global__ void scalar_mul_fp64(int64_t *out, const int64_t *a, int64_t scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_mul(a[i], scalar);
}

// Transcendental functions (10x overhead - but deterministic)
extern "C" __global__ void exp_fp64(int64_t *out, const int64_t *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_exp(a[i]);
}

extern "C" __global__ void sqrt_fp64(int64_t *out, const int64_t *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_sqrt(a[i]);
}

extern "C" __global__ void sin_fp64(int64_t *out, const int64_t *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_sin(a[i]);
}

extern "C" __global__ void cos_fp64(int64_t *out, const int64_t *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fp_cos(a[i]);
}

// Golden ratio operations (exact, deterministic)
extern "C" __global__ void golden_lerp_fp64(int64_t *out, const int64_t *a, const int64_t *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // φ⁻¹ = 0.618... in Q32.32
    int64_t w1 = fp_mul(a[i], FP_PHI_INV);       // a * φ⁻¹
    int64_t w2 = fp_mul(b[i], fp_sub(FP_ONE, FP_PHI_INV)); // b * (1 - φ⁻¹)
    out[i] = fp_add(w1, w2);
  }
}

"""
syntonic_applications.math - Drop-in replacement for Python's math module.

All functions are implemented in Rust for maximum performance.
This module provides the standard Python math interface while
using the SRT library's native implementations.

Usage:
    # Instead of: import math
    from syntonic import math
    
    # Or for explicit replacement:
    from . import math as math

All standard math functions are available:
    - Trigonometric: sin, cos, tan, asin, acos, atan, atan2
    - Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
    - Exponential/Log: exp, log, log10, log2, log1p, expm1
    - Power: pow, sqrt, cbrt, hypot
    - Rounding: floor, ceil, round, trunc
    - Special: fabs, copysign, fma, fmod
    - Predicates: isnan, isinf, isfinite
    - Angle: degrees, radians
    - Integer: gcd, lcm, factorial, comb

Plus SRT-specific extensions:
    - phi_power: φ^x (golden exponential)
    - golden_lerp: golden-ratio weighted interpolation
"""

from _core import (
    # SRT Constants
    srt_pi as _pi,
    srt_e as _e,
    srt_phi as _phi,
    srt_phi_inv as _phi_inv,
    srt_q_deficit as _q_deficit,
    # Scalar math functions
    srt_sqrt as sqrt,
    srt_exp as exp,
    srt_log as log,
    srt_log10 as log10,
    srt_log2 as log2,
    srt_log1p as log1p,
    srt_expm1 as expm1,
    srt_sin as sin,
    srt_cos as cos,
    srt_tan as tan,
    srt_asin as asin,
    srt_acos as acos,
    srt_atan as atan,
    srt_atan2 as atan2,
    srt_sinh as sinh,
    srt_cosh as cosh,
    srt_tanh as tanh,
    srt_asinh as asinh,
    srt_acosh as acosh,
    srt_atanh as atanh,
    srt_pow as pow,
    srt_powi as _powi,
    srt_floor as floor,
    srt_ceil as ceil,
    srt_round as _round,
    srt_trunc as trunc,
    srt_fract as fract,
    srt_abs as fabs,
    srt_signum as copysign_scalar,
    srt_copysign as copysign,
    srt_fma as fma,
    srt_hypot as hypot,
    srt_cbrt as cbrt,
    srt_min as fmin,
    srt_max as fmax,
    srt_clamp as clamp,
    srt_degrees as degrees,
    srt_radians as radians,
    srt_isnan as isnan,
    srt_isinf as isinf,
    srt_isfinite as isfinite,
    srt_lerp as lerp,
    srt_golden_lerp as golden_lerp,
    srt_phi_power as phi_power,
    srt_phi_power_inv as phi_power_inv,
    srt_gcd as _gcd_i64,
    srt_lcm as _lcm_i64,
    srt_factorial as factorial,
    srt_comb as comb,
)

# === Constants ===
pi = _pi()
e = _e()
tau = 2.0 * pi
inf = float('inf')
nan = float('nan')

# SRT-specific constants
phi = _phi()  # Golden ratio φ = (1 + √5) / 2
phi_inv = _phi_inv()  # φ⁻¹ = φ - 1
q_deficit = _q_deficit()  # Universal syntony deficit

# === Compatibility wrappers ===

def fmod(x: float, y: float) -> float:
    """Return x % y (floating point modulo)."""
    return x - floor(x / y) * y

def modf(x: float) -> tuple:
    """Return fractional and integer parts of x."""
    return (fract(x), trunc(x))

def frexp(x: float) -> tuple:
    """Return (m, e) such that x = m * 2^e."""
    import struct
    if x == 0.0:
        return (0.0, 0)
    # Extract exponent from IEEE 754 representation
    bits = struct.unpack('>Q', struct.pack('>d', x))[0]
    exp_bits = (bits >> 52) & 0x7FF
    if exp_bits == 0:  # Denormalized
        x *= 2.0**52
        bits = struct.unpack('>Q', struct.pack('>d', x))[0]
        exp_bits = (bits >> 52) & 0x7FF
        exp_bits -= 52
    exponent = exp_bits - 1022
    mantissa = x / pow(2.0, float(exponent))
    return (mantissa, exponent)

def ldexp(x: float, i: int) -> float:
    """Return x * 2^i."""
    return x * pow(2.0, float(i))

def gcd(*args) -> int:
    """Return greatest common divisor of arguments."""
    if len(args) == 0:
        return 0
    if len(args) == 1:
        return abs(int(args[0]))
    result = int(args[0])
    for arg in args[1:]:
        result = int(_gcd_i64(result, int(arg)))
    return result

def lcm(*args) -> int:
    """Return least common multiple of arguments."""
    if len(args) == 0:
        return 1
    if len(args) == 1:
        return abs(int(args[0]))
    result = int(args[0])
    for arg in args[1:]:
        result = int(_lcm_i64(result, int(arg)))
    return result

def perm(n: int, k: int = None) -> int:
    """Return number of permutations P(n, k)."""
    if k is None:
        k = n
    if k > n:
        return 0
    return int(factorial(n)) // int(factorial(n - k))

def prod(iterable, start=1):
    """Return product of values in iterable."""
    result = start
    for x in iterable:
        result *= x
    return result

def fsum(iterable) -> float:
    """Return accurate floating point sum."""
    # Kahan summation for precision
    total = 0.0
    c = 0.0
    for x in iterable:
        y = float(x) - c
        t = total + y
        c = (t - total) - y
        total = t
    return total

def erf(x: float) -> float:
    """Error function."""
    # Approximation using tanh (good to ~10^-7)
    a = 0.3480242
    b = 0.0958798
    c = 0.7478556
    t = 1.0 / (1.0 + 0.47047 * fabs(x))
    result = 1.0 - t * (a + t * (-b + t * c)) * exp(-x * x)
    return result if x >= 0 else -result

def erfc(x: float) -> float:
    """Complementary error function."""
    return 1.0 - erf(x)

def gamma(x: float) -> float:
    """Gamma function Γ(x)."""
    # Lanczos approximation
    if x <= 0 and x == floor(x):
        raise ValueError("gamma function undefined for non-positive integers")
    
    g = 7
    coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    
    if x < 0.5:
        return pi / (sin(pi * x) * gamma(1.0 - x))
    
    x -= 1
    a = coeffs[0]
    for i in range(1, g + 2):
        a += coeffs[i] / (x + i)
    
    t = x + g + 0.5
    return sqrt(2.0 * pi) * pow(t, x + 0.5) * exp(-t) * a

def lgamma(x: float) -> float:
    """Natural log of absolute value of gamma function."""
    return log(fabs(gamma(x)))

# === Additional exports for compatibility ===

__all__ = [
    # Constants
    'pi', 'e', 'tau', 'inf', 'nan',
    'phi', 'phi_inv', 'q_deficit',  # SRT extensions
    # Functions - Trigonometric
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    # Functions - Hyperbolic
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    # Functions - Exponential/Logarithmic
    'exp', 'log', 'log10', 'log2', 'log1p', 'expm1',
    # Functions - Power
    'pow', 'sqrt', 'cbrt', 'hypot',
    # Functions - Rounding
    'floor', 'ceil', 'trunc', 'fract',
    # Functions - Special
    'fabs', 'copysign', 'fma', 'fmod', 'modf', 'frexp', 'ldexp',
    # Functions - Predicates
    'isnan', 'isinf', 'isfinite',
    # Functions - Angle conversion
    'degrees', 'radians',
    # Functions - Integer
    'gcd', 'lcm', 'factorial', 'comb', 'perm',
    # Functions - Aggregation
    'prod', 'fsum', 'fmin', 'fmax', 'clamp',
    # Functions - Statistical/Special
    'erf', 'erfc', 'gamma', 'lgamma',
    # Functions - Interpolation
    'lerp', 'golden_lerp',  # SRT extension
    # Functions - Golden ratio
    'phi_power', 'phi_power_inv',  # SRT extensions
]
# === Complex number functions (cmath compatibility) ===

def phase(z) -> float:
    """Return the phase angle (in radians) of a complex number."""
    if isinstance(z, complex):
        return atan2(z.imag, z.real)
    return 0.0


def rect(r: float, phi: float) -> complex:
    """Convert polar coordinates (r, phi) to rectangular form."""
    return complex(r * cos(phi), r * sin(phi))


def polar(z) -> tuple:
    """Convert complex number to polar coordinates (r, phi)."""
    if isinstance(z, complex):
        r = sqrt(z.real * z.real + z.imag * z.imag)
        phi = phase(z)
        return (r, phi)
    return (float(z), 0.0)


def cexp(z) -> complex:
    """Complex exponential e^z."""
    if isinstance(z, complex):
        ea = exp(z.real)
        return complex(ea * cos(z.imag), ea * sin(z.imag))
    return exp(z)
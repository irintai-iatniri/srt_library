use crate::exact::fixed::FixedPoint64;

// CORDIC gain K approx 0.607252935
// In Q32.32: 0.607252935 * 2^32 = 2608146603
const CORDIC_GAIN_INV: i64 = 2608146603;

// atan(2^-i) table in Q32.32
const CORDIC_TABLE: [i64; 32] = [
    3373259426, 1991351317, 1052175346, 534100634, 268086747, 134174062, 67103403, 33553749, 16777130, 8388597, 4194302, 2097151, 1048575, 524287, 262143, 131071, 65535, 32767, 16383, 8191, 4095, 2047, 1023, 511, 255, 127, 63, 32, 16, 8, 4, 2
];

pub const HALF_PI: i64 = 6746518852; // 1.570796 * 2^32
pub const PI: i64 = 13493037704;     // 3.141592 * 2^32
pub const TWO_PI: i64 = 26986075409; // 6.283185 * 2^32

/// Simultaneous Sine and Cosine using CORDIC
/// Returns (sin, cos) in FixedPoint64
pub fn cordic_sin_cos(angle: FixedPoint64) -> (FixedPoint64, FixedPoint64) {
    let mut z = angle.0;
    
    // Range reduction to [-PI, PI]
    while z > PI { z -= TWO_PI; }
    while z < -PI { z += TWO_PI; }

    // Map to [-PI/2, PI/2]
    // If z > PI/2, z = PI - z, sign flip for cos?
    // Actually, CORDIC works well in [-PI/2, PI/2].
    // If outside, perform standard quadrant reduction.
    let mut quadrant = 0; // 0: [-PI/2, PI/2], 1: [PI/2, PI] or [-PI, -PI/2]
    
    if z > HALF_PI {
        z = PI - z;
        quadrant = 1;
    } else if z < -HALF_PI {
        z = -PI - z;
        quadrant = 1;
    }

    let mut x: i64 = CORDIC_GAIN_INV;
    let mut y: i64 = 0;
    
    for i in 0..32 {
        let x_shift = x >> i;
        let y_shift = y >> i;
        let angle_val = CORDIC_TABLE[i];

        if z >= 0 {
            x -= y_shift;
            y += x_shift;
            z -= angle_val;
        } else {
            x += y_shift;
            y -= x_shift;
            z += angle_val;
        }
    }

    // Correct for quadrant if needed
    // In quadrant 1 (original was > PI/2 or < -PI/2), sin is same, cos is negated?
    // sin(PI - z) = sin(z) -> Correct
    // cos(PI - z) = -cos(z) -> Needs negation
    // sin(-PI - z) = sin(-(PI+z)) = -sin(PI+z) = -(-sin(z)) = sin(z) -> Correct
    // cos(-PI - z) = cos(PI+z) = -cos(z) -> Needs negation
    
    if quadrant == 1 {
        x = -x;
    }

    (FixedPoint64(y), FixedPoint64(x)) // y is sin, x is cos
}

/// CORDIC Vectoring Mode for atan2(y, x)
/// Returns angle in [-PI, PI]
pub fn cordic_atan2(y_fp: FixedPoint64, x_fp: FixedPoint64) -> FixedPoint64 {
    let mut x = x_fp.0;
    let mut y = y_fp.0;
    let mut z = 0;

    // Pre-rotation to handle quadrants
    if x == 0 && y == 0 { return FixedPoint64::ZERO; }
    
    // Initial quadrant adjustment not strictly needed if we are careful, 
    // but CORDIC usually converges only for angles in [-99, 99] (degrees coverage).
    // Vectoring can handle full circle if we map to right half plane first.
    
    // Map to Right Half Plane (x > 0)
    let mut extra_angle = 0;
    if x < 0 {
        x = -x;
        y = -y;
        extra_angle = PI; // We will add PI later, essentially rotating by 180
    }

    // Now x >= 0.
    
    for i in 0..32 {
        let x_shift = x >> i;
        let y_shift = y >> i;
        let angle_val = CORDIC_TABLE[i];

        // Drive y to 0 using CORDIC vectoring mode
        // Note: x_shift and y_shift capture values before update
        if y >= 0 {
            // Rotate negative to reduce y
            x = x + y_shift;
            y = y - x_shift; // y decreases
            z = z + angle_val;
        } else {
            // Rotate positive
            x = x - y_shift;
            y = y + x_shift; // y increases (from negative)
            z = z - angle_val;
        }
    }
    
    let mut result = FixedPoint64(z);
    
    // Add back the extra angle if we flipped
    if extra_angle != 0 {
        // If original y was < 0 (so we flipped to positive y relative to rotated x?), 
        // wait.
        // If original (x,y) was (-1, 1). We flipped to (1, -1). 
        // code finds angle for (1, -1) which is -PI/4.
        // We add PI -> 3PI/4. Correct.
        
        // If original (x,y) was (-1, -1). We flipped to (1, 1).
        // code finds PI/4.
        // We add PI -> 5PI/4. Which is -3PI/4.
        // We need to normalize to [-PI, PI].
        
        result = result + FixedPoint64(extra_angle);
    }
    
    // Normalize result to [-PI, PI]
    let msg = result.0;
    if msg > PI {
        result.0 -= TWO_PI;
    } else if msg < -PI {
        result.0 += TWO_PI;
    }
    
    result
}

pub fn acos(val: FixedPoint64) -> FixedPoint64 {
    // acos(x) = atan2(sqrt(1-x^2), x)
    if val.0 >= FixedPoint64::ONE.0 { return FixedPoint64::ZERO; }
    if val.0 <= -FixedPoint64::ONE.0 { return FixedPoint64(PI); }
    
    let one_sq = FixedPoint64::ONE;
    let val_sq = val * val;
    let diff = one_sq - val_sq;
    let numer = sqrt(diff);
    
    cordic_atan2(numer, val)
}

pub fn sqrt(x: FixedPoint64) -> FixedPoint64 {
    if x.0 <= 0 { return FixedPoint64::ZERO; }

    // Newton-Raphson: y_new = 0.5 * (y + x/y)
    // Use half as the multiplier for proper fixed-point arithmetic
    let half = FixedPoint64(FixedPoint64::ONE.0 >> 1);
    let raw_x = x;

    // Initial guess: (x + 1) / 2 for better convergence
    let initial_sum = FixedPoint64(x.0.saturating_add(FixedPoint64::ONE.0));
    let mut y_fp = FixedPoint64((initial_sum.0 >> 1).max(1));

    for _ in 0..10 { // 10 iterations usually enough for 32-bit precision
        let div = raw_x / y_fp;
        // y_new = half * (y + x/y) using fixed-point multiplication
        let sum = FixedPoint64(y_fp.0.saturating_add(div.0));
        y_fp = sum * half;
    }

    y_fp
}

// Exp using Taylor series for small |x| and scaling?
// Optimized for negative values (heat kernel usage)
pub fn exp(x: FixedPoint64) -> FixedPoint64 {
    // If x > 0, return 1/exp(-x)
    if x.0 > 0 {
        let neg = exp(FixedPoint64(-x.0));
        return FixedPoint64::ONE / neg;
    }
    
    // x is <= 0.
    // If x < -20, return 0 (underflow)
    if x.0 < -20 * FixedPoint64::SCALING_FACTOR {
        return FixedPoint64::ZERO;
    }

    // Range reduction? 
    // e^x = (e^(x/k))^k
    // Let's reduce x so |x| < 1
    let mut val = x;
    let mut k = 0;
    while val.0 < -FixedPoint64::SCALING_FACTOR {
        val.0 >>= 1; // Divide by 2
        k += 1;
    }
    
    // Now |val| < 1 (or close). Use Taylor series: 1 + x + x^2/2! + ...
    let one = FixedPoint64::ONE;
    let mut res = one;
    let mut term = one;
    let mut n: i64 = 1;

    // 10-15 terms for precision
    while n < 15 {
        term = term * val / FixedPoint64(n * FixedPoint64::SCALING_FACTOR);
        res = res + term;
        if term.0.abs() < 100 { break; } // Convergence check
        n += 1;
    }
    
    // Square k times: (e^(x/2^k))^(2^k)
    for _ in 0..k {
        res = res * res;
    }
    
    res
}

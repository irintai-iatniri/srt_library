//! Rational Rotator - Zero-Drift Wave Generation
//!
//! Implements exact wave generation via Pythagorean rotation matrices.
//! Eliminates accumulating phase error from transcendental approximations.
//!
//! # Theory
//!
//! For Pythagorean triple (a, b, h) where a² + b² = h²:
//! - cos(θ) = a/h (exact rational)
//! - sin(θ) = b/h (exact rational)
//!
//! Rotation matrix:
//! ```text
//! [x']   [a/h  -b/h] [x]
//! [y'] = [b/h   a/h] [y]
//! ```
//!
//! Starting from seed (x₀, y₀), repeatedly apply rotation to generate
//! wave samples y₀, y₁, y₂, ... with zero drift.

use super::rational::Rational;

/// Zero-drift wave generator using Pythagorean rotation
#[derive(Clone, Debug)]
pub struct RationalRotator {
    /// Current x coordinate
    x: Rational,
    /// Current y coordinate (sine value)
    y: Rational,
    /// Pythagorean triple: a component
    a: i128,
    /// Pythagorean triple: b component
    b: i128,
    /// Pythagorean triple: hypotenuse
    h: i128,
}

impl RationalRotator {
    /// Create new rotator with seed point and Pythagorean triple
    ///
    /// # Arguments
    /// * `seed_x` - Initial x coordinate (cosine component)
    /// * `seed_y` - Initial y coordinate (sine component)
    /// * `a` - First leg of Pythagorean triple
    /// * `b` - Second leg of Pythagorean triple
    /// * `h` - Hypotenuse of Pythagorean triple (a² + b² = h²)
    ///
    /// # Panics
    /// Panics if h = 0 (division by zero in rotation matrix)
    pub fn new(seed_x: Rational, seed_y: Rational, a: i128, b: i128, h: i128) -> Self {
        if h == 0 {
            panic!("RationalRotator: hypotenuse cannot be zero");
        }

        RationalRotator {
            x: seed_x,
            y: seed_y,
            a,
            b,
            h,
        }
    }

    /// Apply one rotation step and return current y-coordinate
    ///
    /// # Returns
    /// Current y value (sine component) before advancing to next position
    pub fn next(&mut self) -> Rational {
        let current_y = self.y;

        // Apply rotation matrix:
        // x' = (a*x - b*y) / h
        // y' = (b*x + a*y) / h
        let new_x = Rational::new(self.a * self.x.numerator() - self.b * self.y.numerator(),
                                  self.h * self.x.denominator());
        let new_y = Rational::new(self.b * self.x.numerator() + self.a * self.y.numerator(),
                                  self.h * self.y.denominator());

        self.x = new_x;
        self.y = new_y;

        current_y
    }

    /// Generate n wave samples in batch
    ///
    /// # Arguments
    /// * `n` - Number of samples to generate
    ///
    /// # Returns
    /// Vec of n rational sine values
    pub fn generate_stream(&mut self, n: usize) -> Vec<Rational> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            result.push(self.next());
        }
        result
    }

    /// Get current x coordinate (cosine component)
    pub fn x(&self) -> Rational {
        self.x
    }

    /// Get current y coordinate (sine component)
    pub fn y(&self) -> Rational {
        self.y
    }

    /// Reset to seed position
    ///
    /// # Arguments
    /// * `seed_x` - New x coordinate
    /// * `seed_y` - New y coordinate
    pub fn reset(&mut self, seed_x: Rational, seed_y: Rational) {
        self.x = seed_x;
        self.y = seed_y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotator_basic() {
        // Pythagorean triple (3, 4, 5): 3² + 4² = 25 = 5²
        // This represents rotation by arctan(4/3) ≈ 53.13°
        let mut rotator = RationalRotator::new(
            Rational::from_int(5),  // x₀ = 5 (start at unit circle scaled by 5)
            Rational::from_int(0),  // y₀ = 0
            3, 4, 5
        );

        // First rotation: y should be 4
        let y1 = rotator.next();
        assert_eq!(y1, Rational::from_int(4));

        // After full cycle, should return close to origin
        // (not exact due to discrete rotation angle)
    }

    #[test]
    fn test_rotator_generates_bounded_values() {
        // Use (5, 12, 13) triple
        let mut rotator = RationalRotator::new(
            Rational::from_int(13),
            Rational::from_int(0),
            5, 12, 13
        );

        let samples = rotator.generate_stream(100);

        // All samples should remain bounded
        for sample in samples {
            let magnitude = sample.abs();
            assert!(magnitude <= Rational::from_int(13));
        }
    }

    #[test]
    fn test_rotator_preserves_magnitude() {
        // (3, 4, 5) triple
        let mut rotator = RationalRotator::new(
            Rational::from_int(5),
            Rational::from_int(0),
            3, 4, 5
        );

        // After rotation, x² + y² should remain constant
        let x0 = rotator.x();
        let y0 = rotator.y();
        let mag0_sq = x0 * x0 + y0 * y0;

        rotator.next();
        let x1 = rotator.x();
        let y1 = rotator.y();
        let mag1_sq = x1 * x1 + y1 * y1;

        // Magnitude should be preserved (within rational precision)
        assert_eq!(mag0_sq, mag1_sq);
    }
}

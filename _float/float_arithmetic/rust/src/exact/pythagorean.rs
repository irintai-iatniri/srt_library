//! Pythagorean Triple Generation
//!
//! Generates Pythagorean triples (a, b, h) where a² + b² = h².
//! These provide exact rational rotation angles for TRFT.
//!
//! # Euclid's Formula
//!
//! For coprime integers m > n > 0:
//! - a = m² - n²
//! - b = 2mn
//! - h = m² + n²
//!
//! This generates all primitive Pythagorean triples.
//!
//! # Resonance Ladder
//!
//! For TRFT, we generate a "resonance ladder" of ~400 triples
//! covering wide range of frequencies. Target frequencies:
//! f ≈ n/400 for n ∈ [1, 200] (covers full spectrum)

use num_integer::Integer;

/// Pythagorean triple (a, b, h) where a² + b² = h²
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PythagoreanTriple {
    pub a: i128,
    pub b: i128,
    pub h: i128,
}

impl PythagoreanTriple {
    /// Create new triple, verifying Pythagorean property
    ///
    /// # Panics
    /// Panics if a² + b² ≠ h²
    pub fn new(a: i128, b: i128, h: i128) -> Self {
        debug_assert_eq!(a * a + b * b, h * h, "Invalid Pythagorean triple");
        PythagoreanTriple { a, b, h }
    }

    /// Generate triple from Euclid's formula
    ///
    /// # Arguments
    /// * `m` - First parameter (must be > n)
    /// * `n` - Second parameter (must be > 0)
    ///
    /// # Returns
    /// Pythagorean triple (m² - n², 2mn, m² + n²)
    pub fn from_euclid(m: i128, n: i128) -> Self {
        let a = m * m - n * n;
        let b = 2 * m * n;
        let h = m * m + n * n;
        PythagoreanTriple { a, b, h }
    }

    /// Check if this is a primitive triple (gcd(a, b, h) = 1)
    pub fn is_primitive(&self) -> bool {
        let g1 = self.a.gcd(&self.b);
        let g2 = g1.gcd(&self.h);
        g2 == 1
    }

    /// Get rotation angle: arctan(b/a) in radians (approximate)
    pub fn angle_radians(&self) -> f64 {
        (self.b as f64).atan2(self.a as f64)
    }

    /// Get frequency normalized to [0, 0.5] for sampling
    /// Returns b/h as approximate normalized frequency
    pub fn normalized_frequency(&self) -> f64 {
        (self.b as f64) / (self.h as f64)
    }

    /// Scale triple by integer factor
    ///
    /// # Arguments
    /// * `k` - Scale factor
    ///
    /// # Returns
    /// New triple (ka, kb, kh) which is also Pythagorean
    pub fn scale(&self, k: i128) -> Self {
        PythagoreanTriple {
            a: self.a * k,
            b: self.b * k,
            h: self.h * k,
        }
    }
}

/// Generate resonance ladder of Pythagorean triples
///
/// Creates ~400 triples covering frequency range [0, 0.5].
/// Uses Euclid's formula with parameters chosen to achieve
/// good frequency distribution.
///
/// # Arguments
/// * `max_h` - Maximum hypotenuse value (controls precision)
///
/// # Returns
/// Vec of PythagoreanTriple sorted by frequency
pub fn generate_resonance_ladder(max_h: i128) -> Vec<PythagoreanTriple> {
    let mut triples = Vec::new();

    // Euclid's formula: for m > n > 0 with gcd(m,n)=1 and m-n odd
    // Generate a = m²-n², b = 2mn, h = m²+n²
    let m_max = ((max_h as f64).sqrt() as i128).max(20);

    for m in 2..=m_max {
        for n in 1..m {
            // Check coprimality and parity for primitive triples
            if m.gcd(&n) == 1 && (m - n) % 2 == 1 {
                let triple = PythagoreanTriple::from_euclid(m, n);

                if triple.h <= max_h {
                    triples.push(triple);
                }
            }
        }
    }

    // Sort by normalized frequency (b/h)
    triples.sort_by(|t1, t2| {
        let f1 = (t1.b as f64) / (t1.h as f64);
        let f2 = (t2.b as f64) / (t2.h as f64);
        f1.partial_cmp(&f2).unwrap_or(std::cmp::Ordering::Equal)
    });

    triples
}

/// Generate resonance ladder with target count
///
/// Adaptively adjusts max_h to achieve approximately target_count triples.
///
/// # Arguments
/// * `target_count` - Desired number of triples (~400 recommended)
///
/// # Returns
/// Vec of PythagoreanTriple
pub fn generate_resonance_ladder_adaptive(target_count: usize) -> Vec<PythagoreanTriple> {
    // Heuristic: max_h ≈ sqrt(target_count) * 10
    let initial_max_h = ((target_count as f64).sqrt() * 10.0) as i128;

    let mut max_h = initial_max_h;
    let mut triples = generate_resonance_ladder(max_h);

    // Adjust if needed
    const MAX_ITERATIONS: usize = 5;
    for _ in 0..MAX_ITERATIONS {
        let count = triples.len();
        if count >= target_count * 9 / 10 && count <= target_count * 11 / 10 {
            break; // Within 10% of target
        }

        if count < target_count {
            max_h = (max_h as f64 * 1.2) as i128;
        } else {
            max_h = (max_h as f64 * 0.8) as i128;
        }

        triples = generate_resonance_ladder(max_h);
    }

    triples
}

/// Find closest triple to target frequency
///
/// # Arguments
/// * `ladder` - Resonance ladder (should be sorted by frequency)
/// * `target_freq` - Target normalized frequency [0, 0.5]
///
/// # Returns
/// Index of closest triple
pub fn find_closest_frequency(ladder: &[PythagoreanTriple], target_freq: f64) -> usize {
    if ladder.is_empty() {
        return 0;
    }

    let mut best_idx = 0;
    let mut best_distance = (ladder[0].normalized_frequency() - target_freq).abs();

    for (idx, triple) in ladder.iter().enumerate() {
        let distance = (triple.normalized_frequency() - target_freq).abs();
        if distance < best_distance {
            best_distance = distance;
            best_idx = idx;
        }
    }

    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclid_formula() {
        // (m=2, n=1) → (3, 4, 5)
        let triple = PythagoreanTriple::from_euclid(2, 1);
        assert_eq!(triple.a, 3);
        assert_eq!(triple.b, 4);
        assert_eq!(triple.h, 5);
        assert_eq!(triple.a * triple.a + triple.b * triple.b, triple.h * triple.h);
    }

    #[test]
    fn test_primitive_check() {
        let primitive = PythagoreanTriple::new(3, 4, 5);
        assert!(primitive.is_primitive());

        let non_primitive = PythagoreanTriple::new(6, 8, 10);
        assert!(!non_primitive.is_primitive());
    }

    #[test]
    fn test_resonance_ladder_generation() {
        let ladder = generate_resonance_ladder(500);

        // Should generate multiple triples
        assert!(ladder.len() > 50);

        // All should be valid Pythagorean triples
        for triple in &ladder {
            assert_eq!(
                triple.a * triple.a + triple.b * triple.b,
                triple.h * triple.h
            );
        }

        // Should be sorted by frequency
        for i in 1..ladder.len() {
            let f1 = ladder[i - 1].normalized_frequency();
            let f2 = ladder[i].normalized_frequency();
            assert!(f2 >= f1, "Ladder not sorted by frequency");
        }
    }

    #[test]
    fn test_adaptive_ladder() {
        let ladder = generate_resonance_ladder_adaptive(400);

        // Should be close to target count
        assert!(ladder.len() >= 360 && ladder.len() <= 440);
    }

    #[test]
    fn test_find_closest_frequency() {
        let ladder = generate_resonance_ladder(200);

        // Find triple closest to 0.25 frequency
        let idx = find_closest_frequency(&ladder, 0.25);
        let found_freq = ladder[idx].normalized_frequency();

        // Should be reasonably close
        assert!((found_freq - 0.25).abs() < 0.05);
    }

    #[test]
    fn test_scale_triple() {
        let base = PythagoreanTriple::new(3, 4, 5);
        let scaled = base.scale(2);

        assert_eq!(scaled.a, 6);
        assert_eq!(scaled.b, 8);
        assert_eq!(scaled.h, 10);
        assert_eq!(scaled.a * scaled.a + scaled.b * scaled.b, scaled.h * scaled.h);
    }
}

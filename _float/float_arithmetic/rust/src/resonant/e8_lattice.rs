use crate::resonant::PHI;

/// E8 Lattice Generator and Golden Projector
///
/// Implements:
/// 1. E8 Root generation (240 roots)
/// 2. Golden Projection R^8 -> R^4 (Parallel) + R^4 (Perpendicular)
/// 3. Golden Cone filtering

pub struct E8Lattice {
    pub roots: Vec<[f64; 8]>,
}

impl E8Lattice {
    pub fn new() -> Self {
        let mut roots = Vec::with_capacity(240);

        // Type A: Permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0)
        // Indices pairs: (0..8) choose 2 = 28 pairs.
        // Signs: ++, +-, -+, -- (4 combinations)
        // Total: 28 * 4 = 112
        for i in 0..8 {
            for j in (i + 1)..8 {
                for s1 in [-1.0, 1.0] {
                    for s2 in [-1.0, 1.0] {
                        let mut v = [0.0; 8];
                        v[i] = s1;
                        v[j] = s2;
                        roots.push(v);
                    }
                }
            }
        }

        // Type B: (+-0.5, ..., +-0.5) with even number of minus signs
        // 2^8 = 256 combinations. Half have even signs.
        // Total: 128
        for i in 0..256 {
            let mut v = [0.0; 8];
            let mut neg_count = 0;
            for bit in 0..8 {
                if (i >> bit) & 1 == 1 {
                    v[bit] = -0.5;
                    neg_count += 1;
                } else {
                    v[bit] = 0.5;
                }
            }
            if neg_count % 2 == 0 {
                roots.push(v);
            }
        }

        E8Lattice { roots }
    }

    /// Static helper to get roots as Vec<Vec<f64>>
    pub fn generate_roots() -> Vec<Vec<f64>> {
        let lattice = Self::new();
        lattice.roots.into_iter().map(|arr| arr.to_vec()).collect()
    }

    /// Generate N 8D weight vectors from E8 roots.
    ///
    /// Cycles through the 240 E8 roots with scaling for indices beyond 240.
    /// Useful for initializing neural network weights with E8 lattice structure.
    ///
    /// # Arguments
    /// * `n` - Number of 8D weight vectors to generate
    ///
    /// # Returns
    /// Flat Vec<f64> of length n*8 containing n 8D vectors
    pub fn generate_n_weights(&self, n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n * 8);
        for i in 0..n {
            let root = self.roots[i % 240];
            let scale = 1.0 + (i / 240) as f64;
            for x in root {
                result.push(x * scale);
            }
        }
        result
    }

    /// Static helper to generate N 8D weight vectors without creating an instance.
    pub fn generate_weights(n: usize) -> Vec<f64> {
        let lattice = Self::new();
        lattice.generate_n_weights(n)
    }
}

pub struct GoldenProjector {
    /// Golden ratio φ = (1 + √5) / 2
    phi: f64,
    /// Parallel projection matrix [4][8]
    p_par: [[f64; 8]; 4],
    /// Perpendicular projection matrix [4][8]
    p_perp: [[f64; 8]; 4],
}

impl GoldenProjector {
    pub fn new() -> Self {
        let phi = PHI;
        let norm = 1.0f64 / (1.0f64 + phi * phi).sqrt();

        // Parallel Projection: x_i + phi * x_{i+4} (normalized)
        let mut p_par = [[0.0; 8]; 4];
        let mut p_perp = [[0.0; 8]; 4];

        for i in 0..4 {
            p_par[i][i] = 1.0 * norm;
            p_par[i][i + 4] = phi * norm;

            p_perp[i][i] = phi * norm;
            p_perp[i][i + 4] = -1.0 * norm;
        }

        Self { phi, p_par, p_perp }
    }

    pub fn project_parallel(&self, v: &[f64]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for i in 0..4 {
            let mut sum = 0.0;
            for j in 0..8 {
                sum += self.p_par[i][j] * v[j];
            }
            out[i] = sum;
        }
        out
    }

    pub fn project_perp(&self, v: &[f64]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for i in 0..4 {
            let mut sum = 0.0;
            for j in 0..8 {
                sum += self.p_perp[i][j] * v[j];
            }
            out[i] = sum;
        }
        out
    }

    /// Compute the hyperbolic metric Q = ||v_parallel||² - ||v_perp||²
    ///
    /// This is the key invariant for the Golden Cone:
    /// - Q > 0: Inside the cone (timelike)
    /// - Q = 0: On the cone boundary (lightlike)
    /// - Q < 0: Outside the cone (spacelike)
    pub fn compute_q(&self, v: &[f64]) -> f64 {
        let par = self.project_parallel(v);
        let perp = self.project_perp(v);
        let norm_par: f64 = par.iter().map(|x| x * x).sum();
        let norm_perp: f64 = perp.iter().map(|x| x * x).sum();
        norm_par - norm_perp
    }

    /// Get the golden ratio value
    pub fn phi(&self) -> f64 {
        self.phi
    }
}

/// The 4 null vectors defining the golden cone boundary
/// These satisfy Q(c_a) = 0 and define the 36 Φ⁺(E₆) roots via ⟨c_a, α⟩ > 0
const NULL_VECTORS: [[f64; 8]; 4] = [
    [
        -0.152753, -0.312330, 0.192683, -0.692448, 0.013308, 0.531069, 0.153449, 0.238219,
    ],
    [
        0.270941, 0.201058, 0.532514, -0.128468, 0.404635, -0.475518, -0.294881, 0.330591,
    ],
    [
        0.157560, 0.189639, 0.480036, 0.021016, 0.274831, -0.782436, 0.060319, 0.130225,
    ],
    [
        0.476719, 0.111410, 0.464671, -0.543379, -0.142049, -0.150498, -0.314296, 0.327929,
    ],
];

/// Check if vector is in the golden cone using 4 null vectors
/// A root α is in the cone iff ⟨c_a, α⟩ > 0 for all 4 null vectors
/// This selects exactly 36 roots = Φ⁺(E₆)
pub fn is_in_golden_cone(v: &[f64], tol: f64) -> bool {
    if v.len() != 8 {
        return false;
    }

    for c in &NULL_VECTORS {
        let inner: f64 = (0..8).map(|i| c[i] * v[i]).sum();
        if inner <= tol {
            return false;
        }
    }
    true
}

/// Compute scalar weight from 8D vector using Golden Cone & hyperbolic metric
/// Returns w = exp(-Q / φ) where Q = ||v_parallel||² - ||v_perp||²
///
/// The Q metric measures "how timelike" the vector is:
/// - Q > 0: Timelike (inside cone) → high weight
/// - Q < 0: Spacelike (outside cone) → suppressed
pub fn compute_8d_weight(v: &[f64], projector: &GoldenProjector) -> f64 {
    // First check if vector is in the golden cone using null vectors
    if !is_in_golden_cone(v, 1e-10) {
        // Outside Golden Cone → Suppress heavily
        return 1e-9;
    }

    // Inside Golden Cone - compute weight based on hyperbolic metric Q
    let q = projector.compute_q(v);

    // Weight: exp(-Q / φ)
    // High Q (strongly timelike) → low weight (more "massive")
    // Low Q (near lightlike) → high weight (more "massless")
    (-q / PHI).exp()
}

// =============================================================================
// PyO3 Wrappers for Python Access
// =============================================================================

use pyo3::prelude::*;

/// Generate N 8D weight vectors from E8 roots (Python wrapper)
#[pyfunction]
#[pyo3(name = "e8_generate_weights")]
pub fn py_e8_generate_weights(n: usize) -> Vec<f64> {
    E8Lattice::generate_weights(n)
}

/// Get all 240 E8 roots as list of 8D vectors (Python wrapper)
#[pyfunction]
#[pyo3(name = "e8_generate_roots")]
pub fn py_e8_generate_roots() -> Vec<Vec<f64>> {
    E8Lattice::generate_roots()
}

/// Compute the hyperbolic metric Q for an 8D vector (Python wrapper)
/// Q = ||v_parallel||² - ||v_perp||²
#[pyfunction]
#[pyo3(name = "golden_projector_q")]
pub fn py_golden_projector_q(v: Vec<f64>) -> f64 {
    let projector = GoldenProjector::new();
    projector.compute_q(&v)
}

/// Get the golden ratio φ used in projections (Python wrapper)
#[pyfunction]
#[pyo3(name = "golden_projector_phi")]
pub fn py_golden_projector_phi() -> f64 {
    let projector = GoldenProjector::new();
    projector.phi()
}

/// Project 8D vector to 4D parallel subspace (Python wrapper)
#[pyfunction]
#[pyo3(name = "golden_project_parallel")]
pub fn py_golden_project_parallel(v: Vec<f64>) -> [f64; 4] {
    let projector = GoldenProjector::new();
    projector.project_parallel(&v)
}

/// Project 8D vector to 4D perpendicular subspace (Python wrapper)
#[pyfunction]
#[pyo3(name = "golden_project_perp")]
pub fn py_golden_project_perp(v: Vec<f64>) -> [f64; 4] {
    let projector = GoldenProjector::new();
    projector.project_perp(&v)
}

/// Check if 8D vector is inside the Golden Cone (Python wrapper)
#[pyfunction]
#[pyo3(name = "is_in_golden_cone_py")]
pub fn py_is_in_golden_cone(v: Vec<f64>, tol: f64) -> bool {
    is_in_golden_cone(&v, tol)
}

/// Compute 8D weight for a vector (Python wrapper)
#[pyfunction]
#[pyo3(name = "compute_8d_weight_py")]
pub fn py_compute_8d_weight(v: Vec<f64>) -> f64 {
    let projector = GoldenProjector::new();
    compute_8d_weight(&v, &projector)
}

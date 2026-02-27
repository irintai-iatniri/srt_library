//! Transcendence Module
//!
//! Implements the Fibonacci-Plane mapping and ontological phase transitions
//! from The Grand Synthesis theory.

use pyo3::prelude::*;

/// Plane-Fibonacci mapping structure
#[derive(Debug, Clone)]
pub struct PlaneMapping {
    pub fib_index: u64,
    pub plane_start: u8,
    pub plane_end: u8,
    pub name: &'static str,
    pub description: &'static str,
}

/// The 18 Ontological Planes mapped to Fibonacci Prime indices
pub const PLANE_MAPPINGS: [PlaneMapping; 8] = [
    PlaneMapping {
        fib_index: 3,
        plane_start: 1,
        plane_end: 2,
        name: "Ideological",
        description: "Binary Root - Duality emerges",
    },
    PlaneMapping {
        fib_index: 4,
        plane_start: 3,
        plane_end: 3,
        name: "Mathematics",
        description: "Material Anomaly - Composite→Prime",
    },
    PlaneMapping {
        fib_index: 5,
        plane_start: 4,
        plane_end: 5,
        name: "Physics",
        description: "Golden Code - DNA/φ base",
    },
    PlaneMapping {
        fib_index: 7,
        plane_start: 6,
        plane_end: 10,
        name: "Deterministic",
        description: "Atomic Sleep - Matter solidifies",
    },
    PlaneMapping {
        fib_index: 11,
        plane_start: 11,
        plane_end: 12,
        name: "Chaotic",
        description: "Branching - Complexity emerges",
    },
    PlaneMapping {
        fib_index: 13,
        plane_start: 13,
        plane_end: 16,
        name: "Consciousness",
        description: "Gamma Synchrony - Life alignment",
    },
    PlaneMapping {
        fib_index: 17,
        plane_start: 17,
        plane_end: 17,
        name: "Cosmic",
        description: "Great Filter - Deep Transcendence",
    },
    PlaneMapping {
        fib_index: 23,
        plane_start: 18,
        plane_end: 18,
        name: "Versal",
        description: "Ouroboros - Lattice closes",
    },
];

/// Get plane info for a given Fibonacci index
#[pyfunction]
pub fn get_plane_for_fib_index(fib_index: u64) -> Option<(u8, u8, String, String)> {
    PLANE_MAPPINGS
        .iter()
        .find(|p| p.fib_index == fib_index)
        .map(|p| {
            (
                p.plane_start,
                p.plane_end,
                p.name.to_string(),
                p.description.to_string(),
            )
        })
}

/// Get the current ontological plane based on recursion depth
#[pyfunction]
pub fn compute_current_plane(recursion_depth: u64) -> u8 {
    for mapping in PLANE_MAPPINGS.iter().rev() {
        if recursion_depth >= mapping.fib_index {
            return mapping.plane_start;
        }
    }
    0 // Pre-existence
}

/// Compute transcendence probability based on syntony and plane
#[pyfunction]
pub fn transcendence_probability(syntony: f64, current_plane: u8) -> f64 {
    use crate::tensor::srt_kernels::PHI;

    // Must have syntony > 0.618 (1/φ) to begin transcendence
    if syntony < 1.0 / PHI {
        return 0.0;
    }

    // Probability increases with syntony alignment to φ
    let alignment = (syntony - (1.0 / PHI)) / (1.0 - (1.0 / PHI));
    let plane_factor = 1.0 - (current_plane as f64 / 18.0);

    (alignment * plane_factor).min(1.0).max(0.0)
}

pub fn register_transcendence(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_plane_for_fib_index, m)?)?;
    m.add_function(wrap_pyfunction!(compute_current_plane, m)?)?;
    m.add_function(wrap_pyfunction!(transcendence_probability, m)?)?;
    Ok(())
}

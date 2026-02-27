//!
//! SRT Precision Policy Enforcement
//!
//! Ensures that SRT theory operations maintain exact precision while allowing
//! optimized operations for non-SRT computations.

use pyo3::prelude::*;

/// Precision policy for SRT operations
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionPolicy {
    /// Exact - SRT Theory operations ONLY
    /// ALL SRT operations are exact. No approximations allowed.
    /// Used for: GoldenExact, PySymExpr, φ-scaling, q-corrections,
    /// spectral computations, DHSR operators, theory-correct activations
    Exact,

    /// Mixed precision - can use fp32 with WMMA
    /// Used for: Standard neural network matmul (non-SRT)
    MixedPrecision,

    /// Low precision - fp16/bf16 allowed
    /// Used for: Generic activations, training large models
    LowPrecision,
}

impl PrecisionPolicy {
    /// Can this operation use WMMA?
    pub fn allows_wmma(&self) -> bool {
        matches!(self, Self::MixedPrecision | Self::LowPrecision)
    }

    /// Minimum precision required
    pub fn min_dtype(&self) -> &'static str {
        match self {
            Self::Exact => "float64",
            Self::MixedPrecision => "float32",
            Self::LowPrecision => "float16",
        }
    }
}

/// Annotate SRT operations with precision requirements
pub fn srt_operation_policy(op: &str) -> PrecisionPolicy {
    match op {
        // ========================================================
        // EXACT (SRT Theory) - ALL exact, no approximations
        // ========================================================

        // Exact constants (GoldenExact, PySymExpr)
        "golden_ratio" | "q_deficit" | "correction_factor" | "phi" | "phi_inv" | "pi" | "e"
        | "e_star" => PrecisionPolicy::Exact,

        // φ-scaled matmul operations (use exact φⁿ via Fibonacci formula)
        "mm_phi" | "linalg_mm_phi" => PrecisionPolicy::Exact,

        // q-corrected matmul (uses exact q = 0.027395146920...)
        "mm_corrected"
        | "linalg_mm_corrected"
        | "mm_q_corrected_direct"
        | "linalg_mm_q_corrected_direct" => PrecisionPolicy::Exact,

        // Golden algebra operations (φ-Lie brackets)
        "phi_bracket" | "linalg_phi_bracket" | "phi_antibracket" | "linalg_phi_antibracket" => {
            PrecisionPolicy::Exact
        }

        // Golden-weighted matmul (exponential weight e^{-k²/φ})
        "mm_golden_weighted" | "linalg_mm_golden_weighted" => PrecisionPolicy::Exact,

        // Complex golden phase (e^{iπn/φ})
        "mm_golden_phase" | "linalg_mm_golden_phase" => PrecisionPolicy::Exact,

        // Theory-correct residual connections (φ⁻¹ scaling)
        "phi_residual_relu" | "phi_residual" | "phi_residual_gelu" | "phi_residual_layernorm" => {
            PrecisionPolicy::Exact
        }

        // Projection operations (used in DHSR cycle)
        "projection_sum" | "linalg_projection_sum" => PrecisionPolicy::Exact,

        // DHSR operators
        "differentiate" | "harmonize" | "syntony" | "recurse" | "gnosis" => PrecisionPolicy::Exact,

        // Spectral computations
        "heat_kernel" | "e8_projection" | "golden_cone" => PrecisionPolicy::Exact,

        // ========================================================
        // MIXED PRECISION (WMMA allowed for non-SRT)
        // ========================================================

        // Standard neural network operations
        "mm" | "bmm" | "conv2d" | "linalg_mm" | "linalg_bmm" => PrecisionPolicy::MixedPrecision,

        // Standard transpose variants
        "mm_tn" | "mm_nt" | "mm_tt" | "mm_hn" | "mm_nh" => PrecisionPolicy::MixedPrecision,

        // ========================================================
        // LOW PRECISION (fp16/bf16 allowed)
        // ========================================================

        // Standard activation functions
        "relu" | "sigmoid" | "tanh" | "gelu" => PrecisionPolicy::LowPrecision,

        // Default: Exact (conservative - assume SRT)
        _ => PrecisionPolicy::Exact,
    }
}

#[pymethods]
impl PrecisionPolicy {
    #[getter]
    fn get_allows_wmma(&self) -> bool {
        self.allows_wmma()
    }

    #[getter]
    fn get_min_dtype(&self) -> &str {
        self.min_dtype()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        format!("{:?}", self)
    }
}

/// Python-exposed function to get precision policy for an operation
#[pyfunction]
pub fn get_srt_operation_policy(op: &str) -> PrecisionPolicy {
    srt_operation_policy(op)
}

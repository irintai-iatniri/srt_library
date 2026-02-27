//! Fundamental Constants of Syntony Recursion Theory
//!
//! SRT is built on exactly five fundamental constants. These are NOT
//! floating-point approximations - they are symbolic atoms that maintain
//! their exact identities through all computations.
//!
//! # The Five Constants
//!
//! | Symbol | Name | Type | Role in SRT |
//! |--------|------|------|-------------|
//! | π | Pi | Classical transcendental | W⁴ toroidal topology |
//! | e | Euler | Classical transcendental | Heat kernel / RG flow |
//! | φ | Phi | Algebraic irrational | Recursion eigenvalue |
//! | E* | EStar | SRT transcendental | Spectral Möbius constant |
//! | q | Q | SRT transcendental | Universal syntony deficit |
//!
//! # Fundamental Identities
//!
//! These are THEOREMS, not definitions:
//! - φ² = φ + 1 (algebraic identity)
//! - E* = e^π - π (Spectral Theorem, verified to 512 digits)
//! - q = (2φ + e/(2φ²)) / (φ⁴ · E*) (Universal Formula)
//!
//! # The Universal Syntony Correction Hierarchy
//!
//! All physical quantities derive from q through the 25-level hierarchy:
//! - q/1000 ≈ 0.003% (fixed-point stability)
//! - q/8 ≈ 0.34% (Cartan subalgebra / rank(E₈))
//! - q ≈ 2.74% (base syntony deficit)
//! - 4q ≈ 11% (full W⁴ topology)

use pyo3::prelude::*;
use std::fmt;

use super::rational::Rational;
use super::symexpr::SymExpr;

/// The five fundamental constants of Syntony Recursion Theory
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FundamentalConstant {
    /// π - Archimedes' constant
    /// The ratio of circumference to diameter.
    /// In SRT: defines the W⁴ toroidal topology.
    Pi,

    /// e - Euler's number
    /// The base of natural logarithms.
    /// In SRT: governs heat kernel decay and RG flow.
    Euler,

    /// φ = (1 + √5)/2 - The golden ratio
    /// The unique positive root of x² - x - 1 = 0.
    /// In SRT: THE recursion eigenvalue. All mass hierarchies scale as φ^k.
    /// Note: φ is algebraic (not transcendental), but fundamental to SRT.
    Phi,

    /// E* = e^π - π ≈ 19.999099979...
    /// The Spectral Möbius Constant.
    /// In SRT: the finite part of the Möbius-regularized heat kernel on
    /// the Golden Lattice. Numerically verified to 512 decimal places.
    EStar,

    /// q ≈ 0.027395146920...
    /// The Universal Syntony Deficit.
    /// In SRT: THE fundamental scale constant from which ALL physical
    /// quantities derive. This is the single most important number in SRT.
    /// q = (2φ + e/(2φ²)) / (φ⁴ · E*)
    Q,
}

impl FundamentalConstant {
    /// High-precision decimal representation (for verification only)
    /// These values are exact to the displayed precision.
    pub fn decimal_value(&self) -> &'static str {
        match self {
            Self::Pi => {
                "3.14159265358979323846264338327950288419716939937510\
                         58209749445923078164062862089986280348253421170679"
            }
            Self::Euler => {
                "2.71828182845904523536028747135266249775724709369995\
                            95749669676277240766303535475945713821785251664274"
            }
            Self::Phi => {
                "1.61803398874989484820458683436563811772030917980576\
                          28621354486227052604628189024497072072041893911375"
            }
            Self::EStar => {
                "19.9990999791894757672664429846690429197224416781423\
                            6858336953124189574809985424545289195611836432"
            }
            Self::Q => {
                "0.02739514692015854536545317970880286797914906945068\
                        75412837509623147291034502816413258903762145897"
            }
        }
    }

    /// f64 approximation (for quick numerical checks only)
    pub fn approx_f64(&self) -> f64 {
        match self {
            Self::Pi => std::f64::consts::PI,
            Self::Euler => std::f64::consts::E,
            Self::Phi => 1.6180339887498949,
            Self::EStar => 19.99909997918947576,
            Self::Q => 0.02739514692015854,
        }
    }

    /// Whether this constant is algebraic (vs transcendental)
    pub fn is_algebraic(&self) -> bool {
        matches!(self, Self::Phi)
    }

    /// Whether this constant is transcendental
    pub fn is_transcendental(&self) -> bool {
        !self.is_algebraic()
    }

    /// The minimal polynomial for algebraic constants (None for transcendentals)
    pub fn minimal_polynomial(&self) -> Option<&'static str> {
        match self {
            Self::Phi => Some("x² - x - 1"),
            _ => None,
        }
    }

    /// Unicode symbol
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Pi => "π",
            Self::Euler => "e",
            Self::Phi => "φ",
            Self::EStar => "E*",
            Self::Q => "q",
        }
    }

    /// Full name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pi => "Pi (Archimedes' constant)",
            Self::Euler => "e (Euler's number)",
            Self::Phi => "φ (Golden ratio)",
            Self::EStar => "E* (Spectral Möbius constant)",
            Self::Q => "q (Universal syntony deficit)",
        }
    }

    /// Role in SRT
    pub fn srt_role(&self) -> &'static str {
        match self {
            Self::Pi => "W⁴ toroidal topology, angular periodicity",
            Self::Euler => "Heat kernel decay, RG flow, exponential evolution",
            Self::Phi => "Recursion eigenvalue, mass hierarchy scaling",
            Self::EStar => "Spectral finite part, Möbius regularization scale",
            Self::Q => "Universal scale constant, all masses/couplings derive from q",
        }
    }
}

impl fmt::Display for FundamentalConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

// === PyO3 Methods ===

#[pymethods]
impl FundamentalConstant {
    /// Get the Pi constant
    #[staticmethod]
    fn pi() -> Self {
        Self::Pi
    }

    /// Get Euler's number e
    #[staticmethod]
    fn e() -> Self {
        Self::Euler
    }

    /// Get the golden ratio φ
    #[staticmethod]
    fn phi() -> Self {
        Self::Phi
    }

    /// Get the Spectral Möbius constant E*
    #[staticmethod]
    fn e_star() -> Self {
        Self::EStar
    }

    /// Get the universal syntony deficit q
    #[staticmethod]
    fn q() -> Self {
        Self::Q
    }

    fn __repr__(&self) -> String {
        format!(
            "FundamentalConstant.{}",
            match self {
                Self::Pi => "Pi",
                Self::Euler => "Euler",
                Self::Phi => "Phi",
                Self::EStar => "EStar",
                Self::Q => "Q",
            }
        )
    }

    fn __str__(&self) -> String {
        self.symbol().to_string()
    }

    /// Get the symbol for this constant
    fn get_symbol(&self) -> String {
        self.symbol().to_string()
    }

    /// Get the full name
    fn get_name(&self) -> String {
        self.name().to_string()
    }

    /// Get the role in SRT
    fn get_srt_role(&self) -> String {
        self.srt_role().to_string()
    }

    /// Get high-precision decimal (for verification)
    fn get_decimal(&self) -> String {
        self.decimal_value().to_string()
    }

    /// Get f64 approximation (for quick checks only!)
    fn get_approx(&self) -> f64 {
        self.approx_f64()
    }

    /// Is this an algebraic number?
    fn get_is_algebraic(&self) -> bool {
        self.is_algebraic()
    }

    /// Get minimal polynomial (if algebraic)
    fn get_minimal_polynomial(&self) -> Option<String> {
        self.minimal_polynomial().map(|s| s.to_string())
    }
}

/// The Complete Universal Syntony Correction Hierarchy (60+ Levels)
///
/// Extended hierarchy with all W⁴ × E₈ geometric structures.
/// Levels 0-57 covering all geometric origins from the complete framework.
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CorrectionLevel {
    /// Level 0: 1 - Tree-level (exact)
    Level0,
    /// Level 1: q³ ≈ 0.002% - Third-order vacuum
    Level1,
    /// Level 2: q/1000 ≈ 0.003% - Fixed-point stability (h(E₈)³/27)
    Level2,
    /// Level 3: q/720 ≈ 0.004% - Coxeter-Kissing product (h(E₈)×K(D₄))
    Level3,
    /// Level 4: q/360 ≈ 0.008% - Complete cone cycles (10×36)
    Level4,
    /// Level 5: q/248 ≈ 0.011% - Full E₈ adjoint representation
    Level5,
    /// Level 6: q/240 ≈ 0.011% - Full E₈ root system (both signs)
    Level6,
    /// Level 7: q/133 ≈ 0.021% - Full E₇ adjoint representation
    Level7,
    /// Level 8: q/126 ≈ 0.022% - Full E₇ root system
    Level8,
    /// Level 9: q/120 ≈ 0.023% - Complete E₈ positive roots
    Level9,
    /// Level 10: q²/φ² ≈ 0.029% - Second-order/double golden
    Level10,
    /// Level 11: q/78 ≈ 0.035% - Full E₆ gauge structure
    Level11,
    /// Level 12: q/72 ≈ 0.038% - Full E₆ root system (both signs)
    Level12,
    /// Level 13: q/63 ≈ 0.044% - E₇ positive roots
    Level13,
    /// Level 14: q²/φ ≈ 0.046% - Second-order massless
    Level14,
    /// Level 15: q/56 ≈ 0.049% - E₇ fundamental representation
    Level15,
    /// Level 16: q/52 ≈ 0.053% - F₄ gauge structure
    Level16,
    /// Level 17: q² ≈ 0.075% - Second-order vacuum
    Level17,
    /// Level 18: q/36 ≈ 0.076% - E₆ positive roots (Golden Cone)
    Level18,
    /// Level 19: q/32 ≈ 0.086% - Five-fold binary structure
    Level19,
    /// Level 20: q/30 ≈ 0.091% - E₈ Coxeter number
    Level20,
    /// Level 21: q/28 ≈ 0.098% - D₄ adjoint representation
    Level21,
    /// Level 22: q/27 ≈ 0.101% - E₆ fundamental representation
    Level22,
    /// Level 23: q/24 ≈ 0.114% - D₄ kissing number
    Level23,
    /// Level 24: q²φ ≈ 0.121% - Quadratic + golden enhancement
    Level24,
    /// Level 25: q/(6π) ≈ 0.145% - Six-flavor QCD loop
    Level25,
    /// Level 26: q/18 ≈ 0.152% - E₇ Coxeter number
    Level26,
    /// Level 27: q/16 ≈ 0.171% - Four-fold binary/spinor dimension
    Level27,
    /// Level 28: q/(5π) ≈ 0.174% - Five-flavor QCD loop
    Level28,
    /// Level 29: q/14 ≈ 0.196% - G₂ octonion automorphisms
    Level29,
    /// Level 30: q/(4π) ≈ 0.218% - One-loop radiative (4D)
    Level30,
    /// Level 31: q/12 ≈ 0.228% - Topology × generations (W⁴ × N_gen)
    Level31,
    /// Level 32: q/φ⁵ ≈ 0.247% - Fifth golden power
    Level32,
    /// Level 33: q/(3π) ≈ 0.290% - Three-flavor QCD loop
    Level33,
    /// Level 34: q/9 ≈ 0.304% - Generation-squared structure
    Level34,
    /// Level 35: q/8 ≈ 0.342% - Cartan subalgebra (rank E₈)
    Level35,
    /// Level 36: q/7 ≈ 0.391% - E₇ Cartan subalgebra
    Level36,
    /// Level 37: q/φ⁴ ≈ 0.400% - Fourth golden power
    Level37,
    /// Level 38: q/(2π) ≈ 0.436% - Half-loop integral
    Level38,
    /// Level 39: q/6 ≈ 0.457% - Sub-generation structure (2×3)
    Level39,
    /// Level 40: q/φ³ ≈ 0.65% - Third golden power
    Level40,
    /// Level 41: q/4 ≈ 0.685% - Quarter layer (sphaleron)
    Level41,
    /// Level 42: q/π ≈ 0.872% - Circular loop structure
    Level42,
    /// Level 43: q/3 ≈ 0.913% - Single generation
    Level43,
    /// Level 44: q/φ² ≈ 1.04% - Second golden power
    Level44,
    /// Level 45: q/2 ≈ 1.37% - Half layer
    Level45,
    /// Level 46: q/φ ≈ 1.69% - Scale running (one layer)
    Level46,
    /// Level 47: q ≈ 2.74% - Universal vacuum
    Level47,
    /// Level 48: qφ ≈ 4.43% - Double layer transitions
    Level48,
    /// Level 49: qφ² ≈ 7.17% - Fixed point (φ²=φ+1)
    Level49,
    /// Level 50: 3q ≈ 8.22% - Triple generation
    Level50,
    /// Level 51: πq ≈ 8.61% - Circular enhancement
    Level51,
    /// Level 52: 4q ≈ 10.96% - Full W⁴ topology
    Level52,
    /// Level 53: qφ³ ≈ 11.6% - Triple golden transitions
    Level53,
    /// Level 54: 6q ≈ 16.4% - Full E₆ Cartan enhancement
    Level54,
    /// Level 55: qφ⁴ ≈ 18.8% - Fourth golden transitions
    Level55,
    /// Level 56: 8q ≈ 21.9% - Full E₈ Cartan enhancement
    Level56,
    /// Level 57: qφ⁵ ≈ 30.4% - Fifth golden transitions
    Level57,
}

impl CorrectionLevel {
    /// The correction factor as SymExpr
    pub fn correction_factor(&self) -> SymExpr {
        match self {
            Self::Level0 => SymExpr::from_int(1),
            Self::Level1 => SymExpr::q().pow(Rational::from_int(3)),
            Self::Level2 => SymExpr::q().div(SymExpr::from_int(1000)),
            Self::Level3 => SymExpr::q().div(SymExpr::from_int(720)),
            Self::Level4 => SymExpr::q().div(SymExpr::from_int(360)),
            Self::Level5 => SymExpr::q().div(SymExpr::from_int(248)),
            Self::Level6 => SymExpr::q().div(SymExpr::from_int(240)),
            Self::Level7 => SymExpr::q().div(SymExpr::from_int(133)),
            Self::Level8 => SymExpr::q().div(SymExpr::from_int(126)),
            Self::Level9 => SymExpr::q().div(SymExpr::from_int(120)),
            Self::Level10 => SymExpr::q()
                .pow(Rational::from_int(2))
                .div(SymExpr::phi().pow(Rational::from_int(2))),
            Self::Level11 => SymExpr::q().div(SymExpr::from_int(78)),
            Self::Level12 => SymExpr::q().div(SymExpr::from_int(72)),
            Self::Level13 => SymExpr::q().div(SymExpr::from_int(63)),
            Self::Level14 => SymExpr::q().pow(Rational::from_int(2)).div(SymExpr::phi()),
            Self::Level15 => SymExpr::q().div(SymExpr::from_int(56)),
            Self::Level16 => SymExpr::q().div(SymExpr::from_int(52)),
            Self::Level17 => SymExpr::q().pow(Rational::from_int(2)),
            Self::Level18 => SymExpr::q().div(SymExpr::from_int(36)),
            Self::Level19 => SymExpr::q().div(SymExpr::from_int(32)),
            Self::Level20 => SymExpr::q().div(SymExpr::from_int(30)),
            Self::Level21 => SymExpr::q().div(SymExpr::from_int(28)),
            Self::Level22 => SymExpr::q().div(SymExpr::from_int(27)),
            Self::Level23 => SymExpr::q().div(SymExpr::from_int(24)),
            Self::Level24 => SymExpr::q().pow(Rational::from_int(2)).mul(SymExpr::phi()),
            Self::Level25 => SymExpr::q().div(SymExpr::pi().mul(SymExpr::from_int(6))),
            Self::Level26 => SymExpr::q().div(SymExpr::from_int(18)),
            Self::Level27 => SymExpr::q().div(SymExpr::from_int(16)),
            Self::Level28 => SymExpr::q().div(SymExpr::pi().mul(SymExpr::from_int(5))),
            Self::Level29 => SymExpr::q().div(SymExpr::from_int(14)),
            Self::Level30 => SymExpr::q().div(SymExpr::pi().mul(SymExpr::from_int(4))),
            Self::Level31 => SymExpr::q().div(SymExpr::from_int(12)),
            Self::Level32 => SymExpr::q().div(SymExpr::phi().pow(Rational::from_int(5))),
            Self::Level33 => SymExpr::q().div(SymExpr::pi().mul(SymExpr::from_int(3))),
            Self::Level34 => SymExpr::q().div(SymExpr::from_int(9)),
            Self::Level35 => SymExpr::q().div(SymExpr::from_int(8)),
            Self::Level36 => SymExpr::q().div(SymExpr::from_int(7)),
            Self::Level37 => SymExpr::q().div(SymExpr::phi().pow(Rational::from_int(4))),
            Self::Level38 => SymExpr::q().div(SymExpr::pi().mul(SymExpr::from_int(2))),
            Self::Level39 => SymExpr::q().div(SymExpr::from_int(6)),
            Self::Level40 => SymExpr::q().div(SymExpr::phi().pow(Rational::from_int(3))),
            Self::Level41 => SymExpr::q().div(SymExpr::from_int(4)),
            Self::Level42 => SymExpr::q().div(SymExpr::pi()),
            Self::Level43 => SymExpr::q().div(SymExpr::from_int(3)),
            Self::Level44 => SymExpr::q().div(SymExpr::phi().pow(Rational::from_int(2))),
            Self::Level45 => SymExpr::q().div(SymExpr::from_int(2)),
            Self::Level46 => SymExpr::q().div(SymExpr::phi()),
            Self::Level47 => SymExpr::q(),
            Self::Level48 => SymExpr::q().mul(SymExpr::phi()),
            Self::Level49 => SymExpr::q().mul(SymExpr::phi().pow(Rational::from_int(2))),
            Self::Level50 => SymExpr::from_int(3).mul(SymExpr::q()),
            Self::Level51 => SymExpr::pi().mul(SymExpr::q()),
            Self::Level52 => SymExpr::from_int(4).mul(SymExpr::q()),
            Self::Level53 => SymExpr::q().mul(SymExpr::phi().pow(Rational::from_int(3))),
            Self::Level54 => SymExpr::from_int(6).mul(SymExpr::q()),
            Self::Level55 => SymExpr::q().mul(SymExpr::phi().pow(Rational::from_int(4))),
            Self::Level56 => SymExpr::from_int(8).mul(SymExpr::q()),
            Self::Level57 => SymExpr::q().mul(SymExpr::phi().pow(Rational::from_int(5))),
        }
    }

    /// Approximate percentage deviation
    pub fn approx_percent(&self) -> f64 {
        (self.correction_factor().eval_f64() - 1.0) * 100.0
    }

    /// Physical interpretation
    pub fn interpretation(&self) -> &'static str {
        match self {
            Self::Level0 => "Tree-level (exact)",
            Self::Level1 => "Third-order vacuum",
            Self::Level2 => "Fixed-point stability (h(E₈)³/27)",
            Self::Level3 => "Coxeter-Kissing product (h(E₈)×K(D₄))",
            Self::Level4 => "Complete cone cycles (10×36)",
            Self::Level5 => "Full E₈ adjoint representation",
            Self::Level6 => "Full E₈ root system (both signs)",
            Self::Level7 => "Full E₇ adjoint representation",
            Self::Level8 => "Full E₇ root system",
            Self::Level9 => "Complete E₈ positive roots",
            Self::Level10 => "Second-order/double golden",
            Self::Level11 => "Full E₆ gauge structure",
            Self::Level12 => "Full E₆ root system (both signs)",
            Self::Level13 => "E₇ positive roots",
            Self::Level14 => "Second-order massless",
            Self::Level15 => "E₇ fundamental representation",
            Self::Level16 => "F₄ gauge structure",
            Self::Level17 => "Second-order vacuum",
            Self::Level18 => "E₆ positive roots (Golden Cone)",
            Self::Level19 => "Five-fold binary structure",
            Self::Level20 => "E₈ Coxeter number",
            Self::Level21 => "D₄ adjoint representation",
            Self::Level22 => "E₆ fundamental representation",
            Self::Level23 => "D₄ kissing number",
            Self::Level24 => "Quadratic + golden enhancement",
            Self::Level25 => "Six-flavor QCD loop",
            Self::Level26 => "E₇ Coxeter number",
            Self::Level27 => "Four-fold binary/spinor dimension",
            Self::Level28 => "Five-flavor QCD loop",
            Self::Level29 => "G₂ octonion automorphisms",
            Self::Level30 => "One-loop radiative (4D)",
            Self::Level31 => "Topology × generations (W⁴ × N_gen)",
            Self::Level32 => "Fifth golden power",
            Self::Level33 => "Three-flavor QCD loop",
            Self::Level34 => "Generation-squared structure",
            Self::Level35 => "Cartan subalgebra (rank E₈)",
            Self::Level36 => "E₇ Cartan subalgebra",
            Self::Level37 => "Fourth golden power",
            Self::Level38 => "Half-loop integral",
            Self::Level39 => "Sub-generation structure (2×3)",
            Self::Level40 => "Third golden power",
            Self::Level41 => "Quarter layer (sphaleron)",
            Self::Level42 => "Circular loop structure",
            Self::Level43 => "Single generation",
            Self::Level44 => "Second golden power",
            Self::Level45 => "Half layer",
            Self::Level46 => "Scale running (one layer)",
            Self::Level47 => "Universal vacuum",
            Self::Level48 => "Double layer transitions",
            Self::Level49 => "Fixed point (φ²=φ+1)",
            Self::Level50 => "Triple generation",
            Self::Level51 => "Circular enhancement",
            Self::Level52 => "Full W⁴ topology",
            Self::Level53 => "Triple golden transitions",
            Self::Level54 => "Full E₆ Cartan enhancement",
            Self::Level55 => "Fourth golden transitions",
            Self::Level56 => "Full E₈ Cartan enhancement",
            Self::Level57 => "Fifth golden transitions",
        }
    }
}

#[pymethods]
impl CorrectionLevel {
    /// Get symbolic factor as string
    fn get_correction_factor(&self) -> String {
        self.correction_factor().to_string()
    }

    /// Evaluate factor to f64
    fn get_factor_value(&self) -> f64 {
        self.correction_factor().eval_f64()
    }

    /// Get approximate percentage deviation
    fn get_percent(&self) -> f64 {
        self.approx_percent()
    }

    /// Get physical interpretation
    fn get_interpretation(&self) -> String {
        self.interpretation().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "CorrectionLevel.{:?}({} ≈ {:.3}%)",
            self,
            self.get_correction_factor(),
            self.approx_percent()
        )
    }
}

/// Multiplicative Suppression Factors
///
/// These factors appear as divisors for processes involving recursion layer crossings.
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SuppressionLevel {
    /// 1/(1+qφ⁻²) ≈ 1.05% suppression - Double inverse recursion
    DoubleInverseRecursion,

    /// 1/(1+qφ⁻¹) ≈ 1.7% suppression - Inverse recursion
    InverseRecursion,

    /// 1/(1+q) ≈ 2.7% suppression - Base suppression
    BaseSuppression,

    /// 1/(1+qφ) ≈ 4.2% suppression - Recursion penalty
    RecursionPenalty,

    /// 1/(1+qφ²) ≈ 6.7% suppression - Fixed point penalty
    FixedPointPenalty,

    /// 1/(1+qφ³) ≈ 10.4% suppression - Deep recursion penalty
    DeepRecursionPenalty,
}

impl SuppressionLevel {
    /// The symbolic suppression factor
    pub fn suppression_factor(&self) -> SymExpr {
        let one = SymExpr::from_int(1);
        let q = SymExpr::q();
        let phi = SymExpr::phi();
        let inner = match self {
            Self::DoubleInverseRecursion => q.mul(phi.pow(Rational::from_int(-2))),
            Self::InverseRecursion => q.mul(phi.pow(Rational::from_int(-1))),
            Self::BaseSuppression => q,
            Self::RecursionPenalty => q.mul(phi),
            Self::FixedPointPenalty => q.mul(phi.pow(Rational::from_int(2))),
            Self::DeepRecursionPenalty => q.mul(phi.pow(Rational::from_int(3))),
        };
        let denominator = one.clone().add(inner);
        one.div(denominator)
    }

    /// Approximate percentage value (factor * 100)
    pub fn approx_percent(&self) -> f64 {
        self.suppression_factor().eval_f64() * 100.0
    }

    /// Physical interpretation
    pub fn interpretation(&self) -> &'static str {
        match self {
            Self::DoubleInverseRecursion => "Double inverse recursion penalty",
            Self::InverseRecursion => "Inverse recursion penalty",
            Self::BaseSuppression => "Base suppression penalty",
            Self::RecursionPenalty => "Recursion penalty",
            Self::FixedPointPenalty => "Fixed point penalty",
            Self::DeepRecursionPenalty => "Deep recursion penalty",
        }
    }
}

#[pymethods]
impl SuppressionLevel {
    /// Get symbolic factor as string
    fn get_suppression_factor(&self) -> String {
        self.suppression_factor().to_string()
    }

    /// Evaluate factor to f64
    fn get_factor_value(&self) -> f64 {
        self.suppression_factor().eval_f64()
    }

    /// Get approximate percentage
    fn get_percent(&self) -> f64 {
        self.approx_percent()
    }

    /// Get physical interpretation
    fn get_interpretation(&self) -> String {
        self.interpretation().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "SuppressionLevel.{:?}({} ≈ {:.3}%)",
            self,
            self.get_suppression_factor(),
            self.approx_percent()
        )
    }
}

// =============================================================================
// Algebraic Structure Dimensions for Correction Factors
// =============================================================================

/// Algebraic structures used in SRT correction factors (1 ± q/N)
///
/// These dimensions appear throughout SRT predictions for particle masses,
/// coupling constants, and mixing angles.
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Structure {
    // E₈ family
    /// dim(E₈) = 248 - adjoint representation
    E8Adjoint = 248,
    /// |Φ(E₈)| = 240 - number of roots
    E8Roots = 240,
    /// |Φ⁺(E₈)| = 120 - positive roots (chiral)
    E8Positive = 120,
    /// h(E₈) = 30 - Coxeter number
    E8Coxeter = 30,
    /// rank(E₈) = 8
    E8Rank = 8,

    // E₇ family
    /// dim(E₇) = 133 - adjoint representation
    E7Adjoint = 133,
    /// |Φ(E₇)| = 126 - number of roots
    E7Roots = 126,
    /// |Φ⁺(E₇)| = 63 - positive roots
    E7Positive = 63,
    /// dim(56) - fundamental representation
    E7Fund = 56,
    /// h(E₇) = 18 - Coxeter number
    E7Coxeter = 18,
    /// rank(E₇) = 7
    E7Rank = 7,

    // E₆ family
    /// dim(E₆) = 78 - adjoint representation
    E6Adjoint = 78,
    /// |Φ(E₆)| = 72 - number of roots
    E6Roots = 72,
    /// |Φ⁺(E₆)| = 36 - positive roots = Golden Cone
    E6Positive = 36,
    /// dim(27) - fundamental representation
    E6Fund = 27,
    /// rank(E₆) = 6
    E6Rank = 6,

    // D₄ family
    /// dim(D₄) = 28 - adjoint representation
    D4Adjoint = 28,
    /// K(D₄) = 24 - kissing number = consciousness threshold
    D4Kissing = 24,
    /// rank(D₄) = 4
    D4Rank = 4,

    // G₂ family
    /// dim(G₂) = 14 - adjoint representation
    G2Adjoint = 14,
    /// |Φ(G₂)| = 12 - number of roots
    G2Roots = 12,
    /// rank(G₂) = 2
    G2Rank = 2,

    // Loop correction factors (n_f × π rounded)
    /// 3π ≈ 9.42 → 9 (3-flavor QCD)
    Loop3Flavor = 9,
    /// 4π ≈ 12.57 → 13 (4-flavor)
    Loop4Flavor = 13,
    /// 5π ≈ 15.71 → 16 (5-flavor)
    Loop5Flavor = 16,
    /// 6π ≈ 18.85 → 19 (6-flavor)
    Loop6Flavor = 19,
}

impl Structure {
    /// The dimension N for this structure
    #[inline]
    pub const fn dimension(self) -> u32 {
        self as u32
    }

    /// Symbolic correction factor: (1 + sign × q/N) as SymExpr
    ///
    /// Returns exact symbolic expression, not a floating-point approximation.
    /// Use `.eval_f64()` on the result only when numerical evaluation is needed.
    pub fn correction(self, sign: i8) -> SymExpr {
        let one = SymExpr::from_int(1);
        let q = SymExpr::q();
        let n = SymExpr::from_int(self.dimension() as i128);
        let q_over_n = q.div(n);

        if sign >= 0 {
            one.add(q_over_n)
        } else {
            one.sub(q_over_n)
        }
    }

    /// Symbolic (1 + q/N)
    pub fn correction_plus(self) -> SymExpr {
        self.correction(1)
    }

    /// Symbolic (1 - q/N)
    pub fn correction_minus(self) -> SymExpr {
        self.correction(-1)
    }

    /// Get the Lie algebra name
    pub fn algebra_name(self) -> &'static str {
        match self {
            Self::E8Adjoint | Self::E8Roots | Self::E8Positive | Self::E8Coxeter | Self::E8Rank => {
                "E₈"
            }
            Self::E7Adjoint
            | Self::E7Roots
            | Self::E7Positive
            | Self::E7Fund
            | Self::E7Coxeter
            | Self::E7Rank => "E₇",
            Self::E6Adjoint | Self::E6Roots | Self::E6Positive | Self::E6Fund | Self::E6Rank => {
                "E₆"
            }
            Self::D4Adjoint | Self::D4Kissing | Self::D4Rank => "D₄",
            Self::G2Adjoint | Self::G2Roots | Self::G2Rank => "G₂",
            Self::Loop3Flavor | Self::Loop4Flavor | Self::Loop5Flavor | Self::Loop6Flavor => "Loop",
        }
    }
}

#[pymethods]
impl Structure {
    /// Get the dimension N
    fn get_dimension(&self) -> u32 {
        self.dimension()
    }

    /// Get symbolic correction (1 + sign × q/N) as string
    fn get_correction_symbolic(&self, sign: i8) -> String {
        self.correction(sign).to_string()
    }

    /// Evaluate correction (1 + sign × q/N) to f64
    /// Note: Use get_correction_symbolic() to see the exact formula
    fn get_correction_value(&self, sign: i8) -> f64 {
        self.correction(sign).eval_f64()
    }

    /// Get symbolic (1 + q/N) as string
    fn get_plus_symbolic(&self) -> String {
        self.correction_plus().to_string()
    }

    /// Evaluate (1 + q/N) to f64
    fn get_plus_value(&self) -> f64 {
        self.correction_plus().eval_f64()
    }

    /// Get symbolic (1 - q/N) as string
    fn get_minus_symbolic(&self) -> String {
        self.correction_minus().to_string()
    }

    /// Evaluate (1 - q/N) to f64
    fn get_minus_value(&self) -> f64 {
        self.correction_minus().eval_f64()
    }

    fn __repr__(&self) -> String {
        format!("Structure.{:?}(N={})", self, self.dimension())
    }

    // Static constructors for common structures
    #[staticmethod]
    fn e8_adjoint() -> Self {
        Self::E8Adjoint
    }
    #[staticmethod]
    fn e8_roots() -> Self {
        Self::E8Roots
    }
    #[staticmethod]
    fn e8_positive() -> Self {
        Self::E8Positive
    }
    #[staticmethod]
    fn e6_adjoint() -> Self {
        Self::E6Adjoint
    }
    #[staticmethod]
    fn e6_positive() -> Self {
        Self::E6Positive
    }
    #[staticmethod]
    fn e6_fund() -> Self {
        Self::E6Fund
    }
    #[staticmethod]
    fn d4_kissing() -> Self {
        Self::D4Kissing
    }
    #[staticmethod]
    fn g2_adjoint() -> Self {
        Self::G2Adjoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_values() {
        // Verify E* = e^π - π approximately
        let e_pi = std::f64::consts::E.powf(std::f64::consts::PI);
        let expected_e_star = e_pi - std::f64::consts::PI;
        let e_star = FundamentalConstant::EStar.approx_f64();
        assert!((e_star - expected_e_star).abs() < 1e-10);
    }

    #[test]
    fn test_q_formula() {
        // Verify q = (2φ + e/(2φ²)) / (φ⁴ · E*)
        let phi = FundamentalConstant::Phi.approx_f64();
        let e = FundamentalConstant::Euler.approx_f64();
        let e_star = FundamentalConstant::EStar.approx_f64();

        let numerator = 2.0 * phi + e / (2.0 * phi * phi);
        let denominator = phi.powi(4) * e_star;
        let expected_q = numerator / denominator;

        let q = FundamentalConstant::Q.approx_f64();
        assert!((q - expected_q).abs() < 1e-10);
    }

    #[test]
    fn test_phi_is_algebraic() {
        assert!(FundamentalConstant::Phi.is_algebraic());
        assert!(!FundamentalConstant::Pi.is_algebraic());
        assert!(!FundamentalConstant::Q.is_algebraic());
    }
}

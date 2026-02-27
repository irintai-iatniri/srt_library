use std::ops::{Add, Sub, Mul};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Vibe {
    pub real: i128, // Q32.32 Fixed-point
    pub imag: i128, // Q32.32 Fixed-point
}

impl Vibe {
    pub const SHIFT: u32 = 32;
    pub const SCALE: i128 = 1 << Self::SHIFT;
    pub const Q_DEFICIT: i128 = 117658758; // q × 2³² — derived from crate::constants::Q_DEFICIT_CONST

    pub fn from_f64(f: f64) -> Self {
        Self { real: (f * Self::SCALE as f64) as i128, imag: 0 }
    }

    pub fn rotate_90(self) -> Self {
        Self { real: -self.imag, imag: self.real }
    }

    pub fn apply_drag(self) -> Self {
        let drag_factor = Self::SCALE - Self::Q_DEFICIT;
        Self {
            real: (self.real * drag_factor) >> Self::SHIFT,
            imag: (self.imag * drag_factor) >> Self::SHIFT,
        }
    }
}

// --- OPERATOR OVERLOADING (The "Just Work" Logic) ---

impl Add for Vibe {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { real: self.real + rhs.real, imag: self.imag + rhs.imag }
    }
}

impl Sub for Vibe {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { real: self.real - rhs.real, imag: self.imag - rhs.imag }
    }
}

impl Mul for Vibe {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        // Post-multiplication shift maintains Q32.32 scale
        let ac = (self.real * rhs.real) >> Self::SHIFT;
        let bd = (self.imag * rhs.imag) >> Self::SHIFT;
        let ad = (self.real * rhs.imag) >> Self::SHIFT;
        let bc = (self.imag * rhs.real) >> Self::SHIFT;
        Self { real: ac - bd, imag: ad + bc }
    }
}

impl fmt::Display for Vibe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}i)", 
            self.real as f64 / Self::SCALE as f64, 
            self.imag as f64 / Self::SCALE as f64)
    }
}
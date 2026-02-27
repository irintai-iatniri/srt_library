#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Vibe { pub real: i128, pub imag: i128 }
impl Vibe {
    pub const SHIFT: u32 = 32;
    pub const SCALE: i128 = 1 << 32;
    pub fn from_f64(f: f64) -> Self { Self { real: (f * 4294967296.0) as i128, imag: 0 } }
    pub fn add(self, other: Self) -> Self { Self { real: self.real + other.real, imag: self.imag + other.imag } }
    pub fn rotate_90(self) -> Self { Self { real: -self.imag, imag: self.real } }
}
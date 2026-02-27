use core::ops::{Add, Sub, Mul, Div, Neg};

pub trait ExactScalar: 
    Copy + Clone + PartialEq + Eq + PartialOrd + Ord + 
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> 
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

use _core::exact::{FixedPoint64, ExactScalar, Rational};
use _core::exact::transcendental::PI;

fn fp_from_rational(r: &Rational) -> FixedPoint64 {
    let num = r.numerator();
    let den = r.denominator();
    let scaled_num = num * (1 << 32); 
    FixedPoint64((scaled_num / den) as i64)
}

#[test]
fn test_fixed_point_basic_arithmetic() {
    let a = FixedPoint64::from_f64(1.5);
    let b = FixedPoint64::from_f64(2.5);
    
    // Addition
    let sum = a + b;
    assert_eq!(sum.to_f64(), 4.0);
    
    // Subtraction
    let diff = b - a;
    assert_eq!(diff.to_f64(), 1.0);
    
    // Multiplication
    let prod = a * b;
    // 1.5 * 2.5 = 3.75
    assert_eq!(prod.to_f64(), 3.75);
    
    // Division
    // 3.75 / 1.5 = 2.5
    let quot = prod / a;
    assert_eq!(quot.to_f64(), 2.5);
}

#[test]
fn test_rational_to_fixed_point() {
    let r = Rational::new(3, 2); // 1.5
    let fp = fp_from_rational(&r);
    assert_eq!(fp.to_f64(), 1.5);
    
    let r2 = Rational::new(1, 3); // 0.333...
    let fp2 = fp_from_rational(&r2);
    // tolerance check for 1/3
    let diff = (fp2.to_f64() - (1.0/3.0)).abs();
    assert!(diff < 1e-9, "Fixed point 1/3 precision error: {}", diff);
}

#[test]
fn test_cordic_sine() {
    // Sin(0) = 0
    let zero = FixedPoint64::ZERO;
    let sin_0 = zero.sin().to_f64();
    println!("Sin(0) = {}", sin_0);
    assert!(sin_0.abs() < 1e-4, "Sin(0) error: {}", sin_0);
    
    // Sin(PI/2) = 1
    // We construct PI/2 carefully using the raw constant
    let pi_2 = FixedPoint64(PI / 2);
    let sin_pi_2 = pi_2.sin();
    
    println!("Sin(PI/2) = {}", sin_pi_2.to_f64());
    assert!((sin_pi_2.to_f64() - 1.0).abs() < 1e-4);
    
    // Sin(PI) = 0
    let sin_pi = FixedPoint64(PI).sin();
    println!("Sin(PI) = {}", sin_pi.to_f64());
    assert!(sin_pi.to_f64().abs() < 1e-4);
}

#[test]
fn test_cordic_cosine() {
    // Cos(0) = 1
    let zero = FixedPoint64::ZERO;
    let cos_0 = zero.cos().to_f64();
    println!("Cos(0) = {}", cos_0);
    assert!((cos_0 - 1.0).abs() < 1e-4, "Cos(0) error: {}", (cos_0 - 1.0).abs());
    
    // Cos(PI/2) = 0
    let pi_2 = FixedPoint64(PI / 2);
    let cos_pi_2 = pi_2.cos();
    
    println!("Cos(PI/2) = {}", cos_pi_2.to_f64());
    assert!(cos_pi_2.to_f64().abs() < 1e-4);
}

#[test]
fn test_exp() {
    // exp(0) = 1
    assert_eq!(FixedPoint64::ZERO.exp().to_f64(), 1.0);
    
    // exp(1) = e
    let one = FixedPoint64::ONE;
    let e = one.exp();
    
    println!("exp(1) = {}", e.to_f64());
    assert!((e.to_f64() - std::f64::consts::E).abs() < 1e-4);
}

#[test]
fn test_sqrt() {
    // sqrt(4) = 2
    let four = FixedPoint64::from_f64(4.0);
    assert_eq!(four.sqrt().to_f64(), 2.0);
    
    // sqrt(2)
    let two = FixedPoint64::from_f64(2.0);
    let root2 = two.sqrt();
    println!("sqrt(2) = {}", root2.to_f64());
    assert!((root2.to_f64() - std::f64::consts::SQRT_2).abs() < 1e-6);
}

#[test]
fn test_dhsr_cycle_logic_fixed() {
    // Construct a simple state [1.0, 0.0] (complex 1+0i)
    // Syntony = 1 / norm = 1/1 = 1.0 (assuming simplistic syntony for test)
    
    let val = FixedPoint64::ONE;
    
    // Test differentiation: D(x) = x * syntony
    let syntony = FixedPoint64::from_f64(0.5);
    let diff = val * syntony;
    assert_eq!(diff.to_f64(), 0.5);
    
    // Test harmonization: H(x) = x + syntony
    let harm = val + syntony;
    assert_eq!(harm.to_f64(), 1.5);
    
    // Test recursion: R(x) = x^2
    let rec = val * val;
    assert_eq!(rec.to_f64(), 1.0);
}

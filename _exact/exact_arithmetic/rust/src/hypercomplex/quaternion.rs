use pyo3::prelude::*;
use crate::exact::{FixedPoint64, ExactScalar};

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Quaternion {
    #[pyo3(get, set)]
    pub a: f64,
    #[pyo3(get, set)]
    pub b: f64,
    #[pyo3(get, set)]
    pub c: f64,
    #[pyo3(get, set)]
    pub d: f64,
}

#[pymethods]
impl Quaternion {
    #[new]
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Quaternion { a, b, c, d }
    }

    #[getter]
    pub fn real(&self) -> f64 {
        self.a
    }

    #[getter]
    pub fn imag(&self) -> Vec<f64> {
        vec![self.b, self.c, self.d]
    }

    pub fn conjugate(&self) -> Self {
        // Negation is exact in both domains, but let's be consistent if we enforce grid
        // For simple negation, f64 and FixedPoint64 are consistent enough usually, 
        // but strictly:
        let a_fp = FixedPoint64::from_f64(self.a);
        let b_fp = FixedPoint64::from_f64(self.b);
        let c_fp = FixedPoint64::from_f64(self.c);
        let d_fp = FixedPoint64::from_f64(self.d);
        
        Quaternion::new(
            a_fp.to_f64(),
            (-b_fp).to_f64(),
            (-c_fp).to_f64(),
            (-d_fp).to_f64()
        )
    }

    pub fn norm(&self) -> f64 {
        let a_fp = FixedPoint64::from_f64(self.a);
        let b_fp = FixedPoint64::from_f64(self.b);
        let c_fp = FixedPoint64::from_f64(self.c);
        let d_fp = FixedPoint64::from_f64(self.d);
        
        let sum_sq = a_fp*a_fp + b_fp*b_fp + c_fp*c_fp + d_fp*d_fp;
        sum_sq.sqrt().to_f64()
    }

    pub fn normalize(&self) -> Self {
        let a_fp = FixedPoint64::from_f64(self.a);
        let b_fp = FixedPoint64::from_f64(self.b);
        let c_fp = FixedPoint64::from_f64(self.c);
        let d_fp = FixedPoint64::from_f64(self.d);
        
        let sum_sq = a_fp*a_fp + b_fp*b_fp + c_fp*c_fp + d_fp*d_fp;
        let n = sum_sq.sqrt();
        
        if n.0 == 0 {
            return *self;
        }
        
        Quaternion::new(
            (a_fp / n).to_f64(),
            (b_fp / n).to_f64(),
            (c_fp / n).to_f64(),
            (d_fp / n).to_f64()
        )
    }

    pub fn inverse(&self) -> Self {
        let a_fp = FixedPoint64::from_f64(self.a);
        let b_fp = FixedPoint64::from_f64(self.b);
        let c_fp = FixedPoint64::from_f64(self.c);
        let d_fp = FixedPoint64::from_f64(self.d);
        
        let n2 = a_fp*a_fp + b_fp*b_fp + c_fp*c_fp + d_fp*d_fp;
        
        if n2.0 == 0 {
             return Quaternion::new(
                 (a_fp / n2).to_f64(),
                 (-b_fp / n2).to_f64(),
                 (-c_fp / n2).to_f64(),
                 (-d_fp / n2).to_f64()
             );
        }
        
        Quaternion::new(
            (a_fp / n2).to_f64(),
            (-b_fp / n2).to_f64(),
            (-c_fp / n2).to_f64(),
            (-d_fp / n2).to_f64()
        )
    }

    // Arithmetic

    fn __add__(&self, other: &Quaternion) -> Self {
        let s = self.to_fixed();
        let o = other.to_fixed();
        Quaternion::from_fixed(
            s.0 + o.0, s.1 + o.1, s.2 + o.2, s.3 + o.3
        )
    }

    fn __sub__(&self, other: &Quaternion) -> Self {
        let s = self.to_fixed();
        let o = other.to_fixed();
        Quaternion::from_fixed(
            s.0 - o.0, s.1 - o.1, s.2 - o.2, s.3 - o.3
        )
    }

    fn __neg__(&self) -> Self {
        let s = self.to_fixed();
        Quaternion::from_fixed(
            -s.0, -s.1, -s.2, -s.3
        )
    }

    /// Hamilton product
    fn __mul__(&self, other: &Quaternion) -> Self {
        let (a1, b1, c1, d1) = self.to_fixed();
        let (a2, b2, c2, d2) = other.to_fixed();
        
        Quaternion::from_fixed(
            a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2
        )
    }

    fn __truediv__(&self, other: &Quaternion) -> Self {
        // q1 / q2 = q1 * q2^-1
        self.__mul__(&other.inverse())
    }

    fn __repr__(&self) -> String {
        format!(
            "Quaternion(a={:.4}, b={:.4}, c={:.4}, d={:.4})",
            self.a, self.b, self.c, self.d
        )
    }

    // Advanced Ops

    pub fn to_rotation_matrix(&self) -> Vec<Vec<f64>> {
        let q = self.normalize();
        
        // Use FixedPoint calculation for matrix elements
        // This is important for exact orthogonality
        let (a, b, c, d) = q.to_fixed();
        let one = FixedPoint64::one();
        let two = FixedPoint64::from_f64(2.0);
        
        let m00 = one - two * (c*c + d*d);
        let m01 = two * (b*c - d*a);
        let m02 = two * (b*d + c*a);
        
        let m10 = two * (b*c + d*a);
        let m11 = one - two * (b*b + d*d);
        let m12 = two * (c*d - b*a);
        
        let m20 = two * (b*d - c*a);
        let m21 = two * (c*d + b*a);
        let m22 = one - two * (b*b + c*c);

        vec![
            vec![m00.to_f64(), m01.to_f64(), m02.to_f64()],
            vec![m10.to_f64(), m11.to_f64(), m12.to_f64()],
            vec![m20.to_f64(), m21.to_f64(), m22.to_f64()],
        ]
    }

    #[staticmethod]
    pub fn from_axis_angle(axis: Vec<f64>, theta: f64) -> PyResult<Self> {
        if axis.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Axis must be length 3",
            ));
        }
        
        let ax = FixedPoint64::from_f64(axis[0]);
        let ay = FixedPoint64::from_f64(axis[1]);
        let az = FixedPoint64::from_f64(axis[2]);
        let theta_fp = FixedPoint64::from_f64(theta);
        
        let half = theta_fp / FixedPoint64::from_f64(2.0);
        let s = half.sin();
        let c = half.cos();
        
        let norm = (ax*ax + ay*ay + az*az).sqrt();
        
        if norm.0 == 0 {
            return Ok(Quaternion::new(1.0, 0.0, 0.0, 0.0));
        }
        
        Ok(Quaternion::new(
            c.to_f64(),
            (s * ax / norm).to_f64(),
            (s * ay / norm).to_f64(),
            (s * az / norm).to_f64(),
        ))
    }

    #[staticmethod]
    pub fn slerp(q1: &Quaternion, q2: &Quaternion, t: f64) -> Self {
        let (a1, b1, c1, d1) = q1.to_fixed();
        let (a2, b2, c2, d2) = q2.to_fixed();
        let t_fp = FixedPoint64::from_f64(t);

        let mut dot = a1*a2 + b1*b2 + c1*c2 + d1*d2;

        // Handle negative dot (opposite hemispheres)
        let (a2, b2, c2, d2) = if dot.0 < 0 {
            dot = -dot;
            (-a2, -b2, -c2, -d2)
        } else {
            (a2, b2, c2, d2)
        };

        // Clamp dot to [-1, 1] usually, but logic implies dot <= 1 here.
        // FixedPoint logic handles checking.

        let theta = dot.acos();
        let sin_theta = theta.sin();
        
        let epsilon = FixedPoint64::from_f64(1e-6);

        if sin_theta < epsilon && sin_theta > -epsilon {
            // Linear interpolation: (1-t)*q1 + t*q2
            let one = FixedPoint64::one();
            let one_minus_t = one - t_fp;
            let qa = Quaternion::from_fixed(
                one_minus_t * a1 + t_fp * a2,
                one_minus_t * b1 + t_fp * b2,
                one_minus_t * c1 + t_fp * c2,
                one_minus_t * d1 + t_fp * d2
            );
            return qa.normalize();
        } 
        
        let one_minus_t = FixedPoint64::one() - t_fp;
        let s1 = (one_minus_t * theta).sin() / sin_theta;
        let s2 = (t_fp * theta).sin() / sin_theta;
        
        Quaternion::from_fixed(
            s1 * a1 + s2 * a2,
            s1 * b1 + s2 * b2,
            s1 * c1 + s2 * c2,
            s1 * d1 + s2 * d2
        )
    }
}

// Helper methods not exposed to Python directly but useful internally
impl Quaternion {
    fn to_fixed(&self) -> (FixedPoint64, FixedPoint64, FixedPoint64, FixedPoint64) {
        (
            FixedPoint64::from_f64(self.a),
            FixedPoint64::from_f64(self.b),
            FixedPoint64::from_f64(self.c),
            FixedPoint64::from_f64(self.d)
        )
    }
    
    fn from_fixed(a: FixedPoint64, b: FixedPoint64, c: FixedPoint64, d: FixedPoint64) -> Self {
        Quaternion {
             a: a.to_f64(),
             b: b.to_f64(),
             c: c.to_f64(),
             d: d.to_f64()
        }
    }
}

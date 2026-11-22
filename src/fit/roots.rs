//! The largest inverse root of a lag polynomial, for the auto-ARIMA root check.
//!
//! R's `auto.arima` and pmdarima discard a candidate whose AR or MA polynomial has an
//! inverse root too close to the unit circle: such a fit sits on the edge of
//! stationarity or invertibility and is numerically unreliable.

/// A complex number, just enough for the Durand–Kerner iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Complex {
    /// The real part.
    re: f64,
    /// The imaginary part.
    im: f64,
}

impl Complex {
    /// `self + other`.
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// `self - other`.
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// `self * other`.
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// `self / other`.
    fn div(self, other: Self) -> Self {
        let scale = other.re * other.re + other.im * other.im;
        Self {
            re: (self.re * other.re + self.im * other.im) / scale,
            im: (self.im * other.re - self.re * other.im) / scale,
        }
    }

    /// `|self|`.
    fn abs(self) -> f64 {
        crate::math::sqrt(self.re * self.re + self.im * self.im)
    }
}

/// The most Durand–Kerner sweeps; convergence is quadratic for simple roots.
const MAX_SWEEPS: usize = 500;

/// The largest modulus among the inverse roots of `1 + a_1 z + ... + a_n z^n`.
///
/// The inverse roots are the roots of `z^n + a_1 z^(n-1) + ... + a_n`, found with the
/// Durand–Kerner (Weierstrass) iteration. Returns 0 for an empty polynomial.
pub(crate) fn max_inverse_root(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    // Standard starting points: powers of a complex number that is neither real nor a root of unity.
    let seed = Complex { re: 0.4, im: 0.9 };
    let mut roots = alloc::vec![seed; n];
    for i in 1..n {
        roots[i] = roots[i - 1].mul(seed);
    }

    for _ in 0..MAX_SWEEPS {
        let mut largest_step: f64 = 0.0;
        for i in 0..n {
            let z = roots[i];
            let value = evaluate(a, z);
            let denominator = (0..n)
                .filter(|&j| j != i)
                .fold(Complex { re: 1.0, im: 0.0 }, |product, j| {
                    product.mul(z.sub(roots[j]))
                });
            let step = value.div(denominator);
            roots[i] = z.sub(step);
            largest_step = largest_step.max(step.abs());
        }

        if largest_step < 1e-14 {
            break;
        }
    }

    roots.iter().fold(0.0, |m, z| m.max(z.abs()))
}

/// `z^n + a_1 z^(n-1) + ... + a_n` by Horner's rule.
fn evaluate(a: &[f64], z: Complex) -> Complex {
    a.iter().fold(Complex { re: 1.0, im: 0.0 }, |sum, &c| {
        sum.mul(z).add(Complex { re: c, im: 0.0 })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar1_inverse_root_is_its_coefficient() {
        // AR(1) with phi = 0.7: 1 - 0.7 z has inverse root 0.7.
        assert!((max_inverse_root(&[-0.7]) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn complex_pair() {
        // 1 - 2 r cos(w) z + r^2 z^2 has inverse roots r e^(+-i w); r = 0.9.
        let (r, w) = (0.9_f64, 1.1_f64);
        let a = [-2.0 * r * w.cos(), r * r];
        assert!((max_inverse_root(&a) - r).abs() < 1e-12);
    }

    #[test]
    fn seasonal_polynomial() {
        // 1 - 0.5 z^4 has four inverse roots of modulus 0.5^(1/4).
        let a = [0.0, 0.0, 0.0, -0.5];
        assert!((max_inverse_root(&a) - 0.5_f64.powf(0.25)).abs() < 1e-12);
        assert_eq!(max_inverse_root(&[]), 0.0);
    }
}

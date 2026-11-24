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
///
/// Fails closed: if the iteration breaks down (two iterates coincide, so a step divides
/// by zero), the result is `+inf`, so the caller rejects the candidate rather than
/// accepting it on a partial set of roots.
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

    if roots.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return f64::INFINITY;
    }

    roots.iter().fold(0.0, |m, z| m.max(z.abs()))
}

/// `z^n + a_1 z^(n-1) + ... + a_n` by Horner's rule.
fn evaluate(a: &[f64], z: Complex) -> Complex {
    a.iter().fold(Complex { re: 1.0, im: 0.0 }, |sum, &c| {
        sum.mul(z).add(Complex { re: c, im: 0.0 })
    })
}

/// Whether every root of `z^n + a_1 z^(n-1) + ... + a_n` has a modulus below `radius`: the
/// question [`max_inverse_root`] answers by finding all roots, decided here without them.
///
/// This is the Schur–Cohn test. It scales the polynomial to the unit circle, then reduces
/// it one degree at a time: `p` has all roots inside the circle exactly when its constant
/// term is smaller in modulus than its leading one (the reflection coefficient
/// `k = c_0 / c_n` has `|k| < 1`), and the reduced polynomial `(p(z) - k p*(z)) / z` has
/// all roots inside too. It costs `O(n^2)` operations and
/// no iteration. A root exactly on the circle counts as outside; a `NaN` coefficient fails.
pub(crate) fn roots_inside(a: &[f64], radius: f64) -> bool {
    // `c_i` is the coefficient of `z^i` of `p(radius z)`, leading coefficient last.
    let n = a.len();
    let mut scale = 1.0;
    let mut c = alloc::vec![0.0; n + 1];
    for i in 0..=n {
        let coefficient = if i == n { 1.0 } else { a[n - 1 - i] };
        c[i] = coefficient * scale;
        scale *= radius;
    }

    // Each step reads the previous polynomial from both ends, so it writes into a copy. It
    // works with the reflection coefficient `k = c_0 / c_n` and rescales the result to a
    // leading coefficient of 1: without that, the coefficients square at every step and
    // underflow for roots near the circle.
    let mut previous = c.clone();
    for degree in (1..=n).rev() {
        let k = c[0] / c[degree];
        // A `NaN` coefficient gives a `NaN` `k`, which fails too.
        if k.is_nan() || k.abs() >= 1.0 {
            return false;
        }

        previous[..=degree].copy_from_slice(&c[..=degree]);
        let leading = previous[degree] - k * previous[0];
        for i in 0..degree {
            c[i] = (previous[i + 1] - k * previous[degree - 1 - i]) / leading;
        }
    }

    true
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

    #[test]
    fn repeated_unit_root_is_not_missed() {
        // (1 - z)^2: a double inverse root at 1, on the unit circle.
        assert!(max_inverse_root(&[-2.0, 1.0]) > 0.99);
        // (1 - 0.5 z)^2: a double inverse root at 0.5.
        assert!((max_inverse_root(&[-1.0, 0.25]) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn breakdown_fails_closed() {
        // A NaN coefficient poisons every iterate; the check must reject, not return 0.
        assert_eq!(max_inverse_root(&[f64::NAN, 0.1]), f64::INFINITY);
    }

    #[test]
    fn schur_cohn_agrees_with_the_roots() {
        use alloc::vec::Vec;

        // Random polynomials of degrees 1 to 13, against the moduli of their roots.
        let mut state = 7_u64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f64::from(u32::try_from(state >> 40).expect("24 bits")) / f64::from(1_u32 << 24) * 2.0
                - 1.0
        };
        let mut checked = 0;
        for degree in 1..=13 {
            for _ in 0..200 {
                let a: Vec<f64> = (0..degree).map(|_| 0.6 * uniform()).collect();
                let largest = max_inverse_root(&a);
                // Skip polynomials with a root too near the circle to decide either way.
                if (largest - 1.0).abs() > 1e-6 {
                    assert_eq!(roots_inside(&a, 1.0), largest < 1.0, "{a:?}: {largest}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 2000);
    }

    #[test]
    fn schur_cohn_handles_the_radius_and_nan() {
        // z - 1.0 has its root on the unit circle: outside at radius 1, inside just beyond.
        assert!(!roots_inside(&[-1.0], 1.0));
        assert!(roots_inside(&[-1.0], 1.0 + 1e-10));
        assert!(!roots_inside(&[f64::NAN], 2.0));
    }
}

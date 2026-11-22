//! The quantile function of the standard normal distribution.

use crate::math;

/// The standard normal quantile `z` with `P(Z <= z) = p`, the inverse of the normal CDF.
///
/// Wichura's algorithm AS241 `PPND16` (Applied Statistics 37, 1988): rational
/// approximations accurate to about 1e-16 relative over `(0, 1)`. Returns `-inf`
/// at `p = 0`, `+inf` at `p = 1` and `NaN` outside `[0, 1]`.
///
/// # Examples
///
/// ```
/// use backshift::normal_quantile;
///
/// assert!((normal_quantile(0.975) - 1.959963984540054).abs() < 1e-14);
/// assert_eq!(normal_quantile(0.5), 0.0);
/// ```
#[must_use]
pub fn normal_quantile(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }

    if p == 0.0 {
        return f64::NEG_INFINITY;
    }

    if p == 1.0 {
        return f64::INFINITY;
    }

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        return q * polynomial(&CENTRAL_NUMERATOR, r) / polynomial(&CENTRAL_DENOMINATOR, r);
    }

    let tail = if q < 0.0 { p } else { 1.0 - p };
    let r = math::sqrt(-math::ln(tail));
    let magnitude = if r <= 5.0 {
        let r = r - 1.6;
        polynomial(&NEAR_NUMERATOR, r) / polynomial(&NEAR_DENOMINATOR, r)
    } else {
        let r = r - 5.0;
        polynomial(&FAR_NUMERATOR, r) / polynomial(&FAR_DENOMINATOR, r)
    };

    if q < 0.0 { -magnitude } else { magnitude }
}

/// Evaluates `c[0] + c[1] x + ... + c[7] x^7` by Horner's rule.
fn polynomial(coefficients: &[f64; 8], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |sum, &c| sum * x + c)
}

/// AS241 coefficients `a`, for `|p - 0.5| <= 0.425`. All tables are the published
/// values rounded to the nearest `f64`.
const CENTRAL_NUMERATOR: [f64; 8] = [
    3.3871328727963665,
    133.14166789178438,
    1971.5909503065513,
    13731.69376550946,
    45921.95393154987,
    67265.7709270087,
    33430.57558358813,
    2509.0809287301227,
];

/// AS241 coefficients `b`.
const CENTRAL_DENOMINATOR: [f64; 8] = [
    1.0,
    42.31333070160091,
    687.1870074920579,
    5394.196021424751,
    21213.794301586597,
    39307.89580009271,
    28729.085735721943,
    5226.495278852854,
];

/// AS241 coefficients `c`, for tails with `sqrt(-ln(tail)) <= 5`.
const NEAR_NUMERATOR: [f64; 8] = [
    1.4234371107496835,
    4.630337846156546,
    5.769497221460691,
    3.6478483247632045,
    1.2704582524523684,
    0.2417807251774506,
    0.022723844989269184,
    0.0007745450142783414,
];

/// AS241 coefficients `d`.
const NEAR_DENOMINATOR: [f64; 8] = [
    1.0,
    2.053191626637759,
    1.6763848301838038,
    0.6897673349851,
    0.14810397642748008,
    0.015198666563616457,
    0.0005475938084995345,
    1.0507500716444169e-09,
];

/// AS241 coefficients `e`, for the far tails.
const FAR_NUMERATOR: [f64; 8] = [
    6.657904643501103,
    5.463784911164114,
    1.7848265399172913,
    0.29656057182850487,
    0.026532189526576124,
    0.0012426609473880784,
    2.7115555687434876e-05,
    2.0103343992922881e-07,
];

/// AS241 coefficients `f`.
const FAR_DENOMINATOR: [f64; 8] = [
    1.0,
    0.599832206555888,
    0.1369298809227358,
    0.014875361290850615,
    0.0007868691311456133,
    1.8463183175100548e-05,
    1.421511758316446e-07,
    2.0442631033899397e-15,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// `scipy.stats.norm.ppf(p)` for a spread of `p`, from the far tails to the centre.
    const REFERENCE: [(f64, f64); 25] = [
        (1e-300, -37.0470962993612),
        (1e-100, -21.273453560965322),
        (1e-20, -9.262340089798409),
        (1e-10, -6.361340902404056),
        (1e-06, -4.753424308822899),
        (0.001, -3.090232306167813),
        (0.01, -2.3263478740408408),
        (0.025, -1.9599639845400545),
        (0.05, -1.6448536269514729),
        (0.1, -1.2815515655446004),
        (0.2, -0.8416212335729142),
        (0.3, -0.5244005127080409),
        (0.4, -0.2533471031357997),
        (0.45, -0.12566134685507402),
        (0.5, 0.0),
        (0.55, 0.12566134685507416),
        (0.6, 0.2533471031357997),
        (0.75, 0.6744897501960817),
        (0.9, 1.2815515655446004),
        (0.95, 1.6448536269514722),
        (0.975, 1.959963984540054),
        (0.99, 2.3263478740408408),
        (0.999, 3.090232306167813),
        (0.999999, 4.753424308817087),
        (0.9999999999, 6.361340889697422),
    ];

    #[test]
    fn matches_scipy() {
        for (p, expected) in REFERENCE {
            let actual = normal_quantile(p);
            let error = (actual - expected).abs() / expected.abs().max(1.0);
            assert!(error <= 1e-14, "p = {p}: got {actual}, scipy {expected}");
        }
    }

    #[test]
    fn edges() {
        assert_eq!(normal_quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_quantile(1.0), f64::INFINITY);
        assert!(normal_quantile(-0.1).is_nan());
        assert!(normal_quantile(1.5).is_nan());
        assert!(normal_quantile(f64::NAN).is_nan());
    }

    #[test]
    fn is_antisymmetric() {
        // Powers of two, so `1 - p` is exact.
        for p in [0.5_f64.powi(40), 0.5_f64.powi(10), 0.125, 0.25, 0.375] {
            assert_eq!(normal_quantile(p), -normal_quantile(1.0 - p));
        }
    }
}

//! Special functions behind the p-values of the diagnostic tests.

use crate::math;

/// The survival function of the χ² distribution with `k` degrees of freedom: `P(X > x)`.
///
/// This is the regularised upper incomplete gamma function `Q(k / 2, x / 2)`.
pub(crate) fn chi_squared_survival(x: f64, k: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }

    upper_regularized_gamma(k as f64 / 2.0, x / 2.0)
}

/// The regularised upper incomplete gamma function `Q(a, x) = Γ(a, x) / Γ(a)`, for `a > 0`, `x >= 0`.
///
/// Numerical Recipes 3e §6.2: the series for `x < a + 1`, Lentz's continued fraction
/// otherwise. Both run to full double precision.
pub(crate) fn upper_regularized_gamma(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        1.0 - lower_series(a, x)
    } else {
        upper_continued_fraction(a, x)
    }
}

/// The most terms or iterations the series and the continued fraction take.
const MAX_ITERATIONS: usize = 500;

/// The convergence threshold of both expansions.
const EPS: f64 = 1e-16;

/// `P(a, x)` by its power series.
fn lower_series(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut ap = a;
    for _ in 0..MAX_ITERATIONS {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * EPS {
            break;
        }
    }

    sum * math::exp(-x + a * math::ln(x) - ln_gamma(a))
}

/// `Q(a, x)` by its continued fraction (modified Lentz).
fn upper_continued_fraction(a: f64, x: f64) -> f64 {
    let tiny = f64::MIN_POSITIVE / EPS;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAX_ITERATIONS {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }

        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }

        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }

    math::exp(-x + a * math::ln(x) - ln_gamma(a)) * h
}

/// `ln Γ(x)` for `x >= 0.5`, by the Lanczos approximation (g = 7, 9 terms), accurate to
/// about 1e-15 relative. χ² shapes `k / 2` never go below 0.5, so no reflection is needed.
pub(crate) fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let x = x - 1.0;
    let sum = COEFFICIENTS[1..]
        .iter()
        .enumerate()
        .fold(COEFFICIENTS[0], |sum, (i, c)| {
            sum + c / (x + i as f64 + 1.0)
        });
    let t = x + G + 0.5;

    0.5 * math::ln(core::f64::consts::TAU) + (x + 0.5) * math::ln(t) - t + math::ln(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scipy.stats.chi2.sf(x, k)`.
    const CHI2: [(f64, usize, f64); 17] = [
        (0.1, 1, 0.7518296340458492),
        (0.5, 1, 0.47950012218695337),
        (1.0, 1, 0.31731050786291115),
        (3.84, 1, 0.05004352124870519),
        (10.0, 1, 0.001565402258002549),
        (0.5, 2, 0.7788007830714049),
        (5.0, 3, 0.1717971442967335),
        (12.69814, 10, 0.24104221511255408),
        (3.0, 10, 0.9814240637778593),
        (18.3, 10, 0.050109061411462506),
        (50.0, 10, 2.669083424904495e-07),
        (1.0, 20, 0.999999999829033),
        (30.0, 20, 0.06985366069940986),
        (80.0, 20, 3.925932226286184e-09),
        (100.0, 50, 3.454931382984871e-05),
        (200.0, 30, 4.9527335290032224e-27),
        (7.5, 5, 0.186029833602867),
    ];

    /// `scipy.special.gammaln(x)`.
    const LN_GAMMA: [(f64, f64); 8] = [
        (0.5, 0.5723649429247),
        (1.0, 0.0),
        (1.5, -0.12078223763524526),
        (2.5, 0.2846828704729192),
        (5.0, 3.1780538303479458),
        (10.5, 13.940625219403763),
        (25.0, 54.78472939811232),
        (100.0, 359.1342053695754),
    ];

    #[test]
    fn chi_squared_matches_scipy() {
        for (x, k, expected) in CHI2 {
            let actual = chi_squared_survival(x, k);
            let error = (actual - expected).abs() / expected.max(1e-300);
            assert!(
                error <= 1e-12,
                "x = {x}, k = {k}: got {actual}, scipy {expected}"
            );
        }
    }

    #[test]
    fn ln_gamma_matches_scipy() {
        for (x, expected) in LN_GAMMA {
            let actual = ln_gamma(x);
            assert!(
                (actual - expected).abs() <= 1e-13 * expected.abs().max(1.0),
                "x = {x}: got {actual}, scipy {expected}"
            );
        }
    }

    #[test]
    fn chi_squared_edges() {
        assert_eq!(chi_squared_survival(0.0, 3), 1.0);
        assert_eq!(chi_squared_survival(-1.0, 3), 1.0);
        // Two degrees of freedom: P(X > x) = exp(-x / 2).
        assert!((chi_squared_survival(3.0, 2) - (-1.5_f64).exp()).abs() < 1e-15);
    }
}

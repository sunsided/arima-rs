//! Diagnostic tests: residual whiteness, normality, and stationarity.
//!
//! [`ljung_box`] checks fitted residuals for leftover autocorrelation, [`jarque_bera`]
//! for normality, and [`kpss`] tests a series for stationarity; [`ndiffs`] uses KPSS
//! to choose a differencing order. All but `ndiffs` run without allocation, so a
//! device can check its own residuals. Each matches the statsmodels function of the
//! same name.

use crate::math;
use crate::special::chi_squared_survival;

/// A test statistic with its p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestResult {
    /// The test statistic.
    pub statistic: f64,
    /// The p-value: the probability of a statistic at least this extreme under the null hypothesis.
    pub p_value: f64,
}

/// The Ljung–Box test for autocorrelation in `residuals` up to `lags`.
///
/// `Q = n (n + 2) sum_(k=1..lags) r_k^2 / (n - k)`, compared with a χ² distribution with
/// `lags - model_df` degrees of freedom. Pass the number of estimated ARMA coefficients
/// (`p + q + P + Q`) as `model_df` when testing a fitted model's residuals. Matches
/// statsmodels `acorr_ljungbox(residuals, lags=[lags], model_df=model_df)`.
///
/// Returns [`None`] unless `1 <= lags < n` and `model_df < lags`, or if `residuals` is
/// constant or has non-finite values.
///
/// # Examples
///
/// ```
/// use backshift::ljung_box;
///
/// let alternating = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
/// let test = ljung_box(&alternating, 2, 0).unwrap();
/// assert!(test.p_value < 0.01); // very much not white noise
/// ```
#[must_use]
pub fn ljung_box(residuals: &[f64], lags: usize, model_df: usize) -> Option<TestResult> {
    let n = residuals.len();
    if lags == 0 || lags >= n || model_df >= lags || !all_finite(residuals) {
        return None;
    }

    let mean = mean(residuals);
    let c0: f64 = residuals.iter().map(|x| (x - mean) * (x - mean)).sum();
    if c0 == 0.0 {
        return None;
    }

    let sum: f64 = (1..=lags)
        .map(|k| {
            let ck: f64 = (0..n - k)
                .map(|t| (residuals[t] - mean) * (residuals[t + k] - mean))
                .sum();
            let r = ck / c0;
            r * r / (n - k) as f64
        })
        .sum();
    // In floating point: `n * (n + 2)` overflows a 32-bit `usize` from n = 65535.
    let nf = n as f64;
    let statistic = nf * (nf + 2.0) * sum;

    Some(TestResult {
        statistic,
        p_value: chi_squared_survival(statistic, lags - model_df),
    })
}

/// The Jarque–Bera test for normality of `residuals`.
///
/// `JB = n / 6 (S^2 + (K - 3)^2 / 4)` with the sample skewness `S` and kurtosis `K`
/// (biased moments), compared with a χ² distribution with two degrees of freedom.
/// Matches statsmodels `jarque_bera`.
///
/// Returns [`None`] for fewer than two values, a constant series or non-finite values.
#[must_use]
pub fn jarque_bera(residuals: &[f64]) -> Option<TestResult> {
    let n = residuals.len();
    if n < 2 || !all_finite(residuals) {
        return None;
    }

    let mean = mean(residuals);
    // `powi` is not in `core`, so the moments are built from products.
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for x in residuals {
        let deviation = x - mean;
        let squared = deviation * deviation;
        m2 += squared;
        m3 += squared * deviation;
        m4 += squared * squared;
    }

    let nf = n as f64;
    let (m2, m3, m4) = (m2 / nf, m3 / nf, m4 / nf);
    if m2 == 0.0 {
        return None;
    }

    let skewness = m3 / (m2 * math::sqrt(m2));
    let excess = m4 / (m2 * m2) - 3.0;
    let statistic = n as f64 / 6.0 * (skewness * skewness + excess * excess / 4.0);

    Some(TestResult {
        statistic,
        p_value: chi_squared_survival(statistic, 2),
    })
}

/// The null hypothesis of a [`kpss`] test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssNull {
    /// The series is stationary around a constant level (statsmodels `regression="c"`).
    Level,
    /// The series is stationary around a linear trend (statsmodels `regression="ct"`).
    Trend,
}

/// The KPSS test (Kwiatkowski, Phillips, Schmidt & Shin 1992) of stationarity.
///
/// A small p-value rejects stationarity. The long-run variance uses Bartlett weights
/// over `lags` lags; [`kpss_lags`] gives statsmodels' "legacy" choice. As in
/// statsmodels, the p-value is interpolated in the published table and therefore
/// clipped to `[0.01, 0.10]`. Matches statsmodels `kpss(x, regression, nlags=lags)`.
///
/// Returns [`None`] unless `lags < n` and `n >= 3`, or if `x` has non-finite values or
/// no variation around the null's fit.
#[must_use]
pub fn kpss(x: &[f64], null: KpssNull, lags: usize) -> Option<TestResult> {
    let n = x.len();
    if n < 3 || lags >= n || !all_finite(x) {
        return None;
    }

    let residual = kpss_residuals(x, null);
    let (mut partial, mut eta, mut variance) = (0.0, 0.0, 0.0);
    for t in 0..n {
        let e = residual(t);
        partial += e;
        eta += partial * partial;
        variance += e * e;
    }

    for i in 1..=lags {
        let weight = 1.0 - i as f64 / (lags as f64 + 1.0);
        let product: f64 = (i..n).map(|t| residual(t) * residual(t - i)).sum();
        variance += 2.0 * weight * product;
    }

    let nf = n as f64;
    let (eta, variance) = (eta / (nf * nf), variance / nf);
    if variance <= 0.0 {
        return None;
    }

    let statistic = eta / variance;
    let critical = match null {
        KpssNull::Level => LEVEL_CRITICAL,
        KpssNull::Trend => TREND_CRITICAL,
    };

    Some(TestResult {
        statistic,
        p_value: interpolate_p_value(statistic, critical),
    })
}

/// statsmodels' "legacy" KPSS lag choice, `ceil(12 (n / 100)^(1/4))`, capped at `n - 1`.
#[must_use]
pub fn kpss_lags(n: usize) -> usize {
    let lags = 12.0 * math::sqrt(math::sqrt(n as f64 / 100.0));
    // `lags` is positive and small, so the cast after rounding up is exact.
    let truncated = lags as usize;
    let rounded = truncated + usize::from(lags > truncated as f64);
    rounded.min(n.saturating_sub(1))
}

/// The short KPSS lag rule, `trunc(4 (n / 100)^(1/4))`, capped at `n - 1`.
///
/// Kwiatkowski et al. (1992) call it `l4`; R's `tseries::kpss.test` (`lshort = TRUE`) and
/// pmdarima use it. With fewer lags than [`kpss_lags`] the test keeps more power on
/// short series, which matters when it decides the differencing order.
#[must_use]
pub fn kpss_short_lags(n: usize) -> usize {
    // Truncation is the floor for this non-negative value.
    let lags = (4.0 * math::sqrt(math::sqrt(n as f64 / 100.0))) as usize;
    lags.min(n.saturating_sub(1))
}

/// The number of differences that make `x` level-stationary by KPSS, at most `max_d`.
///
/// Differences while [`kpss`] (level, with [`kpss_short_lags`]) rejects stationarity at
/// `alpha`, the approach of R's `forecast::ndiffs` and pmdarima. Use it to pick `d` before
/// fitting.
///
/// The KPSS table only covers `[0.01, 0.10]`, so `alpha` is clamped to that range, as R
/// does, and a test rejects when its statistic exceeds the critical value at `alpha`
/// (interpolated in the same table). Comparing clipped p-values instead would make the
/// ends of the range misbehave.
/// Non-finite values are dropped first; a series too short to test counts as
/// needing no more differences.
///
/// # Examples
///
/// ```
/// use backshift::ndiffs;
///
/// // A random walk: its first difference is white noise.
/// let mut y = vec![0.0];
/// let mut state = 2_u32;
/// for _ in 0..199 {
///     state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
///     y.push(y.last().unwrap() + f64::from(state >> 8) / f64::from(1_u32 << 24) - 0.5);
/// }
/// assert_eq!(ndiffs(&y, 0.05, 2), 1);
/// ```
#[cfg(feature = "alloc")]
#[must_use]
pub fn ndiffs(x: &[f64], alpha: f64, max_d: usize) -> usize {
    let alpha = alpha.clamp(0.01, 0.10);
    let mut series: alloc::vec::Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let critical = critical_value(alpha, LEVEL_CRITICAL);
    for d in 0..max_d {
        match kpss(&series, KpssNull::Level, kpss_short_lags(series.len())) {
            Some(test) if test.statistic > critical => {
                series = crate::difference(&series, 1);
            }
            _ => return d,
        }
    }

    max_d
}

/// The residual of `x` around the null hypothesis' fit, as a function of `t`.
fn kpss_residuals(x: &[f64], null: KpssNull) -> impl Fn(usize) -> f64 + '_ {
    let n = x.len() as f64;
    let level = mean(x);
    let slope = match null {
        KpssNull::Level => 0.0,
        KpssNull::Trend => {
            // Least squares on t = 1, ..., n.
            let t_mean = (n + 1.0) / 2.0;
            let (sxy, sxx) = x.iter().enumerate().fold((0.0, 0.0), |(sxy, sxx), (t, v)| {
                let dt = (t + 1) as f64 - t_mean;
                (sxy + dt * (v - level), sxx + dt * dt)
            });
            sxy / sxx
        }
    };
    let t_mean = (n + 1.0) / 2.0;

    move |t| x[t] - level - slope * ((t + 1) as f64 - t_mean)
}

/// The significance levels of the KPSS critical-value tables.
const P: [f64; 4] = [0.10, 0.05, 0.025, 0.01];

/// KPSS critical values for level stationarity at [`P`] (Kwiatkowski et al. 1992, table 1).
const LEVEL_CRITICAL: [f64; 4] = [0.347, 0.463, 0.574, 0.739];

/// KPSS critical values for trend stationarity at [`P`].
const TREND_CRITICAL: [f64; 4] = [0.119, 0.146, 0.176, 0.216];

/// The critical value at `alpha`, linear in the table between its levels; `alpha` must
/// lie in `[0.01, 0.10]`.
#[cfg(feature = "alloc")]
fn critical_value(alpha: f64, critical: [f64; 4]) -> f64 {
    for i in 1..4 {
        if alpha >= P[i] {
            let fraction = (P[i - 1] - alpha) / (P[i - 1] - P[i]);
            return critical[i - 1] + fraction * (critical[i] - critical[i - 1]);
        }
    }

    critical[3]
}

/// `np.interp(statistic, critical, [0.10, 0.05, 0.025, 0.01])`: linear in the table,
/// clipped at its ends.
fn interpolate_p_value(statistic: f64, critical: [f64; 4]) -> f64 {
    if statistic <= critical[0] {
        return P[0];
    }

    for i in 1..4 {
        if statistic <= critical[i] {
            let fraction = (statistic - critical[i - 1]) / (critical[i] - critical[i - 1]);
            return P[i - 1] + fraction * (P[i] - P[i - 1]);
        }
    }

    P[3]
}

/// The arithmetic mean.
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Whether every value is finite.
fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_input() {
        let x = [1.0, 2.0, 0.5, 1.5];
        assert!(ljung_box(&x, 0, 0).is_none());
        assert!(ljung_box(&x, 4, 0).is_none());
        assert!(ljung_box(&x, 2, 2).is_none());
        assert!(ljung_box(&[1.0; 5], 2, 0).is_none());
        assert!(jarque_bera(&[1.0]).is_none());
        assert!(jarque_bera(&[f64::NAN, 1.0]).is_none());
        assert!(kpss(&x, KpssNull::Level, 4).is_none());
    }

    #[test]
    fn p_value_table_is_clipped_and_linear() {
        let critical = [0.347, 0.463, 0.574, 0.739];
        assert_eq!(interpolate_p_value(0.1, critical), 0.10);
        assert_eq!(interpolate_p_value(2.0, critical), 0.01);
        assert!((interpolate_p_value(0.463, critical) - 0.05).abs() < 1e-15);
        assert!((interpolate_p_value(0.405, critical) - 0.075).abs() < 1e-12);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn critical_values_invert_the_table() {
        assert!((critical_value(0.10, LEVEL_CRITICAL) - 0.347).abs() < 1e-15);
        assert!((critical_value(0.05, LEVEL_CRITICAL) - 0.463).abs() < 1e-15);
        assert!((critical_value(0.01, LEVEL_CRITICAL) - 0.739).abs() < 1e-15);
        let midway = critical_value(0.075, LEVEL_CRITICAL);
        assert!((interpolate_p_value(midway, LEVEL_CRITICAL) - 0.075).abs() < 1e-12);
    }

    #[test]
    fn kpss_lags_follow_statsmodels() {
        // int(ceil(12 * (n / 100) ** 0.25)), capped at n - 1.
        assert_eq!(kpss_lags(100), 12);
        assert_eq!(kpss_lags(60), 11);
        assert_eq!(kpss_lags(5), 4);
        assert_eq!(kpss_lags(1000), 22);
    }

    #[test]
    fn kpss_short_lags_follow_pmdarima() {
        // int(np.trunc(4 * (n / 100) ** 0.25)).
        assert_eq!(kpss_short_lags(100), 4);
        assert_eq!(kpss_short_lags(150), 4);
        assert_eq!(kpss_short_lags(60), 3);
        assert_eq!(kpss_short_lags(1000), 7);
        assert_eq!(kpss_short_lags(2), 1);
    }

    /// A random walk from a fixed linear congruential generator, `n` values.
    #[cfg(feature = "alloc")]
    fn walk(n: usize) -> alloc::vec::Vec<f64> {
        let mut y = alloc::vec![0.0];
        let mut state = 2_u32;
        for _ in 1..n {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            y.push(y[y.len() - 1] + f64::from(state >> 8) / f64::from(1_u32 << 24) - 0.5);
        }

        y
    }

    /// The running sum of `x`.
    #[cfg(feature = "alloc")]
    fn cumulative(x: &[f64]) -> alloc::vec::Vec<f64> {
        x.iter()
            .scan(0.0, |sum, v| {
                *sum += v;
                Some(*sum)
            })
            .collect()
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn ndiffs_counts_integration_orders() {
        // Checked against statsmodels' KPSS with the same lags.
        let y = walk(200);
        assert_eq!(ndiffs(&crate::difference(&y, 1), 0.05, 2), 0);
        assert_eq!(ndiffs(&y, 0.05, 2), 1);
        assert_eq!(ndiffs(&cumulative(&y), 0.05, 2), 2);
        let twice = cumulative(&cumulative(&y));
        assert_eq!(ndiffs(&twice, 0.05, 2), 2);
        assert_eq!(ndiffs(&twice, 0.05, 3), 3);
    }

    #[test]
    fn ljung_box_handles_more_values_than_a_32_bit_product() {
        // Alternating signs: r_1 = -(n - 1) / n, so Q = (n + 2)(n - 1) / n.
        let n = 70_000;
        let x: std::vec::Vec<f64> = (0..n)
            .map(|t| if t % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let test = ljung_box(&x, 1, 0).unwrap();
        let nf = f64::from(n);
        let expected = (nf + 2.0) * (nf - 1.0) / nf;
        assert!(
            (test.statistic - expected).abs() < 1e-6 * expected,
            "{test:?}"
        );
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn ndiffs_edges() {
        let y = walk(200);
        // alpha at the bottom of the table still rejects a clear random walk.
        assert_eq!(ndiffs(&y, 0.01, 2), 1);
        // alpha above the table is clamped, so white noise is not differenced.
        assert_eq!(ndiffs(&crate::difference(&y, 1), 0.5, 2), 0);

        let mut gappy = y.clone();
        gappy[50] = f64::NAN;
        assert_eq!(ndiffs(&gappy, 0.05, 2), 1);
        assert_eq!(ndiffs(&[1.0, 2.0], 0.05, 2), 0);
    }
}

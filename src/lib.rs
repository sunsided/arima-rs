//! ARIMA modelling that also runs on devices without a heap.
//!
//! Fit a model with `Arima::fit` (feature `alloc`), then stream new values through a [`Filter`],
//! which lives inline with no allocation, and forecast with prediction intervals.
//! Likelihoods, innovations, forecasts and intervals match statsmodels to 1e-9.
//!
//! ```
//! # #[cfg(feature = "alloc")]
//! # fn main() {
//! use backshift::{Arima, Filter};
//!
//! let history = [0.3, 1.1, 0.4, -0.2, 0.9, 1.4, 0.6, 0.1, -0.5, 0.2, 0.8, 1.0, 0.3, -0.1, 0.4];
//! let fitted = Arima::new(1, 0, 0).fit(&history).unwrap();
//! println!("AIC {:.2}", fitted.aic());
//!
//! // The device side needs only the coefficients.
//! let mut filter = Filter::<4>::new(&fitted.spec()).unwrap();
//! filter.warm_up(&history);
//! let innovation = filter.update(0.7).unwrap();
//! assert!(innovation.variance > 0.0);
//!
//! let (mut mean, mut lower, mut upper) = ([0.0; 3], [0.0; 3], [0.0; 3]);
//! filter.forecast_interval_into(0.95, &mut mean, &mut lower, &mut upper);
//! assert!(lower[0] < mean[0] && mean[0] < upper[0]);
//! # }
//! # #[cfg(not(feature = "alloc"))]
//! # fn main() {}
//! ```
//!
//! The crate also has the building blocks: autocorrelation ([`acorr`], [`acov`]) and,
//! with `alloc`, differencing and the Yule–Walker, Burg and Hannan–Rissanen estimators.
//!
//! The crate is `no_std`. Its features:
//!
//! - `std` (default): floating-point functions from the standard library. Implies `alloc`.
//! - `libm`: floating-point functions from the pure-Rust `libm` crate, for targets without `std`.
//! - `alloc`: the functions that return a `Vec`, such as `pacf`, `difference`,
//!   `yule_walker` and `burg`. Without it the crate never touches a heap, so firmware
//!   needs no global allocator.
//!
//! Enable `std` or `libm`. A firmware build is `default-features = false, features = ["libm"]`,
//! plus `"alloc"` if it has an allocator.

#![no_std]
#![forbid(unsafe_code)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("enable the `std` feature or, for `no_std` targets, the `libm` feature");

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

#[cfg(feature = "alloc")]
pub use crate::autoregressive::{ArCoefficients, burg, yule_walker};
#[cfg(feature = "alloc")]
pub use crate::diagnostics::ndiffs;
pub use crate::diagnostics::{
    KpssNull, TestResult, jarque_bera, kpss, kpss_lags, kpss_short_lags, ljung_box,
};
#[cfg(feature = "alloc")]
pub use crate::difference::{difference, integrate};
pub use crate::filter::{ArimaSpec, Filter, FilterError, Seasonal};
#[cfg(feature = "alloc")]
pub use crate::fit::{
    Arima, ArmaCoefficients, AutoArima, AutoFit, Criterion, Fitted, FourierFit, hannan_rissanen,
    nsdiffs,
};
pub use crate::fit::{FitError, Refit, RefitOptions, Refitter};
#[cfg(feature = "alloc")]
pub use crate::fourier::fourier;
pub use crate::fourier::{FourierError, fourier_columns, fourier_row};
#[cfg(feature = "alloc")]
pub use crate::likelihood::{innovations, innovations_exog, loglike, loglike_exog};
pub use crate::normal::normal_quantile;
pub use crate::state_space::Innovation;
#[cfg(feature = "alloc")]
pub use crate::stl::{Decomposition, stl};
pub use crate::time_series::{TimeSeries, TimeSeriesError};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::borrow::Borrow;

#[cfg(feature = "alloc")]
mod autoregressive;
mod buffer;
mod diagnostics;
#[cfg(feature = "alloc")]
mod difference;
mod filter;
mod fit;
mod fourier;
mod likelihood;
mod math;
mod normal;
mod polynomial;
mod special;
mod state_space;
#[cfg(feature = "alloc")]
mod stl;
mod time_series;

/// Compiles the README's code examples as doctests, so they cannot drift.
#[cfg(all(doctest, feature = "alloc"))]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

/// Defines the estimator to use.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Estimator {
    /// Divides by `n`: the statistics of the data itself.
    Biased,
    /// Divides by `n - 1`: the sample estimate of the population statistics.
    Unbiased,
}

/// Splits a series into its first `len - lag` and last `len - lag` values.
///
/// Returns [`None`] if `lag` is not less than the number of elements.
fn lagged<'a>(values: &TimeSeries<'a>, lag: usize) -> Option<(TimeSeries<'a>, TimeSeries<'a>)> {
    if lag >= values.len() {
        return None;
    }

    let length = values.len() - lag;
    let x = values.slice_stationary(..length).ok()?;
    let y = values.slice_stationary(lag..).ok()?;
    Some((x, y))
}

/// Sums the products of the residuals of two equally long series.
fn cross_sum(x: &TimeSeries, y: &TimeSeries) -> f64 {
    x.residuals().zip(y.residuals()).map(|(x, y)| x * y).sum()
}

/// Calculates the autocovariance with a given lag.
///
/// Matches statsmodels `acovf`: [`Estimator::Unbiased`] divides by `n - lag`
/// (`adjusted=True`), [`Estimator::Biased`] divides by `n`. With `Biased`,
/// `acov(x, k) / acov(x, 0) == acorr(x, k)`.
///
/// ## Arguments
/// * `values` - The values to calculate the autocovariance for.
/// * `lag` - The lag of the data. The lag must be less than the number of elements.
///
/// ## Returns
/// [`None`] if the lag is too large, or some value representing the autocovariance.
pub fn acov<'a, T: Borrow<TimeSeries<'a>>>(values: T, lag: usize) -> Option<f64> {
    let values = values.borrow();
    let (x, y) = lagged(values, lag)?;
    let divisor = match values.estimator() {
        Estimator::Unbiased => x.len(),
        Estimator::Biased => values.len(),
    };
    Some(cross_sum(&x, &y) / divisor as f64)
}

/// Calculates the covariance.
///
/// ## Arguments
/// * `x` - The first signal.
/// * `y` - The second signal.
///
/// ## Returns
/// [`None`] if the signals differ in length or estimator or, with [`Estimator::Unbiased`],
/// hold fewer than two values; otherwise some value representing the covariance.
pub fn cov<'x, 'y, X: Borrow<TimeSeries<'x>>, Y: Borrow<TimeSeries<'y>>>(
    x: X,
    y: Y,
) -> Option<f64> {
    let x = x.borrow();
    let y = y.borrow();
    if x.len() != y.len() || x.estimator() != y.estimator() {
        return None;
    }

    let divisor = match x.estimator() {
        Estimator::Unbiased => x.len() - 1,
        Estimator::Biased => x.len(),
    };
    if divisor == 0 {
        return None;
    }

    Some(cross_sum(x, y) / divisor as f64)
}

/// Calculates the autocorrelation coefficient with a given lag.
///
/// Uses the standard estimate `sum((x_t - mean) * (x_t+lag - mean)) / sum((x_t - mean)^2)`,
/// which always lies in `[-1, 1]`. The estimator of the series is not used.
///
/// ## Arguments
/// * `values` - The values to calculate the autocorrelation for. Assumed to be a stationary time series.
/// * `lag` - The lag of the data. The lag must be less than the number of elements.
///
/// ## Returns
/// [`None`] if the lag is too large or the series is constant,
/// or some value representing the autocorrelation.
pub fn acorr<'a, T: Borrow<TimeSeries<'a>>>(values: T, lag: usize) -> Option<f64> {
    let values = values.borrow();
    let (x, y) = lagged(values, lag)?;
    let tss = cross_sum(values, values);
    if !tss.is_finite() || tss == 0.0 {
        return None;
    }
    Some(cross_sum(&x, &y) / tss)
}

/// Calculates the correlation coefficient.
///
/// ## Arguments
/// * `x` - The first signal.
/// * `y` - The second signal.
///
/// ## Returns
/// [`None`] if the covariance is undefined or either signal is constant,
/// or some value representing the correlation coefficient.
pub fn corr<'x, 'y, X: Borrow<TimeSeries<'x>>, Y: Borrow<TimeSeries<'y>>>(
    x: X,
    y: Y,
) -> Option<f64> {
    let x = x.borrow();
    let y = y.borrow();
    let std = x.std() * y.std();
    cov(x, y)
        .filter(|_| std.is_finite() && std != 0.0)
        .map(|phi| phi / std)
}

/// Calculates the partial autocorrelation for lags `0..=max_lags`.
///
/// `max_lags` is capped at `len - 1`.
///
/// ## Returns
/// [`None`] if no lag can be computed (zero lags, a single value or a constant series),
/// or the partial autocorrelations, starting with `1.0` for lag 0.
#[cfg(feature = "alloc")]
pub fn pacf<'x, X: Borrow<TimeSeries<'x>>>(values: X, max_lags: usize) -> Option<Vec<f64>> {
    let values = values.borrow();
    let max_lags = max_lags.min(values.len() - 1);
    if max_lags == 0 {
        return None;
    }

    let acf = (0..=max_lags)
        .map(|lag| acorr(values, lag))
        .collect::<Option<Vec<_>>>()?;
    Some(autoregressive::durbin_levinson(&acf).pacf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_acov() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let values = TimeSeries::try_new(values, Estimator::Unbiased).unwrap();
        let lag: usize = 2;

        let result = acov(values, lag).unwrap();

        let expected: f64 = -4.0 / 3.0; // statsmodels acovf(adjusted=True)
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_acov_2() {
        let values: &[f64] = &[
            0.05, 0.01, -0.02, 0.03, -0.04, 0.06, 0.02, -0.01, -0.03, 0.04,
        ];
        let values = TimeSeries::try_new(values, Estimator::Unbiased).unwrap();
        let lag: usize = 1;

        let result = acov(values, lag).unwrap();

        let expected: f64 = -0.00046011111111111095; // statsmodels acovf(adjusted=True)
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_cov() {
        let values = TimeSeries::try_new(&[2.0, 4.0, 6.0, 8.0, 10.0], Estimator::Unbiased).unwrap();

        let x = values
            .slice_stationary(..3)
            .expect("time series has enough data");
        let y = values
            .slice_stationary(2..)
            .expect("time series has enough data");

        let result = cov(x, y).unwrap();

        assert_approx_eq!(f64, result, -2.0);

        // Because we sliced the time series as stationary, these values now exist.
        assert_approx_eq!(f64, values.mean(), 6.0);
        assert_approx_eq!(f64, values.var(), 10.0);
        assert_approx_eq!(f64, values.std(), 3.1622776601683795);
    }

    #[test]
    fn test_acorr() {
        let values: &[f64] = &[
            0.05, 0.01, -0.02, 0.03, -0.04, 0.06, 0.02, -0.01, -0.03, 0.04,
        ];
        let values = TimeSeries::try_new(values, Estimator::Unbiased).unwrap();
        let lag: usize = 1;

        let result = acorr(values, lag).unwrap();

        let expected: f64 = -0.380257116620753; // statsmodels acf
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn test_pacf() {
        let values = signal_gen(1000);
        let values = TimeSeries::try_new(&values, Estimator::Unbiased).unwrap();
        let max_lags = 10;

        let result = pacf(values, max_lags).unwrap();

        // statsmodels.tsa.stattools.pacf(y, nlags=10, method="ldb")
        let expected = [
            1.0,
            0.9992026377192385,
            -0.9886934907501577,
            0.6421864105266735,
            0.3483789857330886,
            0.18317292590771958,
            0.06350364686564525,
            -0.024538003187321095,
            -0.08462920394055298,
            -0.12177379316633868,
            -0.14225955547380964,
        ];
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected) {
            assert_approx_eq!(f64, *r, e, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_acorr_is_bounded() {
        // statsmodels acf
        let expected = [1.0, 0.4, -0.1, -0.4, -0.4];
        for estimator in [Estimator::Biased, Estimator::Unbiased] {
            let values = TimeSeries::try_new(&[2.0, 4.0, 6.0, 8.0, 10.0], estimator).unwrap();
            for (lag, e) in expected.iter().enumerate() {
                assert_approx_eq!(f64, acorr(&values, lag).unwrap(), *e, epsilon = 1e-12);
            }
            assert_eq!(acorr(&values, 5), None);
        }
    }

    #[test]
    fn test_acov_biased_matches_acorr() {
        let values = TimeSeries::try_new(&[2.0, 4.0, 6.0, 8.0, 10.0], Estimator::Biased).unwrap();
        // statsmodels acovf(adjusted=False)
        for (lag, e) in [8.0, 3.2, -0.8].iter().enumerate() {
            assert_approx_eq!(f64, acov(&values, lag).unwrap(), *e, epsilon = 1e-12);
        }
        let c0 = acov(&values, 0).unwrap();
        for lag in 0..5 {
            let ratio = acov(&values, lag).unwrap() / c0;
            assert_approx_eq!(f64, ratio, acorr(&values, lag).unwrap(), epsilon = 1e-12);
        }
        assert!(acov(&values, 4).is_some());
    }

    #[test]
    fn test_non_finite_returns_none() {
        let values = TimeSeries::try_new(&[1.0, f64::NAN, 3.0], Estimator::Unbiased).unwrap();
        assert_eq!(acorr(&values, 1), None);
        assert_eq!(corr(&values, &values), None);
    }

    #[test]
    fn test_edge_cases_return_none() {
        let short = TimeSeries::try_new(&[1.0, 2.0], Estimator::Unbiased).unwrap();
        assert_eq!(acov(&short, 2), None);

        let constant = TimeSeries::try_new(&[3.0; 5], Estimator::Unbiased).unwrap();
        assert_eq!(acorr(&constant, 1), None);
        assert_eq!(corr(&constant, &constant), None);

        let values = TimeSeries::try_new(&[2.0, 4.0, 6.0, 8.0, 10.0], Estimator::Unbiased).unwrap();
        assert_eq!(cov(&short, &values), None);

        let biased = TimeSeries::try_new(&[2.0, 4.0, 6.0, 8.0, 10.0], Estimator::Biased).unwrap();
        assert_eq!(cov(&values, &biased), None);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn test_pacf_edge_cases() {
        let constant = TimeSeries::try_new(&[3.0; 5], Estimator::Unbiased).unwrap();
        assert_eq!(pacf(&constant, 3), None);

        let values = TimeSeries::try_new(&[2.0, 4.0, 6.0, 8.0, 10.0], Estimator::Unbiased).unwrap();
        assert_eq!(pacf(&values, 0), None);

        let result = pacf(&values, usize::MAX).unwrap();
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|r| r.is_finite()));
    }

    #[cfg(feature = "alloc")]
    fn signal_gen(len: usize) -> Vec<f64> {
        let f_base = 2.5;
        let a_base = 4.0;
        let f_overlay = 16.0;
        let a_overlay = 0.5;

        let scale = 1.0 / (len - 1) as f64;
        let t = (0..len)
            .map(|t| 2.0_f64 * t as f64 * scale)
            .collect::<Vec<f64>>();
        let y: Vec<_> = t
            .iter()
            .map(|t| {
                a_base * (2.0 * std::f64::consts::PI * f_base * t).sin()
                    + a_overlay * (2.0 * std::f64::consts::PI * f_overlay * t).sin()
            })
            .collect();
        y
    }
}

//! Fitting the Theta method as R's `forecast::thetaf`: a seasonality test, a classical
//! multiplicative decomposition where it passes, simple exponential smoothing and a drift of
//! half the least-squares slope.

use super::FitError;
use super::decompose::classical_figure;
use super::ets::Ets;
use crate::ets::Form;
use crate::normal::normal_quantile;
use crate::theta::ThetaSpec;
use crate::{Estimator, TimeSeries, acorr, math};
use alloc::vec::Vec;

/// The Theta method to fit, optionally on a seasonal series.
///
/// # Examples
///
/// ```
/// use backshift::{Theta, ThetaFilter};
///
/// let y: Vec<f64> = (0..60).map(|t| 50.0 + 0.4 * t as f64 + (t as f64 * 1.3).sin()).collect();
/// let fit = Theta::new().fit(&y).unwrap();
/// let filter = ThetaFilter::<1>::new(&fit.spec()).unwrap();
/// let (mut mean, mut variance) = ([0.0; 3], [0.0; 3]);
/// filter.forecast_into(&mut mean, &mut variance);
/// assert!(mean[2] > mean[0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Theta {
    /// The season length; 0 or 1 for none.
    period: usize,
}

impl Theta {
    /// The Theta method without a season.
    #[must_use]
    pub const fn new() -> Self {
        Self { period: 0 }
    }

    /// Tests for a season of length `period` and, where R's test finds one, deseasonalises
    /// with it.
    #[must_use]
    pub const fn seasonal(self, period: usize) -> Self {
        Self { period }
    }

    /// Fits the method to `y`.
    ///
    /// # Errors
    ///
    /// As [`Ets::fit`] for the simple exponential smoothing: [`FitError::Unsupported`] for
    /// non-finite values, [`FitError::TooShort`] for fewer than 7 values,
    /// [`FitError::Degenerate`] for a series the smoothing fits exactly.
    pub fn fit(&self, y: &[f64]) -> Result<ThetaFit, FitError> {
        if !math::all_finite(y) {
            return Err(FitError::Unsupported);
        }

        let m = self.period;
        // R divides by the indices whatever they are; one near zero (a season whose mean is
        // near zero) would blow the adjusted series up, so such a season counts as none.
        let figure = (m > 1 && seasonal(y, m))
            .then(|| classical_figure(y, m, Form::Multiplicative))
            .filter(|figure| figure.iter().all(|s| s.abs() >= MIN_INDEX));
        let adjusted: Vec<f64> = match &figure {
            Some(figure) => y
                .iter()
                .enumerate()
                .map(|(t, v)| v / figure[t % m])
                .collect(),
            None => y.to_vec(),
        };

        let smoothing = Ets::new().fit(&adjusted)?;
        let n = y.len();
        let factors = figure.map_or_else(Vec::new, |figure| {
            (0..m).map(|j| figure[(n + j) % m]).collect()
        });
        Ok(ThetaFit {
            // Positive, as `ThetaFilter` needs: ETS bounds it below by R's 1e-4.
            alpha: smoothing.alpha,
            drift: slope(&adjusted) / 2.0,
            level: smoothing.level,
            count: n,
            sigma2: smoothing.sigma2,
            factors,
        })
    }
}

/// The smallest seasonal index, in absolute value, the fit divides by; see
/// `docs/CONFORMANCE.md`.
const MIN_INDEX: f64 = 1e-4;

/// R's seasonality test in `thetaf`: the lag-`m` autocorrelation exceeds the 95% one-sided
/// normal quantile times its standard error `sqrt((1 + 2 sum_(i < m) r_i^2) / n)`, for a
/// series of more than two seasons that is not constant.
fn seasonal(y: &[f64], m: usize) -> bool {
    let n = y.len();
    if n <= 2 * m || math::is_constant(y) {
        return false;
    }

    let Ok(series) = TimeSeries::try_new(y, Estimator::Biased) else {
        return false;
    };
    let r: Vec<f64> = (1..=m).filter_map(|lag| acorr(&series, lag)).collect();
    if r.len() < m {
        return false;
    }

    let spread: f64 = r[..m - 1].iter().map(|v| v * v).sum();
    let stat = math::sqrt((1.0 + 2.0 * spread) / count_as_f64(n));
    r[m - 1].abs() / stat > normal_quantile(0.95)
}

/// The least-squares slope of `values` against `t = 0, 1, ..., n - 1`.
fn slope(values: &[f64]) -> f64 {
    let n = count_as_f64(values.len());
    let t_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;
    let (sxy, sxx) = values
        .iter()
        .enumerate()
        .fold((0.0, 0.0), |(sxy, sxx), (t, v)| {
            let dt = count_as_f64(t) - t_mean;
            (sxy + dt * (v - y_mean), sxx + dt * dt)
        });
    sxy / sxx
}

/// A count as a float. Counts here are series lengths, far below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "series lengths stay far below 2^53, where the conversion is exact"
)]
fn count_as_f64(count: usize) -> f64 {
    count as f64
}

/// A fitted Theta method.
#[derive(Debug, Clone, PartialEq)]
pub struct ThetaFit {
    /// The smoothing parameter of the level.
    pub alpha: f64,
    /// The drift, half the least-squares slope of the (deseasonalised) series.
    pub drift: f64,
    /// The smoothed level after the last value.
    pub level: f64,
    /// The number of values.
    pub count: usize,
    /// The innovation variance of the smoothing.
    pub sigma2: f64,
    /// The seasonal indices of the next `m` values, next first; empty where R's test found
    /// no season.
    pub factors: Vec<f64>,
}

impl ThetaFit {
    /// The fitted method, for a [`ThetaFilter`](crate::ThetaFilter) that forecasts and
    /// continues from the last value.
    #[must_use]
    pub fn spec(&self) -> ThetaSpec<'_> {
        ThetaSpec {
            alpha: self.alpha,
            drift: self.drift,
            level: self.level,
            count: self.count,
            sigma2: self.sigma2,
            seasonal: &self.factors,
        }
    }
}

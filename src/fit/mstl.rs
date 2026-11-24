//! Forecasting with MSTL plus ARIMA, as R's `forecast::stlm`/`stlf` and statsforecast's
//! `MSTL` model do.
//!
//! MSTL removes the seasonal components. A non-seasonal [`AutoArima`] search models the
//! seasonally adjusted series. A forecast adds back each seasonal component's last
//! cycle, repeated (seasonal naive).
//!
//! ```mermaid
//! flowchart LR
//!     Y["y"] -->|mstl| A["adjusted"]
//!     Y -->|mstl| S["seasonal components"]
//!     A -->|"AutoArima, D = 0"| F["ARIMA forecast"]
//!     S -->|"repeat last cycle"| N["seasonal naive"]
//!     F --> P["forecast = F + N"]
//!     N --> P
//! ```

use super::{AutoArima, AutoFit, FitError};
use crate::filter::{Filter, FilterError};
use crate::stl::{MSTL_ITERATIONS, MultiDecomposition, mstl};
use alloc::vec;
use alloc::vec::Vec;

/// A series decomposed by MSTL, with an ARIMA model of its seasonally adjusted part.
#[derive(Debug, Clone, PartialEq)]
pub struct MstlArima {
    /// The MSTL decomposition of the series.
    pub decomposition: MultiDecomposition,
    /// The selected model of the seasonally adjusted series.
    pub fit: AutoFit,
    /// The seasonally adjusted series the model was fitted to.
    adjusted: Vec<f64>,
}

/// Forecasts with central prediction intervals, one entry per step ahead.
#[derive(Debug, Clone, PartialEq)]
pub struct MstlForecast {
    /// The forecast means.
    pub mean: Vec<f64>,
    /// The lower interval bounds.
    pub lower: Vec<f64>,
    /// The upper interval bounds.
    pub upper: Vec<f64>,
}

impl AutoArima {
    /// Decomposes `y` with [`mstl`] (statsmodels' defaults, [`MSTL_ITERATIONS`] passes) and
    /// fits a non-seasonal ARIMA (`D = 0`, this search's other settings) to the seasonally
    /// adjusted series.
    ///
    /// Forecast with [`MstlArima::forecast`]. This suits several or long seasonal periods,
    /// which a seasonal ARIMA cannot hold.
    ///
    /// # Errors
    ///
    /// - [`FitError::Decomposition`] if MSTL cannot decompose `y` (see [`mstl`]); a
    ///   missing value is one such case.
    /// - As for [`AutoArima::fit`] on the adjusted series.
    ///
    /// # Examples
    ///
    /// ```
    /// use backshift::AutoArima;
    ///
    /// let y: Vec<f64> = (0..140)
    ///     .map(|t| [1.0, -1.0, 0.5][t % 3] + [2.0, 0.0, -1.0, 0.5, -1.5, 0.0, 0.0][t % 7]
    ///         + 0.01 * ((t * 37) % 11) as f64)
    ///     .collect();
    /// let model = AutoArima::new().max_orders(1, 1, 0, 0).fit_mstl(&y, &[3, 7]).unwrap();
    /// let forecast = model.forecast::<4>(0.95, 7).unwrap();
    /// assert!(forecast.lower[0] < forecast.mean[0] && forecast.mean[0] < forecast.upper[0]);
    /// ```
    pub fn fit_mstl(&self, y: &[f64], periods: &[usize]) -> Result<MstlArima, FitError> {
        let decomposition = mstl(y, periods, MSTL_ITERATIONS).ok_or(FitError::Decomposition)?;
        let adjusted = decomposition.seasonally_adjusted();
        let fit = self.non_seasonal().fit(&adjusted)?;

        Ok(MstlArima {
            decomposition,
            fit,
            adjusted,
        })
    }
}

impl MstlArima {
    /// Forecasts the next `horizon` values with central prediction intervals at
    /// probability `level`.
    ///
    /// The mean is the ARIMA forecast of the adjusted series plus each seasonal
    /// component's last cycle, repeated. The intervals are those of the ARIMA part alone:
    /// they ignore the uncertainty of the seasonal components, as R's `stlf` and
    /// statsforecast do, so they are too narrow when the seasonal pattern changes.
    ///
    /// `R` is the capacity of the [`Filter`] that forecasts the adjusted series; it must
    /// hold the fitted model's [`ArimaSpec::states`](crate::ArimaSpec::states), at most
    /// `max(p, q + 1) + d`.
    ///
    /// # Errors
    ///
    /// [`FilterError::Capacity`] if `R` is too small for the fitted model.
    pub fn forecast<const R: usize>(
        &self,
        level: f64,
        horizon: usize,
    ) -> Result<MstlForecast, FilterError> {
        let mut filter = Filter::<R>::new(&self.fit.fitted.spec())?;
        filter.warm_up(&self.adjusted);
        let (mut mean, mut lower, mut upper) =
            (vec![0.0; horizon], vec![0.0; horizon], vec![0.0; horizon]);
        filter.forecast_interval_into(level, &mut mean, &mut lower, &mut upper);

        for (h, seasonal) in self.seasonal_naive(horizon).into_iter().enumerate() {
            mean[h] += seasonal;
            lower[h] += seasonal;
            upper[h] += seasonal;
        }

        Ok(MstlForecast { mean, lower, upper })
    }

    /// The sum of the seasonal components' last cycles, repeated over `horizon` steps.
    fn seasonal_naive(&self, horizon: usize) -> Vec<f64> {
        let parts = &self.decomposition;
        let n = parts.trend.len();
        (0..horizon)
            .map(|h| {
                parts
                    .periods
                    .iter()
                    .zip(&parts.seasonal)
                    .map(|(&period, component)| component[n - period + h % period])
                    .sum()
            })
            .collect()
    }
}

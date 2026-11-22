//! JavaScript bindings for `backshift`: fit, forecast with intervals, stream and refit
//! seasonal ARIMA models through WebAssembly.
//!
//! Every export validates its input and returns a `JsError` instead of panicking, since
//! a panic would kill the whole wasm instance. Series cross the boundary as
//! `Float64Array`; `NaN` marks a missing value.
//!
//! ```mermaid
//! flowchart LR
//!     V["Float64Array"] -->|"fit / autoArima"| M["Model"]
//!     M -->|forecast| F["Forecast"]
//!     M -->|"new Stream"| S["Stream"]
//!     S -->|update| I["innovation"]
//!     M -->|refit| M
//! ```

use backshift::{AutoArima, Filter, Fitted, RefitOptions};
use wasm_bindgen::prelude::*;

/// The state capacity of every filter: enough for the airline model at period 24
/// (`max(p + P s, q + Q s + 1) + d + D s <= 64`). Larger models get an error.
const CAPACITY: usize = 64;

/// The longest series any export accepts.
const MAX_VALUES: usize = 100_000;

/// The longest forecast horizon.
const MAX_HORIZON: usize = 10_000;

/// Upper bounds on the orders `p, q` and `P, Q`, which keep a fit's cost bounded.
const MAX_ORDER: usize = 8;

/// Upper bounds on `d` and `D`.
const MAX_DIFFERENCING: usize = 2;

/// The longest season.
const MAX_PERIOD: usize = 366;

/// The streaming filter the bindings use.
type DeviceFilter = Filter<CAPACITY>;

/// A fitted seasonal ARIMA(p, d, q)(P, D, Q, s) model.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Model {
    /// The fit, with its coefficients and statistics.
    fitted: Fitted,
}

#[wasm_bindgen]
impl Model {
    /// Fits ARIMA(p, d, q)(P, D, Q, s) by exact maximum likelihood.
    ///
    /// `mean` estimates the mean of the differenced series (the drift for `d = 1`); left
    /// out, it is estimated for `d + D = 0` only, as in statsmodels and R. Pass
    /// `P = D = Q = 0` and `period = 0` for a non-seasonal model.
    ///
    /// # Errors
    ///
    /// When the input is too long, an order is out of range, a seasonal part has no
    /// period, or the series is too short or degenerate for the model.
    #[expect(
        clippy::too_many_arguments,
        reason = "JavaScript has no builder here; the orders are plain numbers"
    )]
    pub fn fit(
        values: &[f64],
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        period: usize,
        mean: Option<bool>,
    ) -> Result<Model, JsError> {
        check_values(values)?;
        check_orders(p, d, q)?;
        check_orders(seasonal_p, seasonal_d, seasonal_q)?;
        let seasonal = seasonal_p + seasonal_d + seasonal_q > 0;
        if seasonal && !(2..=MAX_PERIOD).contains(&period) {
            return Err(JsError::new(&format!(
                "a seasonal part needs a period between 2 and {MAX_PERIOD}"
            )));
        }

        let model = backshift::Arima::new(p, d, q);
        let model = if seasonal {
            model.seasonal(seasonal_p, seasonal_d, seasonal_q, period)
        } else {
            model
        };
        let model = match mean {
            Some(include) => model.mean(include),
            None => model,
        };
        let fitted = model.fit(values).map_err(error)?;

        Ok(Model { fitted })
    }

    /// Selects and fits a model with the stepwise search of R's `auto.arima`; `period`
    /// is the season length, or 0 (or 1) for a non-seasonal search.
    ///
    /// # Errors
    ///
    /// When the input is too long, the period is out of range, or no candidate fits.
    #[wasm_bindgen(js_name = autoArima)]
    pub fn auto_arima(values: &[f64], period: usize) -> Result<Model, JsError> {
        check_values(values)?;
        if period > MAX_PERIOD {
            return Err(JsError::new(&format!(
                "the period must be at most {MAX_PERIOD}"
            )));
        }

        let search = if period > 1 {
            AutoArima::new().seasonal(period)
        } else {
            AutoArima::new()
        };
        let best = search.fit(values).map_err(error)?;

        Ok(Model {
            fitted: best.fitted,
        })
    }

    /// The model in R's notation, for example `ARIMA(0,1,1)(0,1,1)[12]`.
    #[must_use]
    pub fn label(&self) -> String {
        let f = &self.fitted;
        let mut label = format!("ARIMA({},{},{})", f.ar.len(), f.d, f.ma.len());
        if f.period > 1
            && (f.seasonal_d > 0 || !f.seasonal_ar.is_empty() || !f.seasonal_ma.is_empty())
        {
            label.push_str(&format!(
                "({},{},{})[{}]",
                f.seasonal_ar.len(),
                f.seasonal_d,
                f.seasonal_ma.len(),
                f.period
            ));
        }

        if f.with_mean {
            label.push_str(if f.d + f.seasonal_d == 0 {
                " with mean"
            } else {
                " with drift"
            });
        }

        label
    }

    /// The AR coefficients `phi_1, ..., phi_p`.
    #[must_use]
    pub fn ar(&self) -> Vec<f64> {
        self.fitted.ar.clone()
    }

    /// The MA coefficients `theta_1, ..., theta_q`.
    #[must_use]
    pub fn ma(&self) -> Vec<f64> {
        self.fitted.ma.clone()
    }

    /// The seasonal AR coefficients.
    #[wasm_bindgen(js_name = seasonalAr)]
    #[must_use]
    pub fn seasonal_ar(&self) -> Vec<f64> {
        self.fitted.seasonal_ar.clone()
    }

    /// The seasonal MA coefficients.
    #[wasm_bindgen(js_name = seasonalMa)]
    #[must_use]
    pub fn seasonal_ma(&self) -> Vec<f64> {
        self.fitted.seasonal_ma.clone()
    }

    /// The differencing order `d`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn d(&self) -> usize {
        self.fitted.d
    }

    /// The seasonal differencing order `D`.
    #[wasm_bindgen(getter, js_name = seasonalD)]
    #[must_use]
    pub fn seasonal_d(&self) -> usize {
        self.fitted.seasonal_d
    }

    /// The season length; 0 without a seasonal part.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn period(&self) -> usize {
        self.fitted.period
    }

    /// The mean of the differenced series (the drift for `d = 1`); 0 if not estimated.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.fitted.mean
    }

    /// The innovation variance.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn sigma2(&self) -> f64 {
        self.fitted.sigma2
    }

    /// The maximised exact log-likelihood of the differenced series.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn loglike(&self) -> f64 {
        self.fitted.loglike
    }

    /// Akaike's information criterion.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn aic(&self) -> f64 {
        self.fitted.aic()
    }

    /// The small-sample corrected AIC.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn aicc(&self) -> f64 {
        self.fitted.aicc()
    }

    /// The Bayesian information criterion.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn bic(&self) -> f64 {
        self.fitted.bic()
    }

    /// The number of non-missing differenced values the likelihood covers.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn nobs(&self) -> usize {
        self.fitted.nobs
    }

    /// The streaming filter's state count for this model, `ArimaSpec::capacity()`; at most
    /// 64 can stream or forecast.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn states(&self) -> usize {
        self.fitted.spec().capacity()
    }

    /// Forecasts the `horizon` values after `history` with central prediction intervals
    /// at probability `level` (for example 0.95).
    ///
    /// # Errors
    ///
    /// When `history` is too long, `horizon` is out of range, or the model needs more
    /// than 64 states.
    pub fn forecast(
        &self,
        history: &[f64],
        horizon: usize,
        level: f64,
    ) -> Result<Forecast, JsError> {
        check_values(history)?;
        let mut stream = Stream::new(self)?;
        stream.warm_up(history);
        stream.forecast(horizon, level)
    }

    /// Re-estimates the coefficients on `window`, warm-started from this model, through
    /// the heap-free refit a device would run. `mean` keeps the mean (or drift) estimated.
    ///
    /// # Errors
    ///
    /// When the window is too long, too short or degenerate for the model.
    pub fn refit(&self, window: &[f64], mean: bool) -> Result<Model, JsError> {
        check_values(window)?;
        let spec = self.fitted.spec();
        let options = RefitOptions {
            mean,
            ..RefitOptions::default()
        };
        let mut coefficients = vec![0.0; spec.coefficient_len()];
        let mut scratch = vec![0.0; spec.refit_scratch_len(window.len(), options)];
        let refit = spec
            .refit(window, options, &mut coefficients, &mut scratch)
            .map_err(error)?;
        let new = refit.spec;
        let fitted = Fitted {
            ar: new.ar.to_vec(),
            ma: new.ma.to_vec(),
            seasonal_ar: new.seasonal.ar.to_vec(),
            seasonal_ma: new.seasonal.ma.to_vec(),
            with_mean: mean,
            mean: new.mean,
            sigma2: new.sigma2,
            loglike: refit.loglike,
            nobs: refit.nobs,
            iterations: refit.iterations,
            converged: refit.converged,
            ..self.fitted.clone()
        };

        Ok(Model { fitted })
    }
}

/// Forecast means with central prediction intervals, one entry per step ahead.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Forecast {
    /// The forecast means.
    mean: Vec<f64>,
    /// The lower interval bounds.
    lower: Vec<f64>,
    /// The upper interval bounds.
    upper: Vec<f64>,
}

#[wasm_bindgen]
impl Forecast {
    /// The forecast means.
    #[must_use]
    pub fn mean(&self) -> Vec<f64> {
        self.mean.clone()
    }

    /// The lower bounds of the prediction intervals.
    #[must_use]
    pub fn lower(&self) -> Vec<f64> {
        self.lower.clone()
    }

    /// The upper bounds of the prediction intervals.
    #[must_use]
    pub fn upper(&self) -> Vec<f64> {
        self.upper.clone()
    }
}

/// A streaming filter: feed values one at a time and get each one's innovation, the
/// heap-free path a device runs.
#[wasm_bindgen]
pub struct Stream {
    /// The filter, boxed so its 35 KB of state stays off the wasm stack.
    filter: Box<DeviceFilter>,
}

#[wasm_bindgen]
impl Stream {
    /// A filter for `model`, at the model's stationary start.
    ///
    /// # Errors
    ///
    /// When the model needs more than 64 states.
    #[wasm_bindgen(constructor)]
    pub fn new(model: &Model) -> Result<Stream, JsError> {
        let spec = model.fitted.spec();
        let filter = DeviceFilter::new(&spec).map_err(error)?;
        Ok(Stream {
            filter: Box::new(filter),
        })
    }

    /// Feeds every value of `values`, discarding the innovations.
    #[wasm_bindgen(js_name = warmUp)]
    pub fn warm_up(&mut self, values: &[f64]) {
        self.filter.warm_up(values);
    }

    /// Feeds one value and returns `[innovation, variance]`, or `undefined` while the
    /// first `d + D s` values only prime the differencing. A `NaN` value counts as
    /// missing; its innovation is `NaN`.
    pub fn update(&mut self, value: f64) -> Option<Vec<f64>> {
        self.filter
            .update(value)
            .map(|innovation| vec![innovation.value, innovation.variance])
    }

    /// Forecasts the next `horizon` values with central prediction intervals at `level`.
    ///
    /// # Errors
    ///
    /// When `horizon` is 0 or above 10,000.
    pub fn forecast(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        if horizon == 0 || horizon > MAX_HORIZON {
            return Err(JsError::new(&format!(
                "the horizon must be between 1 and {MAX_HORIZON}"
            )));
        }

        let (mut mean, mut lower, mut upper) =
            (vec![0.0; horizon], vec![0.0; horizon], vec![0.0; horizon]);
        self.filter
            .forecast_interval_into(level, &mut mean, &mut lower, &mut upper);
        Ok(Forecast { mean, lower, upper })
    }
}

/// The Ljung–Box test for leftover autocorrelation: `[statistic, pValue]` over `lags`
/// lags with `modelDf` fitted coefficients, or `undefined` when the input does not allow
/// the test (too few or non-finite values, or `modelDf >= lags`).
#[wasm_bindgen(js_name = ljungBox)]
#[must_use]
pub fn ljung_box(residuals: &[f64], lags: usize, model_df: usize) -> Option<Vec<f64>> {
    backshift::ljung_box(residuals, lags, model_df).map(|test| vec![test.statistic, test.p_value])
}

/// The heap-free filter's state capacity: models with more states cannot stream.
#[wasm_bindgen(js_name = filterCapacity)]
#[must_use]
pub fn filter_capacity() -> usize {
    CAPACITY
}

/// Rejects series longer than [`MAX_VALUES`].
fn check_values(values: &[f64]) -> Result<(), JsError> {
    if values.len() > MAX_VALUES {
        return Err(JsError::new(&format!(
            "at most {MAX_VALUES} values are accepted"
        )));
    }

    Ok(())
}

/// Rejects AR or MA orders above [`MAX_ORDER`] and differencing above [`MAX_DIFFERENCING`].
fn check_orders(p: usize, d: usize, q: usize) -> Result<(), JsError> {
    if p > MAX_ORDER || q > MAX_ORDER || d > MAX_DIFFERENCING {
        return Err(JsError::new(&format!(
            "orders must be at most {MAX_ORDER} and differencing at most {MAX_DIFFERENCING}"
        )));
    }

    Ok(())
}

/// A core error as a JavaScript `Error` with its message.
fn error(error: impl core::fmt::Display) -> JsError {
    JsError::new(&error.to_string())
}

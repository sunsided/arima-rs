//! The exact Gaussian log-likelihood of an ARIMA model, for any state dimension.
//!
//! This is the runtime-sized twin of [`Filter`](crate::Filter): the same recursion
//! from [`state_space`](crate::state_space), on `Vec` buffers sized to the model.
//! The series is differenced first (simple differencing, as R's `arima` does).

use crate::difference::difference;
use crate::filter::{ArimaSpec, FilterError};
use crate::math;
use crate::state_space::{self, Innovation, LyapunovScratch, System};
use alloc::vec;
use alloc::vec::Vec;

/// Runs the filter over `diff(y, d) - mean` from the stationary start and returns
/// every innovation: `y.len() - d` of them, or none if `y` is shorter.
///
/// Matches statsmodels `SARIMAX(diff(y, d) - mean, order=(p, 0, q), trend="n").filter(params)`:
/// `value` is `forecasts_error` and `variance` is `forecasts_error_cov`. `NaN`
/// observations count as missing. For `d = 0` this is statsmodels
/// `ARIMA(y, order=(p, 0, q), trend="c")` with `const = mean`.
///
/// # Errors
///
/// [`FilterError::InvalidParameter`] or [`FilterError::NonStationary`], as for
/// [`Filter::new`](crate::Filter::new). There is no capacity limit.
pub fn innovations(y: &[f64], spec: &ArimaSpec<'_>) -> Result<Vec<Innovation>, FilterError> {
    let w = demeaned_difference(y, spec);
    let mut result = Vec::with_capacity(w.len());
    Workspace::new(spec)?.run(&w, |innovation| result.push(innovation));

    Ok(result)
}

/// The exact Gaussian log-likelihood of `y` under the model.
///
/// `-0.5 * sum(ln(2 pi) + ln F_t + v_t^2 / F_t)` over the innovations `v_t` with
/// variances `F_t` of [`innovations`], skipping missing (`NaN`) observations.
///
/// For `d > 0` this differs from statsmodels' default `ARIMA(y, order=(p, d, q))`,
/// which keeps the first `d` values in the likelihood through a diffuse start, by
/// a small amount (about 1e-5 on 60 values). It equals statsmodels with
/// `simple_differencing=True` and R's `arima`.
///
/// # Errors
///
/// As for [`innovations`].
///
/// # Examples
///
/// ```
/// use backshift::{loglike, ArimaSpec};
///
/// // White noise with variance 1: each term is ln(2 pi) + y^2.
/// let spec = ArimaSpec { ar: &[], d: 0, ma: &[], mean: 0.0, sigma2: 1.0 };
/// let llf = loglike(&[0.0, 0.0], &spec).unwrap();
/// assert!((llf + std::f64::consts::TAU.ln()).abs() < 1e-12);
/// ```
pub fn loglike(y: &[f64], spec: &ArimaSpec<'_>) -> Result<f64, FilterError> {
    let w = demeaned_difference(y, spec);
    let ln_two_pi = math::ln(core::f64::consts::TAU);
    let mut sum = 0.0;
    Workspace::new(spec)?.run(&w, |Innovation { value, variance }| {
        if !value.is_nan() {
            sum += ln_two_pi + math::ln(variance) + value * value / variance;
        }
    });

    Ok(-0.5 * sum)
}

/// `diff(y, d) - mean`: the series the ARMA part models.
fn demeaned_difference(y: &[f64], spec: &ArimaSpec<'_>) -> Vec<f64> {
    let mut w = difference(y, spec.d);
    for value in &mut w {
        *value -= spec.mean;
    }

    w
}

/// Heap buffers for one filter run, sized to the model's state dimension `r`.
struct Workspace {
    /// The padded first column of the transition matrix. Length `r`.
    phi: Vec<f64>,
    /// The padded selection vector. Length `r`.
    selection: Vec<f64>,
    /// The innovation variance.
    sigma2: f64,
    /// The predicted state. Length `r`.
    a: Vec<f64>,
    /// The predicted state covariance, `r x r` row-major.
    p: Vec<f64>,
    /// Scratch for the recursion. Length `r`.
    scratch: Vec<f64>,
}

impl Workspace {
    /// Allocates the buffers and computes the stationary starting covariance.
    fn new(spec: &ArimaSpec<'_>) -> Result<Self, FilterError> {
        spec.validate()?;
        let r = spec.arma_states();
        let mut workspace = Self {
            phi: vec![0.0; r],
            selection: vec![0.0; r],
            sigma2: spec.sigma2,
            a: vec![0.0; r],
            p: vec![0.0; r * r],
            scratch: vec![0.0; r],
        };
        spec.pad_into(&mut workspace.phi, &mut workspace.selection);

        let (mut power, mut product, mut term) =
            (vec![0.0; r * r], vec![0.0; r * r], vec![0.0; r * r]);
        let scratch = LyapunovScratch {
            power: &mut power,
            product: &mut product,
            term: &mut term,
        };
        let (system, _, p, _) = workspace.parts();
        if !state_space::stationary_covariance(system, p, r, scratch) {
            return Err(FilterError::NonStationary);
        }

        Ok(workspace)
    }

    /// Feeds every observation and hands each innovation to `sink`.
    fn run(mut self, y: &[f64], mut sink: impl FnMut(Innovation)) {
        let r = self.phi.len();
        let (system, a, p, scratch) = self.parts();
        for &value in y {
            sink(state_space::update(r, a, p, r, value, scratch));
            state_space::predict(system, a, p, r, scratch);
        }
    }

    /// Splits the workspace into the system matrices, the state, the covariance and scratch.
    fn parts(&mut self) -> (System<'_>, &mut [f64], &mut [f64], &mut [f64]) {
        let system = System {
            phi: &self.phi,
            selection: &self.selection,
            sigma2: self.sigma2,
        };

        (system, &mut self.a, &mut self.p, &mut self.scratch)
    }
}

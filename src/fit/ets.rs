//! Maximum-likelihood fitting of additive-error ETS models, and R's automatic selection.
//!
//! [`Ets::fit`] maximises the same likelihood as R's `forecast::ets`, over the smoothing
//! parameters and the initial states, with the innovation variance concentrated out. The
//! parameters stay within R's "usual" bounds through logistic transforms, and R's
//! admissibility conditions reject the rest.
//!
//! It searches differently. The one-step errors are linear in the initial states, so for
//! given smoothing parameters the best states follow by least squares, and BFGS searches
//! only the 1 to 4 smoothing parameters. R runs Nelder–Mead over all of them, up to 28 for a
//! monthly season, and often stops at a worse point: on every model of the R conformance
//! suite this fit reaches the same or a higher likelihood.
//!
//! ```mermaid
//! flowchart LR
//!     U["smoothing parameters u"] -->|"least squares"| S["best initial states"]
//!     S -->|"recursion"| L["0.5 ln(SSE / n)"]
//!     L -->|"BFGS over u, two starts"| F["EtsFit"]
//! ```

use super::hannan_rissanen::cholesky_solve;
use super::roots::roots_inside;
use super::{Criterion, FitError, bfgs};
use crate::ets::{EtsSpec, Smoothing, Trend, unroll};
use crate::math;
use alloc::vec;
use alloc::vec::Vec;

/// R's "usual" lower bound of `alpha`, `beta` and `gamma`.
const LOWER: f64 = 1e-4;

/// R's "usual" upper bound of `alpha`, `beta` and `gamma`.
const UPPER: f64 = 0.9999;

/// R's bounds of the damping parameter `phi`.
const PHI_BOUNDS: (f64, f64) = (0.8, 0.98);

/// The grid of starting points, as fractions of each smoothing parameter's range: `alpha`
/// takes every value, `beta`, `gamma` and `phi` the values of their own grid.
///
/// The likelihood can have more than one local optimum in the smoothing parameters. One
/// evaluation per grid point is cheap next to an optimisation, so the fit scores the grid
/// and runs BFGS from the best [`REFINED`] points only.
const ALPHA_GRID: [f64; 6] = [0.01, 0.05, 0.15, 0.3, 0.5, 0.8];

/// The grid of `beta`, as fractions of its range `[lower, alpha]`; see [`ALPHA_GRID`].
const BETA_GRID: [f64; 3] = [0.01, 0.3, 0.99];

/// The grid of `gamma`, as fractions of its range `[lower, 1 - alpha]`; see [`ALPHA_GRID`].
const GAMMA_GRID: [f64; 2] = [0.01, 0.3];

/// The grid of `phi`, as fractions of its range `[0.8, 0.98]`; see [`ALPHA_GRID`].
const PHI_GRID: [f64; 2] = [0.5, 0.99];

/// How many of the best grid points the fit refines with BFGS.
const REFINED: usize = 2;

/// The largest season length R's `ets` fits; longer seasons need STL or Fourier terms.
pub(crate) const MAX_PERIOD: usize = 24;

/// An additive-error ETS model to fit: simple exponential smoothing by default, with an
/// optional additive or damped trend and an optional additive season.
///
/// # Examples
///
/// ```
/// use backshift::Ets;
///
/// // Holt-Winters with a damped trend on a quarterly series.
/// let y: Vec<f64> = (0..48)
///     .map(|t| 20.0 + 0.3 * t as f64 + [2.0, -1.0, 0.5, -1.5][t % 4] + (t as f64 * 1.7).sin())
///     .collect();
/// let fit = Ets::new().damped().seasonal(4).fit(&y).unwrap();
/// assert_eq!(fit.label(), "ETS(A,Ad,A)");
/// assert!(fit.aicc().is_finite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ets {
    /// Whether the model has an additive trend.
    trend: bool,
    /// Whether the trend is damped; implies `trend`.
    damped: bool,
    /// The season length; 0 without a season.
    period: usize,
}

impl Ets {
    /// Simple exponential smoothing, ETS(A,N,N).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            trend: false,
            damped: false,
            period: 0,
        }
    }

    /// Adds an additive trend (Holt's linear method).
    #[must_use]
    pub const fn trend(self) -> Self {
        Self {
            trend: true,
            ..self
        }
    }

    /// Adds a damped additive trend.
    #[must_use]
    pub const fn damped(self) -> Self {
        Self {
            trend: true,
            damped: true,
            ..self
        }
    }

    /// Adds an additive season of length `period`; 0 or 1 means no season.
    #[must_use]
    pub const fn seasonal(self, period: usize) -> Self {
        Self {
            period: if period > 1 { period } else { 0 },
            ..self
        }
    }

    /// Fits the model to `y` by maximum likelihood.
    ///
    /// # Errors
    ///
    /// - [`FitError::Unsupported`] if a value is not finite (R's `ets` drops all but the
    ///   longest stretch without gaps; this crate leaves that choice to the caller), or the
    ///   period is above 24 (R's limit; use [`mstl`](crate::mstl) or Fourier terms).
    /// - [`FitError::TooShort`] if `y` has no more than four values per parameter beyond
    ///   R's count (`n <= npars + 4`, where R falls back to `HoltWinters`), or no more
    ///   values than one season.
    /// - [`FitError::Degenerate`] if the residuals vanish (a series the model fits exactly).
    pub fn fit(&self, y: &[f64]) -> Result<EtsFit, FitError> {
        self.check(y)?;
        let layout = Layout::of(self);
        let (u, minimum) = layout
            .best_starts(y)
            .into_iter()
            .map(|mut u| {
                let mut scratch = vec![0.0; bfgs::scratch_len(u.len())];
                let minimum = bfgs::minimize(
                    |u| objective(&layout, y, u),
                    &mut u,
                    &mut scratch,
                    bfgs::Options::default(),
                );
                (objective(&layout, y, &u), u, minimum)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, u, minimum)| (u, minimum))
            .ok_or(FitError::Degenerate)?;

        let run = layout.run(y, &u).ok_or(FitError::Degenerate)?;
        if run.sse <= 0.0 || !run.sse.is_finite() {
            return Err(FitError::Degenerate);
        }

        Ok(EtsFit::new(&layout, y.len(), run, minimum))
    }

    /// Rejects input the fit cannot use; see [`Ets::fit`].
    fn check(&self, y: &[f64]) -> Result<(), FitError> {
        if !math::all_finite(y) || self.period > MAX_PERIOD {
            return Err(FitError::Unsupported);
        }

        let needed = self.npars() + 5;
        if y.len() < needed || (self.period > 0 && y.len() <= self.period) {
            return Err(FitError::TooShort {
                needed: needed.max(self.period + 1),
                present: y.len(),
            });
        }

        Ok(())
    }

    /// R's rough parameter count for the short-series check: 2, plus 2 for a trend, `m` for
    /// a season and 1 for damping.
    const fn npars(&self) -> usize {
        2 + 2 * (self.trend as usize) + self.period + (self.damped as usize)
    }

    /// The model in R's notation, for example `ETS(A,Ad,A)`.
    fn label(&self) -> &'static str {
        match (self.trend, self.damped, self.period > 0) {
            (false, _, false) => "ETS(A,N,N)",
            (false, _, true) => "ETS(A,N,A)",
            (true, false, false) => "ETS(A,A,N)",
            (true, false, true) => "ETS(A,A,A)",
            (true, true, false) => "ETS(A,Ad,N)",
            (true, true, true) => "ETS(A,Ad,A)",
        }
    }
}

/// The optimiser's vector `u` for one model: the unconstrained smoothing parameters, `alpha`,
/// then `beta`, `gamma` and `phi` where the model has them.
///
/// The initial states are not in `u`. The one-step errors are linear in them (the recursion
/// is linear), so for given smoothing parameters the states that minimise the residual sum
/// of squares follow exactly by least squares ([`Layout::best_states`]). That concentrates
/// them out of the likelihood: the optimiser searches 1 to 4 numbers instead of up to 28,
/// and reaches the joint optimum that R searches for with all of them.
#[derive(Debug, Clone, Copy)]
struct Layout {
    /// The model.
    model: Ets,
}

impl Layout {
    /// The layout of `model`.
    const fn of(model: &Ets) -> Self {
        Self { model: *model }
    }

    /// The number of smoothing parameters: `alpha`, and `beta`, `gamma`, `phi` where used.
    const fn smoothing_len(&self) -> usize {
        let model = &self.model;
        1 + (model.trend as usize) + ((model.period > 0) as usize) + (model.damped as usize)
    }

    /// The number of free initial states: the level, the slope with a trend, and `m - 1`
    /// seasonal states (the last follows from the others).
    const fn states_len(&self) -> usize {
        1 + (self.model.trend as usize) + self.model.period.saturating_sub(1)
    }

    /// The [`REFINED`] grid points with the lowest objective, as optimiser vectors.
    fn best_starts(&self, y: &[f64]) -> Vec<Vec<f64>> {
        let mut scored: Vec<(f64, Vec<f64>)> = self
            .grid()
            .into_iter()
            .map(|u| (objective(self, y, &u), u))
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.into_iter().take(REFINED).map(|(_, u)| u).collect()
    }

    /// Every grid point of [`ALPHA_GRID`] and the grids of the model's other parameters, as
    /// optimiser vectors (the logits of the fractions).
    fn grid(&self) -> Vec<Vec<f64>> {
        let model = &self.model;
        let axes: [&[f64]; 4] = [
            &ALPHA_GRID,
            if model.trend { &BETA_GRID } else { &[] },
            if model.period > 0 { &GAMMA_GRID } else { &[] },
            if model.damped { &PHI_GRID } else { &[] },
        ];
        let mut points = vec![Vec::new()];
        for axis in axes.iter().filter(|axis| !axis.is_empty()) {
            points = points
                .iter()
                .flat_map(|point| {
                    axis.iter().map(move |&fraction| {
                        let mut next = point.clone();
                        next.push(logit(fraction));
                        next
                    })
                })
                .collect();
        }

        points
    }

    /// The smoothing parameters `u` maps to, within R's usual bounds: `beta < alpha`,
    /// `gamma < 1 - alpha`.
    fn smoothing(&self, u: &[f64]) -> Smoothing {
        let model = &self.model;
        let mut next = u.iter().copied();
        let mut take = || next.next().unwrap_or(0.0);
        let alpha = LOWER + (UPPER - LOWER) * logistic(take());
        let beta = if model.trend {
            LOWER + (alpha.min(UPPER) - LOWER) * logistic(take())
        } else {
            0.0
        };
        let gamma = if model.period > 0 {
            LOWER + ((1.0 - alpha).min(UPPER) - LOWER) * logistic(take())
        } else {
            0.0
        };
        let phi = if model.damped {
            PHI_BOUNDS.0 + (PHI_BOUNDS.1 - PHI_BOUNDS.0) * logistic(take())
        } else {
            1.0
        };

        Smoothing {
            alpha,
            beta,
            gamma,
            phi,
            trend: model.trend,
        }
    }

    /// Writes the seasonal part of the free states `free` into `seasonal` (the last state
    /// makes them sum to zero) and returns the level and slope.
    fn expand(&self, free: &[f64], seasonal: &mut [f64]) -> (f64, f64) {
        let level = free[0];
        let slope = if self.model.trend { free[1] } else { 0.0 };
        let rest = &free[1 + usize::from(self.model.trend)..];
        if let Some(last) = seasonal.len().checked_sub(1) {
            seasonal[..last].copy_from_slice(rest);
            seasonal[last] = -rest.iter().sum::<f64>();
        }

        (level, slope)
    }

    /// Runs the recursion over `y` from the free states `free` and writes the one-step
    /// errors into `errors`. Leaves the final seasonal states in `seasonal` as a ring and
    /// returns the final level, slope and the ring's head (see [`unroll`]).
    fn errors(
        &self,
        smoothing: &Smoothing,
        free: &[f64],
        y: &[f64],
        seasonal: &mut [f64],
        errors: &mut [f64],
    ) -> (f64, f64, usize) {
        let (level, slope) = self.expand(free, seasonal);
        recurse(smoothing, level, slope, seasonal, y, errors)
    }

    /// The free initial states that minimise the residual sum of squares at `smoothing`, or
    /// [`None`] if the least-squares problem is singular.
    ///
    /// The errors are `e = e_0 + E x` in the free states `x`: `e_0` from the data with all
    /// states 0, and column `j` of `E` from all-zero data with state `j` at 1. The fit needs
    /// more values than states, which [`Ets::fit`] checks.
    fn best_states(&self, smoothing: &Smoothing, y: &[f64]) -> Option<Vec<f64>> {
        let (n, k) = (y.len(), self.states_len());
        let mut seasonal = vec![0.0; self.model.period];
        let mut base = vec![0.0; n];
        self.errors(smoothing, &vec![0.0; k], y, &mut seasonal, &mut base);

        let columns = self.state_columns(smoothing, n);

        // The normal equations of `E x = -e_0`: `E'E x = -E'e_0`, lower triangle only.
        let mut gram = vec![vec![0.0; k]; k];
        let mut moment = vec![0.0; k];
        for (i, column_i) in columns.chunks_exact(n).enumerate() {
            moment[i] = -column_i.iter().zip(&base).map(|(c, e)| c * e).sum::<f64>();
            for (j, column_j) in columns.chunks_exact(n).take(i + 1).enumerate() {
                gram[i][j] = column_i.iter().zip(column_j).map(|(a, b)| a * b).sum();
            }
        }

        cholesky_solve(gram, moment)
    }

    /// The columns of `E`, all `k` back to back: the one-step errors over `n` all-zero values
    /// from each free initial state at 1.
    ///
    /// Three recursion passes give them all: the level, the slope, and the oldest seasonal
    /// state. With zero data a state whose forecast error is zero only ages, so a unit in
    /// the seasonal state `j` places from the oldest acts as one in the oldest, `j` steps
    /// later. A free seasonal state `j` is that unit less one in the oldest state (the
    /// oldest makes them sum to zero).
    fn state_columns(&self, smoothing: &Smoothing, n: usize) -> Vec<f64> {
        let k = self.states_len();
        let leading = 1 + usize::from(self.model.trend);
        let zeros = vec![0.0; n];
        let mut seasonal = vec![0.0; self.model.period];
        let mut columns = vec![0.0; k * n];
        let mut unit = vec![0.0; k];
        for (j, column) in columns.chunks_exact_mut(n).take(leading).enumerate() {
            unit.fill(0.0);
            unit[j] = 1.0;
            self.errors(smoothing, &unit, &zeros, &mut seasonal, column);
        }

        let m = self.model.period;
        if m == 0 {
            return columns;
        }

        // The oldest seasonal state at 1 and all other states at 0, which the free states
        // cannot express (the oldest is minus their sum), so the recursion starts from them.
        seasonal.fill(0.0);
        seasonal[m - 1] = 1.0;
        let mut oldest = vec![0.0; n];
        recurse(smoothing, 0.0, 0.0, &mut seasonal, &zeros, &mut oldest);
        for (j, column) in columns.chunks_exact_mut(n).skip(leading).enumerate() {
            // Free state `j` is `m - 1 - j` places from the oldest.
            let delay = m - 1 - j;
            for (t, value) in column.iter_mut().enumerate() {
                let aged = t.checked_sub(delay).map_or(0.0, |s| oldest[s]);
                *value = aged - oldest[t];
            }
        }

        columns
    }

    /// Fits the initial states at `u` and runs the recursion over `y`, or [`None`] if the
    /// parameters are not admissible or the states cannot be solved for.
    fn run(&self, y: &[f64], u: &[f64]) -> Option<Run> {
        let smoothing = self.smoothing(u);
        if !admissible(&smoothing, self.model.period) {
            return None;
        }

        let initial = self.best_states(&smoothing, y)?;
        let mut ring = vec![0.0; self.model.period];
        let mut errors = vec![0.0; y.len()];
        let (level, slope, head) = self.errors(&smoothing, &initial, y, &mut ring, &mut errors);
        let mut seasonal = vec![0.0; self.model.period];
        unroll(&ring, head, &mut seasonal);
        Some(Run {
            smoothing,
            sse: errors.iter().map(|e| e * e).sum(),
            initial,
            level,
            slope,
            seasonal,
        })
    }
}

/// Runs the recursion over `y` from the given states, with the seasonal ring `seasonal` (most
/// recent first, head 0), and writes the one-step errors into `errors`. Returns the final
/// level, slope and the ring's head.
fn recurse(
    smoothing: &Smoothing,
    mut level: f64,
    mut slope: f64,
    seasonal: &mut [f64],
    y: &[f64],
    errors: &mut [f64],
) -> (f64, f64, usize) {
    let mut head = 0;
    for (&value, error) in y.iter().zip(errors.iter_mut()) {
        *error = value - smoothing.forecast(level, slope, seasonal, head);
        smoothing.update(&mut level, &mut slope, seasonal, &mut head, value);
    }

    (level, slope, head)
}

/// What one pass of the recursion leaves: the parameters, the residual sum of squares, the
/// fitted initial states and the states after the last value.
#[derive(Debug, Clone)]
struct Run {
    /// The smoothing parameters.
    smoothing: Smoothing,
    /// The sum of the squared one-step errors.
    sse: f64,
    /// The free initial states: level, slope (with a trend), then `m - 1` seasonal states.
    initial: Vec<f64>,
    /// The level after the last value.
    level: f64,
    /// The slope after the last value.
    slope: f64,
    /// The seasonal states after the last value, most recent first.
    seasonal: Vec<f64>,
}

/// The fitting objective at `u`: `0.5 ln(SSE / n)` at the best initial states, the negative
/// concentrated log-likelihood per value up to a constant, or `+inf` where it cannot be
/// evaluated.
fn objective(layout: &Layout, y: &[f64], u: &[f64]) -> f64 {
    match layout.run(y, u) {
        Some(run) if run.sse > 0.0 && run.sse.is_finite() => {
            0.5 * math::ln(run.sse / count_as_f64(y.len()))
        }
        _ => f64::INFINITY,
    }
}

/// R's `admissible`: the parameters give a forecastable model (the weights of past errors
/// decay). For seasonal models it checks the roots of the characteristic polynomial.
fn admissible(smoothing: &Smoothing, m: usize) -> bool {
    let Smoothing {
        alpha,
        beta,
        gamma,
        phi,
        trend,
    } = *smoothing;
    if !(0.0..=1.0 + 1e-8).contains(&phi) {
        return false;
    }

    if m == 0 {
        let level_ok = alpha >= 1.0 - 1.0 / phi && alpha <= 1.0 + 1.0 / phi;
        let slope_ok =
            !trend || (beta >= alpha * (phi - 1.0) && beta <= (1.0 + phi) * (2.0 - alpha));
        return level_ok && slope_ok;
    }

    seasonal_admissible(alpha, beta, gamma, phi, m)
}

/// The seasonal part of R's `admissible`, roots included.
fn seasonal_admissible(alpha: f64, beta: f64, gamma: f64, phi: f64, m: usize) -> bool {
    let mf = count_as_f64(m);
    let gamma_ok = gamma >= (1.0 - 1.0 / phi - alpha).max(0.0) && gamma <= 1.0 + 1.0 / phi - alpha;
    let alpha_ok =
        alpha >= 1.0 - 1.0 / phi - gamma * (1.0 - mf + phi + phi * mf) / (2.0 * phi * mf);
    let beta_ok = beta >= -(1.0 - phi) * (gamma / mf + alpha);
    if !(gamma_ok && alpha_ok && beta_ok) {
        return false;
    }

    // R's `P`, in increasing powers, is `phi (1 - alpha - gamma)`, `alpha + beta - alpha phi
    // + gamma - 1`, `m - 2` times `alpha + beta - alpha phi`, `alpha + beta - phi`, then 1.
    // Its roots are those of `z^(m + 1) + a_1 z^m + ... + a_(m + 1)` with `a` reversed.
    let middle = alpha + beta - alpha * phi;
    let mut a = vec![alpha + beta - phi];
    a.extend(core::iter::repeat_n(middle, m - 2));
    a.push(middle + gamma - 1.0);
    a.push(phi * (1.0 - alpha - gamma));
    roots_inside(&a, 1.0 + 1e-10)
}

/// The logistic function `1 / (1 + e^-u)`.
fn logistic(u: f64) -> f64 {
    1.0 / (1.0 + math::exp(-u))
}

/// The inverse of [`logistic`] for `0 < p < 1`.
fn logit(p: f64) -> f64 {
    math::ln(p / (1.0 - p))
}

/// A count as a float. Counts here are series lengths and periods, far below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "series lengths and parameter counts stay far below 2^53, where the conversion is exact"
)]
fn count_as_f64(count: usize) -> f64 {
    count as f64
}

/// A fitted additive-error ETS model.
///
/// The log-likelihood is the full Gaussian one, as for [`Fitted`](crate::Fitted). R's `ets`
/// drops its constants: R's `loglik` is this one plus `n (ln(2 pi) + 1 - ln n) / 2`, and
/// R's AIC, `AICc` and BIC are these minus `n (ln(2 pi) + 1 - ln n)`. Both rank models alike.
#[derive(Debug, Clone, PartialEq)]
pub struct EtsFit {
    /// The model in R's notation, for example `ETS(A,Ad,A)`.
    label: &'static str,
    /// The level smoothing parameter.
    pub alpha: f64,
    /// The trend and its fitted parameters.
    pub trend: Trend,
    /// The seasonal smoothing parameter; 0 without a season.
    pub gamma: f64,
    /// The fitted initial states: level, slope (with a trend), then all `m` seasonal
    /// states, most recent first.
    pub initial_states: Vec<f64>,
    /// The level after the last value.
    pub level: f64,
    /// The slope after the last value; 0 without a trend.
    pub slope: f64,
    /// The seasonal states after the last value, most recent first.
    pub seasonal: Vec<f64>,
    /// The innovation variance as R estimates it: the residual sum of squares over
    /// `n - np`, where `np` counts the smoothing parameters and the free initial states.
    pub sigma2: f64,
    /// The exact Gaussian log-likelihood at the estimates.
    pub loglike: f64,
    /// The number of values.
    pub nobs: usize,
    /// R's `np`: the smoothing parameters and the free initial states.
    pub parameters: usize,
    /// The optimiser iterations taken.
    pub iterations: usize,
    /// Whether the optimiser met its tolerance rather than its iteration limit.
    pub converged: bool,
}

impl EtsFit {
    /// Collects the fit from the final run of the recursion.
    fn new(layout: &Layout, n: usize, run: Run, minimum: bfgs::Minimum) -> Self {
        let model = &layout.model;
        let Smoothing {
            alpha,
            beta,
            gamma,
            phi,
            ..
        } = run.smoothing;
        let trend = match (model.trend, model.damped) {
            (false, _) => Trend::None,
            (true, false) => Trend::Additive { beta },
            (true, true) => Trend::Damped { beta, phi },
        };
        let mut initial_seasonal = vec![0.0; model.period];
        let (level, slope) = layout.expand(&run.initial, &mut initial_seasonal);
        let mut initial_states = vec![level];
        if model.trend {
            initial_states.push(slope);
        }
        initial_states.extend(initial_seasonal);

        let parameters = layout.smoothing_len() + layout.states_len();
        let nf = count_as_f64(n);
        Self {
            label: model.label(),
            alpha,
            trend,
            gamma,
            initial_states,
            level: run.level,
            slope: run.slope,
            seasonal: run.seasonal,
            sigma2: run.sse / count_as_f64(n - parameters),
            loglike: -0.5 * nf * (math::ln(core::f64::consts::TAU * run.sse / nf) + 1.0),
            nobs: n,
            parameters,
            iterations: minimum.iterations,
            converged: minimum.converged,
        }
    }

    /// The model in R's notation, for example `ETS(A,Ad,A)`.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        self.label
    }

    /// The number of estimated quantities the criteria count: `parameters + 1` for the
    /// innovation variance, as R counts them.
    const fn counted(&self) -> usize {
        self.parameters + 1
    }

    /// The Akaike information criterion `-2 loglike + 2 k`.
    #[must_use]
    pub fn aic(&self) -> f64 {
        -2.0 * self.loglike + 2.0 * count_as_f64(self.counted())
    }

    /// The small-sample corrected AIC; `+inf` when `n <= k + 1`.
    #[must_use]
    pub fn aicc(&self) -> f64 {
        let k = count_as_f64(self.counted());
        let n = count_as_f64(self.nobs);
        if n <= k + 1.0 {
            return f64::INFINITY;
        }

        self.aic() + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    }

    /// The Bayesian information criterion `-2 loglike + k ln n`.
    #[must_use]
    pub fn bic(&self) -> f64 {
        -2.0 * self.loglike + count_as_f64(self.counted()) * math::ln(count_as_f64(self.nobs))
    }

    /// The fitted model at its states after the last value, for an
    /// [`EtsFilter`](crate::EtsFilter) that continues from there.
    #[must_use]
    pub fn spec(&self) -> EtsSpec<'_> {
        EtsSpec {
            alpha: self.alpha,
            trend: self.trend,
            gamma: self.gamma,
            level: self.level,
            slope: self.slope,
            seasonal: &self.seasonal,
            sigma2: self.sigma2,
        }
    }
}

/// R's automatic ETS selection over the additive-error models, as
/// `ets(y, additive.only = TRUE)`: every combination of no, additive or damped trend with no
/// or an additive season, ranked by `AICc` by default.
///
/// # Examples
///
/// ```
/// use backshift::AutoEts;
///
/// let y: Vec<f64> = (0..60).map(|t| 5.0 + 0.2 * t as f64 + (t as f64 * 2.3).sin()).collect();
/// let best = AutoEts::new().fit(&y).unwrap();
/// assert!(best.label().starts_with("ETS(A,"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoEts {
    /// The season length; 0 for a non-seasonal search.
    period: usize,
    /// The criterion that ranks the models.
    criterion: Criterion,
}

impl Default for AutoEts {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoEts {
    /// A non-seasonal search ranked by `AICc`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            period: 0,
            criterion: Criterion::Aicc,
        }
    }

    /// Also searches seasonal models with season length `period`.
    #[must_use]
    pub const fn seasonal(self, period: usize) -> Self {
        Self { period, ..self }
    }

    /// The criterion that ranks the models.
    #[must_use]
    pub const fn criterion(self, criterion: Criterion) -> Self {
        Self { criterion, ..self }
    }

    /// Fits every candidate model and returns the best by the criterion.
    ///
    /// Seasonal models are skipped where `y` holds no more than one season, or the period is
    /// above 24, as in R; models too short to fit are skipped too.
    ///
    /// # Errors
    ///
    /// The error of simple exponential smoothing if no candidate can be fitted.
    pub fn fit(&self, y: &[f64]) -> Result<EtsFit, FitError> {
        let mut best: Option<(f64, EtsFit)> = None;
        let mut first_error = None;
        for model in self.candidates(y.len()) {
            match model.fit(y) {
                Ok(fit) => {
                    let score = self.score(&fit);
                    if score.is_finite() && best.as_ref().is_none_or(|(b, _)| score < *b) {
                        best = Some((score, fit));
                    }
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }

        best.map(|(_, fit)| fit)
            .ok_or(first_error.unwrap_or(FitError::Degenerate))
    }

    /// The models to try, in R's order: no trend, additive, damped; each without, then with
    /// the season.
    fn candidates(&self, n: usize) -> Vec<Ets> {
        let seasonal = self.period > 1 && self.period <= MAX_PERIOD && n > self.period;
        let trends = [Ets::new(), Ets::new().trend(), Ets::new().damped()];
        trends
            .iter()
            .flat_map(|&model| {
                let with_season = seasonal.then(|| model.seasonal(self.period));
                core::iter::once(model).chain(with_season)
            })
            .collect()
    }

    /// The criterion value of `fit`.
    fn score(&self, fit: &EtsFit) -> f64 {
        match self.criterion {
            Criterion::Aicc => fit.aicc(),
            Criterion::Aic => fit.aic(),
            Criterion::Bic => fit.bic(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concentrated_states_minimise_the_squared_errors() {
        // Checks the state columns directly: at fixed smoothing, moving any free state off
        // the least-squares solution must not lower the sum of squared errors.
        let layout = Layout::of(&Ets::new().damped().seasonal(4));
        let smoothing = layout.smoothing(&[-1.0, -2.0, -1.5, 0.5]);
        let y: Vec<f64> = (0..40)
            .map(|t| {
                let t = f64::from(t);
                20.0 + 0.3 * t + [2.0, -1.0, 0.5, -1.5][(t as usize) % 4] + (t * 1.7).sin()
            })
            .collect();
        let best = layout.best_states(&smoothing, &y).unwrap();
        let sse = |free: &[f64]| {
            let (mut ring, mut errors) = (vec![0.0; 4], vec![0.0; y.len()]);
            layout.errors(&smoothing, free, &y, &mut ring, &mut errors);
            errors.iter().map(|e| e * e).sum::<f64>()
        };
        let minimum = sse(&best);
        for j in 0..best.len() {
            for step in [-1e-3, 1e-3] {
                let mut moved = best.clone();
                moved[j] += step;
                assert!(sse(&moved) > minimum, "state {j}, step {step}");
            }
        }
    }

    #[test]
    fn transforms_respect_rs_bounds() {
        let layout = Layout::of(&Ets::new().damped().seasonal(4));
        for u in [-40.0, 0.0, 40.0] {
            let s = layout.smoothing(&[u, u, u, u]);
            assert!((LOWER..=UPPER).contains(&s.alpha));
            assert!(s.beta >= LOWER && s.beta <= s.alpha + 1e-12);
            assert!(s.gamma >= LOWER - 1e-12 && s.gamma <= 1.0 - s.alpha + 1e-12);
            assert!((PHI_BOUNDS.0..=PHI_BOUNDS.1).contains(&s.phi));
        }
    }

    #[test]
    fn seasonal_states_sum_to_zero() {
        let layout = Layout::of(&Ets::new().seasonal(3));
        let mut seasonal = [0.0; 3];
        assert_eq!(
            layout.expand(&[10.0, 1.5, -0.5], &mut seasonal),
            (10.0, 0.0)
        );
        assert_eq!(seasonal, [1.5, -0.5, -1.0]);
    }

    #[test]
    fn short_series_are_rejected_like_r() {
        let error = Ets::new().trend().fit(&[1.0; 8]).unwrap_err();
        assert_eq!(
            error,
            FitError::TooShort {
                needed: 9,
                present: 8
            }
        );
        assert_eq!(
            Ets::new().fit(&[1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            Err(FitError::Unsupported)
        );
    }
}

//! Maximum-likelihood fitting of ETS models, and R's automatic selection.
//!
//! [`Ets::fit`] maximises the same likelihood as R's `forecast::ets`, over the smoothing
//! parameters and the initial states, with the innovation variance concentrated out. The
//! parameters stay within R's "usual" bounds through logistic transforms, and R's
//! admissibility conditions reject the rest.
//!
//! It searches differently where it can. With an additive error and additive components the
//! one-step errors are linear in the initial states, so for given smoothing parameters the
//! best states follow by least squares, and BFGS searches only the 1 to 4 smoothing
//! parameters. R runs Nelder–Mead over all of them, up to 28 for a monthly season, and often
//! stops at a worse point. With a multiplicative error or season BFGS carries the initial
//! states too, from R's starting states ([`start::start_states`]).
//!
//! ```mermaid
//! flowchart LR
//!     G["grid of smoothing parameters"] -->|"best two"| U["u"]
//!     U -->|"additive: least squares"| S["initial states"]
//!     U -->|"multiplicative: in u"| S
//!     S -->|"recursion"| L["R's lik / 2n"]
//!     L -->|"BFGS"| F["EtsFit"]
//! ```

mod layout;
mod start;

use super::{Criterion, FitError, bfgs};
use crate::ets::{EtsSpec, Form, Smoothing, Trend};
use crate::math;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use layout::{Layout, Run, count_as_f64, objective};

/// The largest season length R's `ets` fits; longer seasons need STL or Fourier terms.
pub(crate) const MAX_PERIOD: usize = 24;

/// An ETS model to fit, in R's taxonomy: simple exponential smoothing by default, with an
/// optional additive or damped trend, an optional additive or multiplicative season, and an
/// additive or multiplicative error.
///
/// # Examples
///
/// ```
/// use backshift::{Ets, Form};
///
/// // Holt-Winters with a damped trend on a quarterly series.
/// let y: Vec<f64> = (0..48)
///     .map(|t| 20.0 + 0.3 * t as f64 + [2.0, -1.0, 0.5, -1.5][t % 4] + (t as f64 * 1.7).sin())
///     .collect();
/// let fit = Ets::new().damped().seasonal(4).fit(&y).unwrap();
/// assert_eq!(fit.label(), "ETS(A,Ad,A)");
///
/// // The multiplicative Holt-Winters method, ETS(M,A,M).
/// let multiplicative = Ets::new()
///     .error(Form::Multiplicative)
///     .trend()
///     .seasonal(4)
///     .season(Form::Multiplicative);
/// assert!(multiplicative.fit(&y).unwrap().aicc().is_finite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ets {
    /// The error form.
    error: Form,
    /// Whether the model has an additive trend.
    trend: bool,
    /// Whether the trend is damped; implies `trend`.
    damped: bool,
    /// The season length; 0 without a season.
    period: usize,
    /// The season form; unused without a season.
    season: Form,
}

impl Ets {
    /// Simple exponential smoothing, ETS(A,N,N).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            error: Form::Additive,
            trend: false,
            damped: false,
            period: 0,
            season: Form::Additive,
        }
    }

    /// Sets the error form: additive (the default), or multiplicative, relative to the
    /// forecast, which needs positive values.
    #[must_use]
    pub const fn error(self, error: Form) -> Self {
        Self { error, ..self }
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

    /// Adds a season of length `period`; 0 or 1 means no season. It is additive unless
    /// [`Ets::season`] says otherwise.
    #[must_use]
    pub const fn seasonal(self, period: usize) -> Self {
        Self {
            period: if period > 1 { period } else { 0 },
            ..self
        }
    }

    /// Sets the season form: additive (the default), or multiplicative, which needs a
    /// multiplicative error and positive values, as in R.
    #[must_use]
    pub const fn season(self, season: Form) -> Self {
        Self { season, ..self }
    }

    /// Fits the model to `y` by maximum likelihood.
    ///
    /// # Errors
    ///
    /// - [`FitError::Unsupported`] if a value is not finite (R's `ets` drops all but the
    ///   longest stretch without gaps; this crate leaves that choice to the caller), the
    ///   period is above 24 (R's limit; use [`mstl`](crate::mstl) or Fourier terms), a
    ///   multiplicative model meets a value that is not positive, or the season is
    ///   multiplicative and the error additive (a combination R forbids).
    /// - [`FitError::TooShort`] if `y` has no more than four values per parameter beyond
    ///   R's count (`n <= npars + 4`, where R falls back to `HoltWinters`), or no more
    ///   values than one season.
    /// - [`FitError::Degenerate`] if the residuals vanish (a series the model fits exactly)
    ///   or no starting point can be scored.
    pub fn fit(&self, y: &[f64]) -> Result<EtsFit, FitError> {
        self.check(y)?;
        let layout = Layout::of(self);
        let start = if layout.concentrated() {
            Vec::new()
        } else {
            start::start_states(y, self.trend, self.period.max(1), self.season)
        };

        let (u, minimum) = layout
            .best_starts(y, &start)
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
            .filter(|(value, ..)| value.is_finite())
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
        let multiplicative = self.error == Form::Multiplicative || self.multiplicative_season();
        let forbidden = self.error == Form::Additive && self.multiplicative_season();
        let positive = y.iter().all(|&v| v > 0.0);
        if !math::all_finite(y)
            || self.period > MAX_PERIOD
            || forbidden
            || (multiplicative && !positive)
        {
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

    /// Whether the model has a season that multiplies the forecast.
    fn multiplicative_season(&self) -> bool {
        self.period > 0 && self.season == Form::Multiplicative
    }

    /// R's rough parameter count for the short-series check: 2, plus 2 for a trend, `m` for
    /// a season and 1 for damping.
    const fn npars(&self) -> usize {
        2 + 2 * (self.trend as usize) + self.period + (self.damped as usize)
    }

    /// The model in R's notation, for example `ETS(M,Ad,M)`.
    fn label(&self) -> String {
        let letter = |form: Form| match form {
            Form::Additive => "A",
            Form::Multiplicative => "M",
        };
        let trend = match (self.trend, self.damped) {
            (false, _) => "N",
            (true, false) => "A",
            (true, true) => "Ad",
        };
        let season = if self.period > 0 {
            letter(self.season)
        } else {
            "N"
        };
        format!("ETS({},{trend},{season})", letter(self.error))
    }
}

/// A fitted ETS model.
///
/// The log-likelihood is the full Gaussian one, as for [`Fitted`](crate::Fitted). R's `ets`
/// drops its constants: R's `loglik` is this one plus `n (ln(2 pi) + 1 - ln n) / 2`, and
/// R's AIC, `AICc` and BIC are these minus `n (ln(2 pi) + 1 - ln n)`. Both rank models alike.
#[derive(Debug, Clone, PartialEq)]
pub struct EtsFit {
    /// The model in R's notation, for example `ETS(M,Ad,M)`.
    label: String,
    /// The error form.
    pub error: Form,
    /// The level smoothing parameter.
    pub alpha: f64,
    /// The trend and its fitted parameters.
    pub trend: Trend,
    /// The season form; unused without a season.
    pub season: Form,
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
    /// The innovation variance as R estimates it: the sum of the squared one-step errors
    /// (relative to the forecasts with a multiplicative error) over `n - np`, where `np`
    /// counts the smoothing parameters and the free initial states.
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
            error: model.error,
            alpha,
            trend,
            season: model.season,
            gamma,
            initial_states,
            level: run.level,
            slope: run.slope,
            seasonal: run.seasonal,
            sigma2: run.sse / count_as_f64(n - parameters),
            loglike: -0.5 * nf * (math::ln(core::f64::consts::TAU * run.sse / nf) + 1.0)
                - run.log_forecasts,
            nobs: n,
            parameters,
            iterations: minimum.iterations,
            converged: minimum.converged,
        }
    }

    /// The model in R's notation, for example `ETS(M,Ad,M)`.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
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
            error: self.error,
            alpha: self.alpha,
            trend: self.trend,
            season: self.season,
            gamma: self.gamma,
            level: self.level,
            slope: self.slope,
            seasonal: &self.seasonal,
            sigma2: self.sigma2,
        }
    }
}

/// R's automatic ETS selection, as `ets(y)`: every combination of an additive or
/// multiplicative error, no, additive or damped trend, and no, additive or multiplicative
/// season that R allows, ranked by `AICc` by default. Multiplicative models need positive
/// values, and R forbids a multiplicative season with an additive error; like R, the search
/// leaves out multiplicative trends.
///
/// # Examples
///
/// ```
/// use backshift::AutoEts;
///
/// let y: Vec<f64> = (0..60).map(|t| 5.0 + 0.2 * t as f64 + (t as f64 * 2.3).sin()).collect();
/// let best = AutoEts::new().fit(&y).unwrap();
/// assert!(best.label().starts_with("ETS("));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoEts {
    /// The season length; 0 for a non-seasonal search.
    period: usize,
    /// The criterion that ranks the models.
    criterion: Criterion,
    /// Whether to search the additive models only, as R's `additive.only = TRUE`.
    additive_only: bool,
}

impl Default for AutoEts {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoEts {
    /// A non-seasonal search over R's default model set, ranked by `AICc`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            period: 0,
            criterion: Criterion::Aicc,
            additive_only: false,
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

    /// Searches the additive models only, as R's `ets(y, additive.only = TRUE)`.
    #[must_use]
    pub const fn additive_only(self, additive_only: bool) -> Self {
        Self {
            additive_only,
            ..self
        }
    }

    /// Fits every candidate model and returns the best by the criterion.
    ///
    /// Seasonal models are skipped where `y` holds no more than one season, or the period is
    /// above 24, as in R; models too short to fit are skipped too.
    ///
    /// # Errors
    ///
    /// The error of the first candidate if no candidate can be fitted.
    pub fn fit(&self, y: &[f64]) -> Result<EtsFit, FitError> {
        let mut best: Option<(f64, EtsFit)> = None;
        let mut first_error = None;
        for model in self.candidates(y) {
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

    /// The models to try: error, then trend (none, damped, additive), then
    /// season (none, additive, multiplicative), leaving out the combinations R forbids.
    fn candidates(&self, y: &[f64]) -> Vec<Ets> {
        let multiplicative = !self.additive_only && y.iter().all(|&v| v > 0.0);
        let seasonal = self.period > 1 && self.period <= MAX_PERIOD && y.len() > self.period;
        let errors: &[Form] = if multiplicative {
            &[Form::Additive, Form::Multiplicative]
        } else {
            &[Form::Additive]
        };
        let trends = [Ets::new(), Ets::new().damped(), Ets::new().trend()];
        let mut seasons = vec![None];
        if seasonal {
            seasons.push(Some(Form::Additive));
            if multiplicative {
                seasons.push(Some(Form::Multiplicative));
            }
        }

        let mut models = Vec::new();
        for &error in errors {
            for trend in trends {
                for &season in &seasons {
                    let model = trend.error(error);
                    let model = match season {
                        None => model,
                        Some(form) => model.seasonal(self.period).season(form),
                    };
                    let forbidden = error == Form::Additive && model.multiplicative_season();
                    if !forbidden {
                        models.push(model);
                    }
                }
            }
        }

        models
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

    #[test]
    fn multiplicative_models_need_positive_data_and_errors() {
        let y = [1.0, 2.0, -1.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0];
        let multiplicative = Ets::new().error(Form::Multiplicative);
        assert_eq!(multiplicative.fit(&y), Err(FitError::Unsupported));
        let forbidden = Ets::new().seasonal(2).season(Form::Multiplicative);
        assert_eq!(forbidden.fit(&[1.0; 12]), Err(FitError::Unsupported));
    }

    #[test]
    fn labels_follow_r() {
        let model = Ets::new()
            .error(Form::Multiplicative)
            .damped()
            .seasonal(12)
            .season(Form::Multiplicative);
        assert_eq!(model.label(), "ETS(M,Ad,M)");
        assert_eq!(Ets::new().seasonal(4).label(), "ETS(A,N,A)");
    }
}

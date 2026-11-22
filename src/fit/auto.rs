//! Automatic order selection: the Hyndman–Khandakar stepwise search of R's `auto.arima`.
//!
//! 1. Choose `D` by seasonal strength ([`nsdiffs`]) and then `d` by KPSS
//!    ([`ndiffs`](crate::ndiffs)), unless they are given.
//! 2. Fit four start models, then move one or two orders at a time to the neighbour
//!    with the best information criterion, until no neighbour improves.
//!
//! Hyndman & Khandakar (2008), "Automatic time series forecasting: the forecast package
//! for R", Journal of Statistical Software 27(3). The start models, the move order and
//! the root check follow R's `auto.arima` and pmdarima's `auto_arima`.
//!
//! ```mermaid
//! flowchart LR
//!     Y["y"] -->|nsdiffs, ndiffs| O["d, D"]
//!     O --> S["4 start models"]
//!     S --> B["best so far"]
//!     B -->|"fit next neighbour"| N{"better?"}
//!     N -->|yes| B
//!     N -->|"no neighbour left"| R["result"]
//! ```

use super::roots::max_inverse_root;
use super::{Arima, FitError, Fitted};
use crate::{ndiffs, polynomial, stl};
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

/// The information criterion that ranks candidate models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Criterion {
    /// The small-sample corrected AIC, R's default.
    #[default]
    Aicc,
    /// Akaike's information criterion.
    Aic,
    /// The Bayesian information criterion.
    Bic,
}

/// An automatic ARIMA model search.
///
/// The defaults are R's `auto.arima`: orders up to `p, q <= 5` and `P, Q <= 2`, `d <= 2`,
/// `D <= 1`, AICc, a stepwise search, and a mean or drift where `d + D <= 1`.
///
/// # Examples
///
/// ```
/// use backshift::AutoArima;
///
/// let y = [0.3, 1.1, 0.4, -0.2, 0.9, 1.4, 0.6, 0.1, -0.5, 0.2, 0.8, 1.0, 0.3, -0.1, 0.4,
///          0.7, 1.2, 0.5, 0.0, -0.3, 0.6, 0.9, 0.2, -0.4, 0.1];
/// let best = AutoArima::new().fit(&y).unwrap();
/// assert!(best.candidates >= 4);
/// assert!(best.fitted.converged);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoArima {
    /// The season length, or 0 for a non-seasonal search.
    period: usize,
    /// Upper bounds on `p, q, P, Q`.
    max: Orders,
    /// The bound on `p + q + P + Q` in the exhaustive search.
    max_order: usize,
    /// The largest `d` [`ndiffs`] may choose.
    max_d: usize,
    /// The largest `D` [`nsdiffs`] may choose.
    max_seasonal_d: usize,
    /// A fixed `d`, or [`None`] to choose it.
    d: Option<usize>,
    /// A fixed `D`, or [`None`] to choose it.
    seasonal_d: Option<usize>,
    /// The criterion that ranks candidates.
    criterion: Criterion,
    /// Whether to search stepwise (`true`) or over every order within the bounds.
    stepwise: bool,
}

/// The four searched orders `(p, q, P, Q)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Orders {
    /// AR order.
    p: usize,
    /// MA order.
    q: usize,
    /// Seasonal AR order.
    seasonal_p: usize,
    /// Seasonal MA order.
    seasonal_q: usize,
}

impl Orders {
    /// `p + q + P + Q`.
    fn total(self) -> usize {
        self.p + self.q + self.seasonal_p + self.seasonal_q
    }
}

/// The best model of a search.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoFit {
    /// The selected model.
    pub fitted: Fitted,
    /// Its value of the search criterion.
    pub criterion: f64,
    /// How many distinct candidate models were fitted.
    pub candidates: usize,
}

impl Default for AutoArima {
    fn default() -> Self {
        Self::new()
    }
}

/// R's limit on the number of models a stepwise search fits.
const MAX_STEPWISE_MODELS: usize = 94;

/// The inverse-root modulus above which a candidate counts as non-stationary or
/// non-invertible, as in pmdarima (R uses roots of modulus below 1.01).
const MAX_INVERSE_ROOT: f64 = 0.99;

impl AutoArima {
    /// A non-seasonal search with R's defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            period: 0,
            max: Orders {
                p: 5,
                q: 5,
                seasonal_p: 2,
                seasonal_q: 2,
            },
            max_order: 5,
            max_d: 2,
            max_seasonal_d: 1,
            d: None,
            seasonal_d: None,
            criterion: Criterion::Aicc,
            stepwise: true,
        }
    }

    /// Searches seasonal models with season length `period` (for example 12 for monthly data).
    #[must_use]
    pub fn seasonal(self, period: usize) -> Self {
        Self { period, ..self }
    }

    /// Upper bounds on `p, q, P, Q`.
    #[must_use]
    pub fn max_orders(self, p: usize, q: usize, seasonal_p: usize, seasonal_q: usize) -> Self {
        Self {
            max: Orders {
                p,
                q,
                seasonal_p,
                seasonal_q,
            },
            ..self
        }
    }

    /// The bound on `p + q + P + Q` in the exhaustive search (R's `max.order`).
    #[must_use]
    pub fn max_order(self, max_order: usize) -> Self {
        Self { max_order, ..self }
    }

    /// Fixes `d` instead of choosing it with [`ndiffs`](crate::ndiffs).
    #[must_use]
    pub fn d(self, d: usize) -> Self {
        Self { d: Some(d), ..self }
    }

    /// Fixes `D` instead of choosing it with [`nsdiffs`]. Ignored without a season length.
    #[must_use]
    pub fn seasonal_d(self, seasonal_d: usize) -> Self {
        Self {
            seasonal_d: Some(seasonal_d),
            ..self
        }
    }

    /// The criterion that ranks candidates.
    #[must_use]
    pub fn criterion(self, criterion: Criterion) -> Self {
        Self { criterion, ..self }
    }

    /// Searches every order within the bounds instead of stepwise. With the `rayon`
    /// feature the candidates are fitted in parallel.
    #[must_use]
    pub fn exhaustive(self) -> Self {
        Self {
            stepwise: false,
            ..self
        }
    }

    /// Selects and fits a model for `y`.
    ///
    /// # Errors
    ///
    /// [`FitError::TooShort`] if no candidate can be fitted, typically because the
    /// series is too short for even the smallest model after differencing.
    pub fn fit(&self, y: &[f64]) -> Result<AutoFit, FitError> {
        self.fit_exog(y, &[], 0)
    }

    /// Selects and fits a regression with ARIMA errors; `regressors` as in
    /// [`Arima::fit_exog`]. As in R, the differencing orders are chosen on the residuals
    /// of a least-squares regression of `y` on the regressors.
    ///
    /// # Errors
    ///
    /// [`FitError::RegressorShape`] for a malformed `regressors`, and
    /// [`FitError::TooShort`] if no candidate can be fitted.
    pub fn fit_exog(
        &self,
        y: &[f64],
        regressors: &[f64],
        columns: usize,
    ) -> Result<AutoFit, FitError> {
        if regressors.len() != y.len() * columns {
            return Err(FitError::RegressorShape {
                expected: y.len() * columns,
                got: regressors.len(),
            });
        }

        let (d, seasonal_d) = self.differencing(&regression_residuals(y, regressors, columns));
        let search = Search {
            auto: self,
            y,
            regressors,
            columns,
            d,
            seasonal_d,
            constant_allowed: d + seasonal_d <= 1,
            seasonal: self.period > 1,
            cache: BTreeMap::new(),
            best: None,
        };

        search.run()
    }

    /// `(d, D)`: the given orders, or `D` by [`nsdiffs`] and then `d` by
    /// [`ndiffs`](crate::ndiffs) on the seasonally differenced series.
    fn differencing(&self, x: &[f64]) -> (usize, usize) {
        let seasonal_d = match (self.seasonal_d, self.period > 1) {
            (Some(seasonal_d), true) => seasonal_d,
            (None, true) => nsdiffs(x, self.period, self.max_seasonal_d),
            (_, false) => 0,
        };

        let mut seasonally = x.to_vec();
        for _ in 0..seasonal_d {
            seasonally = seasonal_difference(&seasonally, self.period);
        }

        let d = self
            .d
            .unwrap_or_else(|| ndiffs(&seasonally, 0.05, self.max_d));
        (d, seasonal_d)
    }
}

/// A candidate key: the orders and whether a constant is included.
type Key = (Orders, bool);

/// The state of one search.
struct Search<'a> {
    /// The configuration.
    auto: &'a AutoArima,
    /// The series.
    y: &'a [f64],
    /// The regressor rows.
    regressors: &'a [f64],
    /// The number of regressors.
    columns: usize,
    /// The chosen `d`.
    d: usize,
    /// The chosen `D`.
    seasonal_d: usize,
    /// Whether a mean or drift may be included (`d + D <= 1`).
    constant_allowed: bool,
    /// Whether seasonal orders are searched.
    seasonal: bool,
    /// Every candidate fitted so far, with its criterion (`+inf` when rejected).
    cache: BTreeMap<Key, (f64, Option<Fitted>)>,
    /// The best candidate so far.
    best: Option<(Key, f64)>,
}

impl Search<'_> {
    /// Runs the configured search and returns the best model.
    fn run(mut self) -> Result<AutoFit, FitError> {
        if self.auto.stepwise {
            self.stepwise();
        } else {
            self.exhaustive();
        }

        let candidates = self.cache.len();
        let (key, criterion) = self.best.ok_or(FitError::TooShort {
            needed: 0,
            present: 0,
        })?;
        let fitted = self
            .cache
            .remove(&key)
            .and_then(|(_, fitted)| fitted)
            .expect("the best key was cached");

        Ok(AutoFit {
            fitted,
            criterion,
            candidates,
        })
    }

    /// The Hyndman–Khandakar stepwise search, in R's move order.
    fn stepwise(&mut self) {
        let max = self.limits();
        let mut constant = self.constant_allowed;
        let mut at = Orders {
            p: max.p.min(2),
            q: max.q.min(2),
            seasonal_p: max.seasonal_p.min(1),
            seasonal_q: max.seasonal_q.min(1),
        };

        self.try_model(at, constant);
        let null = Orders {
            p: 0,
            q: 0,
            seasonal_p: 0,
            seasonal_q: 0,
        };
        if self.try_model(null, constant) {
            at = null;
        }

        if max.p > 0 || max.seasonal_p > 0 {
            let ar = Orders {
                p: usize::from(max.p > 0),
                seasonal_p: usize::from(max.seasonal_p > 0),
                ..null
            };
            if self.try_model(ar, constant) {
                at = ar;
            }
        }

        if max.q > 0 || max.seasonal_q > 0 {
            let ma = Orders {
                q: usize::from(max.q > 0),
                seasonal_q: usize::from(max.seasonal_q > 0),
                ..null
            };
            if self.try_model(ma, constant) {
                at = ma;
            }
        }

        if constant && self.try_model(null, false) {
            at = null;
        }

        while self.cache.len() < MAX_STEPWISE_MODELS {
            match self.improving_neighbour(at, constant, max) {
                Some((next, next_constant)) => {
                    at = next;
                    constant = next_constant;
                }
                None => break,
            }
        }
    }

    /// Tries the neighbours of `at` in R's order and returns the first that improves the
    /// best model, if any.
    fn improving_neighbour(
        &mut self,
        at: Orders,
        constant: bool,
        max: Orders,
    ) -> Option<(Orders, bool)> {
        // (dp, dq, dP, dQ) in the order of R's newarima2 and pmdarima.
        const MOVES: [(isize, isize, isize, isize); 16] = [
            (0, 0, -1, 0),
            (0, 0, 0, -1),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (0, 0, -1, -1),
            (0, 0, -1, 1),
            (0, 0, 1, -1),
            (0, 0, 1, 1),
            (-1, 0, 0, 0),
            (0, -1, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (-1, -1, 0, 0),
            (-1, 1, 0, 0),
            (1, -1, 0, 0),
            (1, 1, 0, 0),
        ];

        // The candidates in R's order: the 16 moves, then the constant toggle.
        let mut candidates: Vec<Key> = MOVES
            .iter()
            .filter_map(|&delta| moved(at, delta, max))
            .map(|orders| (orders, constant))
            .collect();
        if self.constant_allowed {
            candidates.push((at, !constant));
        }
        candidates.retain(|key| !self.cache.contains_key(key));

        // With rayon, a seasonal search fits every candidate at once. Either way they are
        // recorded in order and the walk stops at the first improvement, so the search, its
        // cache and its model cap behave exactly as the sequential R algorithm; fits past
        // the first improvement are discarded. Non-seasonal candidates are cheap enough that
        // the discarded fits cost more than the parallelism saves, so they stay sequential.
        let mut speculative = self.speculate(&candidates).into_iter();

        for key in candidates {
            if self.cache.len() >= MAX_STEPWISE_MODELS {
                return None;
            }

            let score = speculative
                .next()
                .unwrap_or_else(|| self.score(key.0, key.1));

            if self.record(key, score) {
                return Some(key);
            }
        }

        None
    }

    /// Fits every order within the bounds and `p + q + P + Q <= max_order`.
    fn exhaustive(&mut self) {
        let max = self.limits();
        let constants: &[bool] = if self.constant_allowed {
            &[true, false]
        } else {
            &[false]
        };
        let mut keys = Vec::new();
        for p in 0..=max.p {
            for q in 0..=max.q {
                for seasonal_p in 0..=max.seasonal_p {
                    for seasonal_q in 0..=max.seasonal_q {
                        let orders = Orders {
                            p,
                            q,
                            seasonal_p,
                            seasonal_q,
                        };
                        if orders.total() <= self.auto.max_order {
                            keys.extend(constants.iter().map(|&c| (orders, c)));
                        }
                    }
                }
            }
        }

        for (key, scored) in keys.iter().zip(self.score_all(&keys)) {
            self.record(*key, scored);
        }
    }

    /// Fits and scores every key, in parallel with the `rayon` feature.
    fn score_all(&self, keys: &[Key]) -> Vec<(f64, Option<Fitted>)> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            keys.par_iter()
                .map(|&(orders, constant)| self.score(orders, constant))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            keys.iter()
                .map(|&(orders, constant)| self.score(orders, constant))
                .collect()
        }
    }

    /// Scores all `keys` in parallel for a seasonal search with the `rayon` feature, and
    /// nothing otherwise (the caller then scores lazily).
    ///
    /// Only as many candidates as the model cap leaves room for are fitted.
    fn speculate(&self, keys: &[Key]) -> Vec<(f64, Option<Fitted>)> {
        if cfg!(feature = "rayon") && self.seasonal {
            let room = MAX_STEPWISE_MODELS.saturating_sub(self.cache.len());
            self.score_all(&keys[..keys.len().min(room)])
        } else {
            Vec::new()
        }
    }

    /// The bounds, with the seasonal ones zeroed for a non-seasonal search.
    fn limits(&self) -> Orders {
        let max = self.auto.max;
        if self.seasonal {
            max
        } else {
            Orders {
                seasonal_p: 0,
                seasonal_q: 0,
                ..max
            }
        }
    }

    /// Fits a candidate once and reports whether it became the best model.
    fn try_model(&mut self, orders: Orders, constant: bool) -> bool {
        let key = (orders, constant);
        if self.cache.contains_key(&key) {
            return false;
        }

        let scored = self.score(orders, constant);
        self.record(key, scored)
    }

    /// Stores a scored candidate and reports whether it became the best model.
    fn record(&mut self, key: Key, scored: (f64, Option<Fitted>)) -> bool {
        let score = scored.0;
        self.cache.insert(key, scored);
        let improves = score.is_finite() && self.best.is_none_or(|(_, best)| score < best);
        if improves {
            self.best = Some((key, score));
        }

        improves
    }

    /// Fits a candidate and returns its criterion, or `+inf` if it fails or has an inverse
    /// root of modulus above [`MAX_INVERSE_ROOT`].
    fn score(&self, orders: Orders, constant: bool) -> (f64, Option<Fitted>) {
        let model = Arima::new(orders.p, self.d, orders.q);
        let model = if self.seasonal {
            model.seasonal(
                orders.seasonal_p,
                self.seasonal_d,
                orders.seasonal_q,
                self.auto.period,
            )
        } else {
            model
        };
        let model = model.mean(constant);
        let Ok(fitted) = model.fit_exog(self.y, self.regressors, self.columns) else {
            return (f64::INFINITY, None);
        };

        let value = match self.auto.criterion {
            Criterion::Aicc => fitted.aicc(),
            Criterion::Aic => fitted.aic(),
            Criterion::Bic => fitted.bic(),
        };
        if !value.is_finite() || near_unit_root(&fitted) {
            return (f64::INFINITY, Some(fitted));
        }

        (value, Some(fitted))
    }
}

/// `at` moved by `delta`, or [`None`] if that leaves `[0, max]` in any order.
fn moved(at: Orders, delta: (isize, isize, isize, isize), max: Orders) -> Option<Orders> {
    let step = |value: usize, by: isize, limit: usize| {
        let next = value.checked_add_signed(by)?;
        (next <= limit).then_some(next)
    };

    Some(Orders {
        p: step(at.p, delta.0, max.p)?,
        q: step(at.q, delta.1, max.q)?,
        seasonal_p: step(at.seasonal_p, delta.2, max.seasonal_p)?,
        seasonal_q: step(at.seasonal_q, delta.3, max.seasonal_q)?,
    })
}

/// Whether the fitted AR or MA polynomial (seasonal parts expanded) has an inverse root
/// of modulus above [`MAX_INVERSE_ROOT`].
fn near_unit_root(fitted: &Fitted) -> bool {
    let expand = |a: &[f64], b: &[f64], sign: f64| {
        let mut out = vec![0.0; a.len() + b.len() * fitted.period];
        polynomial::seasonal_product(a, b, fitted.period, sign, &mut out);
        out
    };

    // 1 - sum c_k z^k for AR, 1 + sum m_k z^k for MA.
    let ar: Vec<f64> = expand(&fitted.ar, &fitted.seasonal_ar, -1.0)
        .iter()
        .map(|c| -c)
        .collect();
    let ma = expand(&fitted.ma, &fitted.seasonal_ma, 1.0);

    max_inverse_root(&ar) > MAX_INVERSE_ROOT || max_inverse_root(&ma) > MAX_INVERSE_ROOT
}

/// `x_t - x_(t-period)`.
fn seasonal_difference(x: &[f64], period: usize) -> Vec<f64> {
    (period..x.len()).map(|t| x[t] - x[t - period]).collect()
}

/// The residuals of a least-squares regression of `y` on `[1, regressors]`, or `y`
/// itself without regressors or when the regression fails.
fn regression_residuals(y: &[f64], regressors: &[f64], columns: usize) -> Vec<f64> {
    if columns == 0 {
        return y.to_vec();
    }

    let row = |t: usize| {
        let mut r = vec![1.0];
        r.extend_from_slice(&regressors[t * columns..(t + 1) * columns]);
        r
    };
    let complete: Vec<usize> = (0..y.len())
        .filter(|&t| y[t].is_finite() && row(t).iter().all(|v| v.is_finite()))
        .collect();
    let rows: Vec<Vec<f64>> = complete.iter().map(|&t| row(t)).collect();
    let targets: Vec<f64> = complete.iter().map(|&t| y[t]).collect();

    match super::hannan_rissanen::least_squares(&rows, &targets) {
        Some(beta) => (0..y.len())
            .map(|t| y[t] - row(t).iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>())
            .collect(),
        None => y.to_vec(),
    }
}

/// The number of seasonal differences, at most `max_seasonal_d`, by seasonal strength.
///
/// R's `nsdiffs` default: difference while the seasonal strength of an STL
/// decomposition (see [`Decomposition::seasonal_strength`](crate::Decomposition::seasonal_strength))
/// is at least 0.64. The decomposition uses statsmodels' STL defaults; R's own STL
/// settings for this heuristic may differ near the threshold. A series too short for
/// [`stl`](crate::stl), or with non-finite values, counts as non-seasonal.
///
/// # Examples
///
/// ```
/// use backshift::nsdiffs;
///
/// let seasonal: Vec<f64> = (0..48).map(|t| [3.0, -1.0, 0.5, -2.5][t % 4] + 0.01 * t as f64).collect();
/// assert_eq!(nsdiffs(&seasonal, 4, 1), 1);
/// ```
#[must_use]
pub fn nsdiffs(x: &[f64], period: usize, max_seasonal_d: usize) -> usize {
    let mut series = x.to_vec();
    for seasonal_d in 0..max_seasonal_d {
        let strength = stl(&series, period).map_or(0.0, |parts| parts.seasonal_strength());
        if strength < 0.64 {
            return seasonal_d;
        }

        series = seasonal_difference(&series, period);
    }

    max_seasonal_d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_respect_bounds() {
        let max = Orders {
            p: 2,
            q: 2,
            seasonal_p: 1,
            seasonal_q: 1,
        };
        let at = Orders {
            p: 0,
            q: 2,
            seasonal_p: 1,
            seasonal_q: 0,
        };
        assert_eq!(moved(at, (-1, 0, 0, 0), max), None);
        assert_eq!(moved(at, (0, 1, 0, 0), max), None);
        assert_eq!(moved(at, (0, 0, 1, 0), max), None);
        assert_eq!(
            moved(at, (1, -1, -1, 1), max),
            Some(Orders {
                p: 1,
                q: 1,
                seasonal_p: 0,
                seasonal_q: 1
            })
        );
    }

    #[test]
    fn nsdiffs_separates_seasonal_and_noise() {
        let seasonal: Vec<f64> = (0..48)
            .map(|t| [3.0, -1.0, 0.5, -2.5][t % 4] + 0.05 * t as f64)
            .collect();
        assert_eq!(nsdiffs(&seasonal, 4, 1), 1);

        let mut state = 9_u32;
        let noise: Vec<f64> = (0..48)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f64::from(state >> 8) / f64::from(1_u32 << 24)
            })
            .collect();
        assert_eq!(nsdiffs(&noise, 4, 1), 0);
        assert_eq!(nsdiffs(&noise[..7], 4, 1), 0);
    }
}

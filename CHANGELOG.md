# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Seasonal ARIMA(p, d, q)(P, D, Q, s): `Seasonal` in `ArimaSpec`, `Arima::seasonal`, and seasonal fields on
  `Fitted`. The filter expands the seasonal polynomials into its own arrays, so it stays heap-free.
- Regression with ARIMA errors: `ArimaSpec::exog`, `Arima::fit_exog`, `Filter::update_exog`, `warm_up_exog`,
  `forecast_exog_into` and `forecast_interval_exog_into`, `innovations_exog` and `loglike_exog`. Fitted models
  report `exog`; `FilterError::Regressors` and `FitError::RegressorShape` are new.
- `ArimaSpec::capacity()`: the smallest `R` a `Filter` needs, regressors included.
- Diagnostics: `ljung_box`, `jarque_bera`, `kpss` (with `KpssNull` and `kpss_lags`) returning `TestResult`, all
  heap-free, and `ndiffs` to choose the differencing order.
- `AutoArima`: the Hyndman–Khandakar stepwise search of R's `auto.arima` (or an exhaustive search), returning
  `AutoFit`; `Criterion` picks AICc, AIC or BIC. `nsdiffs` chooses `D` by STL seasonal strength.
- `stl` and `Decomposition`: STL decomposition matching statsmodels' `STL`, with `seasonal_strength`.
- `kpss_short_lags`; `ndiffs` now uses it (as pmdarima and R's tseries do), which gives KPSS more power on short
  series.
- The `rayon` feature fits the candidates of an exhaustive `AutoArima` search in parallel, and the neighbours of a
  seasonal stepwise search speculatively (with identical results).
- Heap-free refitting: `ArimaSpec::refit` and `refit_exog` re-estimate a model on a window of recent values by exact
  maximum likelihood, warm-started from its current coefficients, in caller-owned memory sized by the `const fn`s
  `refit_scratch_len` and `coefficient_len`. `Refitter` holds both buffers inline, `RefitOptions` sets the mean and
  the iteration limit, and `Refit` returns the new spec with its log-likelihood. `FitError` is available without
  `alloc` and gains `BufferTooSmall`. `examples/device_refit.rs` shows a drift-check-and-refit loop.
- `bindings/wasm`: the `@sunsided/backshift` npm package (0.1.1 is the first release published by CI) (WebAssembly, ES module) with `Model.fit`,
  `Model.autoArima`, forecasts with intervals, `Stream` for streaming innovations, `refit` and `ljungBox`, built
  with the `libm` backend. `publish-npm.yml` publishes it to GitHub Packages and npmjs.com on `wasm-v*` tags.
- `site/`: a demo site for GitHub Pages (SvelteKit, built on the npm package) with three interactive panels: fit and
  forecast with intervals, streaming anomaly scores, and drift detection with in-place refits. `pages.yml` deploys it
  from `main`.
- README charts (forecast, streaming anomalies, drift and refit) in light and dark variants, drawn by
  `site/scripts/readme-charts.ts` with the site's chart code; CI checks that they are current.
- Criterion benchmarks (`cargo bench`) and `ci/compare`, a comparison with foresight and anofox-forecast.

### Performance

- Fitting allocates once per fit instead of on every likelihood evaluation and optimiser step.

- The Lyapunov solve uses branch-free, symmetry-aware matrix products, and the Kalman predict and update steps skip
  the redundant half of the symmetric covariance: about 4x faster starting covariances and 1.5x faster likelihoods.
- The fitting objective freezes the covariance once the filter converges (statsmodels' 1e-19 tolerance), so long
  series cost `O(r)` per step; `loglike`, `innovations` and the filter keep the full recursion.

### Changed

- `ArimaSpec` gains a `seasonal` field and implements `Default`; write `..ArimaSpec::default()` for the fields you
  don't set.
- Differencing inside the filter and the likelihood now applies the full polynomial `(1 - B)^d (1 - B^s)^D` in one
  step, which can change results for `d >= 2` in the last bits.
- `FitError` gains `Degenerate`: a model that fits the series exactly (a constant series with a mean, a straight
  line with a drift) has zero innovation variance, and `Arima::fit` now returns this error instead of a model with
  `sigma2 = 0` and an infinite log-likelihood that `Filter::new` rejects.
- A fit whose innovation standard deviation is below 1e-9 of the range of the differenced data returns
  `FitError::Degenerate` too, so an exact fit is caught however close the optimiser gets to it. A model without a mean and with `d = 0` can
  still run towards a unit root on a constant series, because the likelihood is unbounded there; `refit` rejects a
  constant window outright.
- `Arima::fit` and `fit_exog` treat infinite values in `y` or the regressors as missing, as `NaN` already was.
- `cov` returns `None` for two series with different estimators instead of using the first one's.
- `AutoArima` reports the error of the simplest failed candidate when no candidate fits, instead of
  `TooShort { needed: 0, present: 0 }`.

### Fixed

- `Arima::fit_exog` and `AutoArima::fit_exog` no longer panic on an infinite value.
- `ljung_box` overflowed `n * (n + 2)` on 32-bit targets (WASM, Cortex-M) from 65,535 residuals: a panic in debug
  builds, a wrong statistic in release builds.
- `TimeSeries::slice_stationary` returns `NotEnoughData` for an out-of-bounds range instead of panicking.

## [0.1.0] - 2026-11-22

### Added

- `Arima::new(p, d, q).fit(y)`: exact maximum-likelihood fitting of ARIMA models with an optional mean or drift,
  reporting the log-likelihood, AIC, AICc and BIC.
- `Filter<R>`: a heap-free streaming ARIMA filter built from coefficients alone, with innovations, missing-value
  handling, and original-scale forecasts with prediction intervals.
- `loglike` and `innovations` for any model size.
- Estimators and statistics: `acov`, `cov`, `acorr`, `corr`, `pacf`, `difference`, `integrate`, `yule_walker`,
  `burg`, `hannan_rissanen` and `normal_quantile`.
- `no_std` support with `std`, `libm` and `alloc` features; without `alloc` the crate needs no global allocator.

[0.1.0]: https://github.com/sunsided/arima-rs/releases/tag/v0.1.0

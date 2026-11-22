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

### Changed

- `ArimaSpec` gains a `seasonal` field and implements `Default`; write `..ArimaSpec::default()` for the fields you
  don't set.
- Differencing inside the filter and the likelihood now applies the full polynomial `(1 - B)^d (1 - B^s)^D` in one
  step, which can change results for `d >= 2` in the last bits.

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

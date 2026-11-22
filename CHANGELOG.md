# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

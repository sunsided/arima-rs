# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/sunsided/arima-rs/compare/v0.1.0...v0.2.0) - 2026-11-22

### Added

- Add an interactive demo site for GitHub Pages
- Add benchmarks, a competitor comparison and faster fitting
- Add auto-ARIMA, STL and seasonal differencing selection
- Add Ljung-Box, Jarque-Bera, KPSS and ndiffs
- Add regression with ARIMA errors
- Add seasonal ARIMA

### Changed

- Refresh the bindings' lock entry for the crate in CI
- Automate crate releases with release-plz
- Run the crate on wasm32-wasip1 in CI
- Say statsmodels, not Python, where the crate goes instead
- Show the crate's edges as charts in the README
- Retitle the demo page
- Fit in a worker so slow searches never freeze the page
- Tighten the drift panel's wording
- Bound the state count and fix prerelease detection in the npm release
- Build JavaScript models from known coefficients
- Package the crate for JavaScript as @sunsided/backshift
- Build the no-alloc refit window from plain arithmetic
- Reject constant refit windows and make the noise floor offset-invariant
- Refit models on the device without a heap
- Run the fitting objective on caller-owned memory
- Reject an infinite concentrated variance
- Reject degenerate fits and treat infinite values as missing
- Share one dot product helper
- Return errors instead of panicking in slice_stationary and cov
- Compute the Ljung-Box factor in floating point
- Report the exact likelihood after fitting and cap speculative fits
- Make ndiffs robust at the table edges and with missing values
- Document the filter capacity with regressors
- Date the 0.1.0 changelog entry

### Fixed

- Fix pasted-data parsing, worker failures and wasm leaks in the site
- Fix the STL trend window and make the root check fail closed

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

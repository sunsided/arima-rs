# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/sunsided/arima-rs/compare/v0.2.1...v0.3.0) - 2026-11-24

### Added

- Expose approximate fits and searches in the WASM bindings
- Add approximate CSS fits and an approximate auto-ARIMA search
- Add tests for the gaps mutation testing found
- Add a fixed-memory streaming wrapper for MSTL plus ARIMA
- Add the Box-Cox transformation for MSTL
- Add fixed parts to Arima fits
- Add the PMM2 estimator for skewed innovations
- Add an adaptive online AR mode
- Add conformal prediction intervals for streaming forecasts
- Add seasonal period detection
- Add forecasting with MSTL plus ARIMA
- Add MSTL decomposition
- Add start values to Arima fits
- Add a Fourier term search by AICc to AutoArima
- Add a heap-free Fourier regressor generator
- Add an opt-in steady-state mode to Filter
- Add a SARIMAX example with bank holidays and outages

### Changed

- Restore residual_is_missing's docs and check the CSS scratch size
- State the symmetry invariant precisely and strengthen two Kani proofs
- Document constant STL inputs and empty period lists
- Explain the gap in the seasonal search times
- Benchmark the approximate fits and searches against the other crates
- Update the benchmark tables to the faster fits
- Document the verification tools and remove equivalent Ljung-Box mutants
- Test Jarque-Bera at its two-value boundary
- Warm-start exact ML fits with conditional sum of squares
- Halve the Kalman filter's work per fitted model
- Build the comparison crate and no_std with alloc in CI, pin the R image
- Fail the R conformance job on any drift
- Change ndiffs to R's KPSS lag and add an R conformance suite
- Test multi-coefficient and non-stationary fixed parts
- Test that PMM2's innovations line up with the differenced series
- Assert PMM2's innovation alignment and test a mean with an MA term
- Document PMM2's convergence and efficiency range, and test a drift
- Cap the online AR inverse against windup
- Test that infinite values give no period candidates
- Forecast a series whose length is no multiple of its periods
- Reject non-finite values up front in mstl
- Check the regressor shape before the start values
- Pin the Fourier search's fit to its terms in a harder case
- Document the invariant of the Fourier term search
- Reject empty Fourier terms
- Clarify the regressor capacity in the docs and the wasm binding
- Change Filter to hold regressors in a separate capacity M
- Size the example's filter by the model's capacity
- Build the SARIMAX example only with alloc

### Fixed

- Fix state-count overflows on the device path, add Kani proofs and model properties
- Fix demo-site hangs, leaks and stale charts
- Fix the stepwise search's constant and edge-case panics
- Fix constant series that read as varying, and bound find_period's cost
- Fix the streaming filters on infinite input

### Removed

- Drop the conformal Kani proof, which does not finish

## [0.2.1](https://github.com/sunsided/arima-rs/compare/v0.2.0...v0.2.1) - 2026-11-22

### Fixed

- Fix the README's images and links on crates.io

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

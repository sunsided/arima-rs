# Plan: an ARIMA crate that leads on one thing first

Research date: 2026-11-22. Toolchain used for checks: rustc 1.100.0, statsmodels 0.14.0 (numpy 1.24.3, scipy 1.11.1).
Evidence sources: crates.io API (metadata, `.crate` downloads from static.crates.io), the downloaded crate sources,
the GitHub REST API (stars and last push), local `cargo tree` and cross-compile probes, local statsmodels runs.

## Status

The crate is published as `backshift`, because `arima` is taken on crates.io. Phase 1 ships as one pull request per step, in this order. It differs from section 3: the streaming filter comes before the optimiser, because it is the lead edge and already works with coefficients fitted elsewhere.

1. `no_std`, the `std`/`libm` features, and CI for bare-metal and WASM targets.
   - Follow-up: an `alloc` feature (implied by `std`) gates every `Vec` API. With `alloc` in the crate graph, firmware fails to link without a `#[global_allocator]` even if nothing allocates. CI links a bare-metal binary (`ci/firmware`) to keep that true.
2. Differencing and integration.
3. AR fitting: Yule–Walker and Burg.
4. The Kalman core and `Filter<R>`, in two parts. The likelihood and the filter share one allocation-free recursion over slices with a stride, written for this crate rather than on `minikalman`, whose dimensions are compile-time only.
   - 4a: ARMA only (`d = 0`, no mean). Log-likelihood, innovations and forecasts at fixed parameters, streaming checked against statsmodels `extend`, and a counting-allocator test.
     - statsmodels freezes the state covariance once it changes by less than `ssm.tolerance` (1e-19) and reuses it, which moves later innovations by up to about 1e-9. The crate always runs the full recursion, so references are generated with `tolerance = 0`. `Filter::with_steady_state` is that opt-in mode for devices; it matches the default to 1e-13 (#41).
   - 4b: differencing and a mean or drift inside the filter. statsmodels `trend='c'` in SARIMAX is an intercept, not a mean, and `d >= 1` has no stationary start, so these need their own conventions.
5. Fitting: Hannan–Rissanen, BFGS and `fit`.
6. Forecast intervals and fit statistics: AIC, AICc and BIC.

Phase 2 (seasonal ARIMA) is done: the filter and the likelihood expand `phi(B) Phi(B^s)`, `theta(B) Theta(B^s)` and `(1 - B)^d (1 - B^s)^D` into plain coefficient vectors, and the forecast integrates through the last `d + D s` raw values. Seasonal fits start the seasonal parts at zero.

Phase 3 (exogenous regressors) is done: the ARIMA part models `u = y - X beta`, the regressors are differenced with `y` (R's `arima(xreg=)`, statsmodels' simple differencing), and `beta` is estimated jointly, starting from least squares on the differenced data.

Phase 4 (diagnostics) is done: Ljung–Box, Jarque–Bera and KPSS match statsmodels to 1e-9 including p-values, through an in-crate regularised incomplete gamma (Numerical Recipes 3e §6.2) and Lanczos log-gamma. `ndiffs` now uses R's own KPSS lag, `trunc(3 sqrt(n) / 13)`, pinned against forecast 8.23 in the R conformance suite (`scripts/generate_r_reference.R`).

Phase 5 (auto-ARIMA) is done. The stepwise search follows R's `auto.arima` (start models, move order, the constant only for `d + D <= 1`, the 0.99 inverse-root check) and picks the same orders as pmdarima 2.1.1 on eight simulated series with `d` and `D` fixed; pmdarima also served as the reference that R was meant to be. `ndiffs` follows R's lag rule; pmdarima's short rule differs on two of its eight series, where this crate sides with R. `nsdiffs` uses STL seasonal strength, R's default, with a port of the STL Fortran that matches statsmodels to 1e-9; pmdarima's OCSB test missed the seasonal unit root of a simulated airline series. The `rayon` feature parallelises the exhaustive search only, since the stepwise search is sequential by design.

Phase 6 (benchmarks and the proof of the edge) is done. `ci/compare` measures fit time and the exact likelihood each crate's estimates reach, against foresight 0.7.3 and anofox-forecast 0.21.0: backshift reaches the same maximum as foresight 5–29x faster, and a higher likelihood than anofox-forecast on every model; anofox-forecast stays faster on the seasonal fit. In fully automatic searches all three agree on non-seasonal series, while on a simulated airline series only backshift recovers the generating model; foresight and anofox-forecast pick (0,2,2)(1,0,1,12). The embedded side is the 30 KB firmware check, the counting-allocator test and the `size_of` table in the README. QEMU cannot count cycles reliably, so there are no cycle figures. With that, every phase of the plan is built; the research ideas below remain.

Phase 1 is complete with step 6. Two conventions differ from statsmodels' defaults on purpose and are documented in the code: the likelihood for `d > 0` uses simple differencing (as R does), and BIC and AICc count only non-missing values (as R does).

Reference vectors come from `scripts/generate_reference.py`. Every quoted number in this document must be regenerated from the series in that script before a test uses it.

## 0. Summary

**Lead edge:** this crate would be the only ARIMA crate on crates.io that compiles for `no_std` bare-metal targets. It
would also be the only one that can carry a fitted model forward one observation at a time, in fixed memory with no heap
allocation, and produce the same innovations, forecasts and intervals as statsmodels `results.extend()`. That is one
claim with one proof: a CI job that builds for `thumbv7em-none-eabihf`, plus streaming tests against statsmodels vectors.

**Why not the other candidate edges:**

- Zero dependencies: `foresight` already has it, along with exact-MLE SARIMA, regressors, auto-ARIMA and R-checked tests.
- WASM: `foresight`, `x13` and `tsorder-core` all build for `wasm32-unknown-unknown`, and `anofox-forecast` ships an npm WASM package.
- Reference validation: `foresight` and `anofox-forecast` check against R `forecast`, and `inferust` checks against statsmodels.
- Tiny auto-ARIMA: `foresight` already ships a zero-dependency stepwise Hyndman–Khandakar search.

Every one of these is now table stakes, not a differentiator.

**Phase 1** is the smallest release that is a usable ARIMA and already owns the lead edge:

- `no_std` + `alloc` support.
- Differencing.
- AR fitting by Yule–Walker and Burg.
- ARMA fitting by Hannan–Rissanen, refined by exact Gaussian MLE through a Kalman filter and an in-crate BFGS.
- Forecasts with prediction intervals at any level.
- Log-likelihood, AIC, AICc and BIC.
- A heap-free streaming `Filter`.

## 1. Comparison with other ARIMA crates

Dependency counts are unique transitive normal dependencies with default features, excluding the crate itself.
"no_std" means it builds for `thumbv7em-none-eabihf`, which was tested only for the four smallest crates. All other
crates pull `std`-only dependencies (rayon, chrono, ndarray, nalgebra, faer). "wasm32" was tested locally where marked
(tested). Feature cells come from reading the public API and module docs of each ARIMA module. Cells marked "kw" rest
on keyword hits only and are unverified.

| Crate (version) | SARIMA | Exog | Auto order | Estimation | Pred. intervals | Diagnostics | Deps | no_std | wasm32 | Reference checks | Activity | License |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `arima` 0.3.0 (krfricke) | no | no | `autofit(x, d)`: orders from ACF/PACF significance bands, not an information criterion | CSS minimised by L-BFGS (`liblbfgs`, a pure-Rust port with no C build) | no (`arima_forecast` simulates paths with a user noise fn) | none | 31 | no | **fails** (getrandom without `js`) (tested) | none found | last push 2023-11-22, 40 stars | Apache-2.0 (crates.io says "non-standard") |
| `foresight` 0.7.3 | yes | yes (`Regressors`, regression with ARIMA errors) | yes, stepwise H–K, KPSS for d | exact MLE (innovations algorithm, B&D §5.2/§8.7), Nelder–Mead over PACF reparam | analytic `forecast_variance` (ψ-weights) plus empirical backtest intervals | KPSS, ndiffs, seasonal strength, ACF | **0** | no (`std::thread`) (tested) | yes (tested) | R `forecast` 9.0.2 and `prophet` 1.1.7 (`tests/against_r.rs`: Arima ML, auto.arima orders, Box-Cox) | created 2026-09-30, 256 downloads, 0 stars, pushed 2026-11-13 | MIT |
| `anofox-forecast` 0.21.0 | yes | yes (OLS residuals then ARIMA) | yes | faithful port of R `stats::arima` CSS-ML with R's BFGS (`r_arima.rs`) | yes (R-referenced) | Ljung–Box, ADF/KPSS and many more | 77 | no | npm WASM package (`js` feature) | R `forecast`, statsforecast, Nixtla (many test files) | very active (76 versions, pushed 2026-11-19), 11 stars | MIT |
| `chronos-ts` 0.1.0 | yes | yes (SARIMAX) | yes, stepwise or grid | CSS default, Kalman MLE option, `argmin` | yes (MA(∞) variance) | Ljung–Box, Jarque–Bera, ADF | 56 | no | not tested (rayon) | coefficient sanity tests only | one release 2026-08-20, 0 stars | MIT |
| `scirs2-series` 0.6.5 | yes (`SarimaModel`) | regression with ARIMA errors | `auto_arima`, `auto_sarima` | kw: CSS/MLE | `forecast_with_confidence` | kw: Ljung–Box | 108 | no | `wasm` feature | kw: 2 statsmodels mentions | active (39 versions, scirs repo 310 stars) | Apache-2.0 |
| `inferust` 0.8.2 | kw | kw | no | CSS default, Kalman exact MLE option | kw | Ljung–Box, ADF, KPSS | 28 | no | not tested | statsmodels parity for ARIMA(1,0,0) and (1,0,1) (`tests/parity_timeseries.rs`) | pushed 2026-09-06, 2 stars | MIT |
| `greeners-timeseries` 2.0.1 | yes (`fit_sarimax`) | yes | no | `fit_mle` | `predict_with_ci` | `ljung_box` | 113 | no | not tested | kw | pushed 2026-08-28 | MIT |
| `oxidiviner` 1.2.0 | yes (`SARIMAModel`) | no | `auto_select` across models | kw | kw | AIC/BIC | 51 | no | not tested | none found | last release 2025-06-06 (stale) | MIT |
| `solow-tsa` 0.7.4 | no | no | `arma_order_select_ic` grid | Kalman exact MLE (mirrors statsmodels ARIMA trend="c") | kw | Ljung–Box | 17 | no | not tested | modelled on statsmodels | pushed 2026-09-06 | BSD-3-Clause |
| `so-tsa` 0.3.0 | yes | kw | kw | CSS / ML / exact ML enum | kw | kw | not measured | no | not tested | kw | single release 2026-02-27, no repo | MIT AND Apache-2.0 |
| `rusty-series` 0.1.0 | no | no | no | `fit_ar` / `fit_arma`, method unclear | no | kw | not measured | no | not tested | none | 27 downloads | MIT OR Apache-2.0 |
| `x13` 0.1.0 | regARIMA inside X-13 | yes | yes (X-13 automdl) | X-13 regARIMA | yes | X-13 diagnostics | 6 (`thiserror`) | no (tested) | yes (tested) | Census X-13 Build 62 on ~500 FRED series | created 2026-11-19 | MIT OR Apache-2.0 |
| `tsorder-core` 0.1.0 | n/a (order selection only) | n/a | AIC/AICc/BIC/HQ/FPE, Fine–Hwang | YW, Burg, H–R, innovations | n/a | n/a | **0** | no (tested; its docs only claim "no_std-friendly in spirit") | yes (tested) | none found | pushed 2026-11-14 | MIT |
| `augurs` 0.10.3 (Grafana) | **no ARIMA at all** (ETS, MSTL, Prophet, outliers, clustering) | – | – | – | – | – | 29–42 per sub-crate | no | JS/WASM bindings | – | very active, 583 stars | MIT OR Apache-2.0 |

**Notes on the table:**

**API ergonomics**, from the public signatures and README examples:

| Crate | Input type | Style | What `fit` gives back | Friction |
|---|---|---|---|---|
| `arima` | `&[T]` with `T: Float + From<u32> + From<f64> + …` | free functions | raw `Vec<f64>` of coefficients, `anyhow::Result` | no model object, heavy generic bounds, no forecast with intervals |
| `foresight` | `Series::monthly(&values, start)` (borrowed) | builder `Arima::new(p,d,q).seasonal(..)`, then `Model::fit` | `Fitted` trait object with `forecast(h)` | best of the field; `Vec` everywhere |
| `anofox-forecast` | `TimeSeries` with chrono timestamps | `Forecaster` trait, builders | fitted model on `&mut self` | large API surface, chrono is mandatory |
| `chronos-ts` | `ndarray::Array1<f64>` | `SarimaModel::fit(data, order)` | `SarimaModel` | history must be passed again to `forecast` |
| `scirs2-series` | `ArrayBase<S, Ix1>`, generic `F` | `&mut self` `fit` | mutated model | history must be passed again to `forecast`; ndarray required |
| `inferust` | `&[f64]` | builder (`.exact_mle()`) | `ArimaResult` with statsmodels-style summary | – |
| `greeners-timeseries` | ndarray | unit struct `ARIMA::fit(...)` | `ArimaResult` | 113 dependencies |
| `oxidiviner` | `TimeSeriesData` with chrono timestamps | `&mut self` `fit` | mutated model | stale |
| `x13` | `Series::new(Period, &values)` | `Spec` constructors | adjustment components | seasonal adjustment only, not a forecasting API |

- `temporalseries` 0.1.3 has no ARIMA, so it is omitted. `autoregressive` 0.1.6 only simulates AR series, and `arma-rs` is an unrelated game-extension crate.
- `foresight` and `tsorder-core` come from the same lab (castlab) and show the same "zero dependencies" philosophy. Expect them to keep moving.
- Streaming or online updates:
  - `anofox-forecast` has `IncrementalForecaster::update` with fixed parameters, but only for SES, Holt, Holt-Winters and ETS. It does **not** cover ARIMA.
  - `scirs2-series::online::OnlineArima` is a recursive-least-squares AR approximation with a forgetting factor. It is not exact ARIMA filtering, and its docs admit `std::collections` use.
  - `oxidiviner` has an "adaptive forecaster" `update` that refits.
  - No crate offers exact fixed-parameter ARIMA state propagation.
- None of the 13 inspected ARIMA crates builds `no_std`. The probe built each crate as a dependency of an empty lib for `thumbv7em-none-eabihf`. `foresight`, `tsorder-core`, `x13` and `arima` all fail with `E0463: can't find crate for std`.
- Demand signal: `arima` (krfricke) is unmaintained since 2023, yet it shows 54,116 of its 78,107 downloads as "recent" (last 90 days). Something still pulls it. Its reverse-dependency list could not be fetched because crates.io returned HTTP 429.

## 2. Gaps and the choice of lead edge

Each candidate edge was weighed against the evidence:

| Candidate | Verdict | Evidence |
|---|---|---|
| Pure Rust, zero dependencies | Rejected as a lead (keep as a property) | `foresight` and `tsorder-core` already have 0 deps. `foresight` is full-featured. |
| WASM | Rejected as a lead (keep as a CI target) | `foresight`, `x13` and `tsorder-core` build for wasm32. `anofox` ships npm WASM. |
| Proven numerics against statsmodels/R | Rejected as a lead (do it anyway, it is table stakes) | `foresight` and `anofox` validate against R. `inferust` validates against statsmodels, but only for two tiny ARIMA cases. A statsmodels-parity suite is a secondary selling point for Python porters, not a headline. |
| Tiny auto-ARIMA | Rejected | `foresight` `AutoArima` is zero-dependency and matches R orders. |
| `no_std` | **Unique among the 13 ARIMA crates inspected** | None of the four smallest builds bare-metal (tested). The others need `std`-only dependencies. Caveat: generic `no_std` Kalman crates exist (`kfilter`, `typed-kalman`, `ukalman`). A user could hand-build an ARMA state space on top of one, but they would get no estimation, no stationary initialisation and no statsmodels parity. |
| Exact streaming update at fixed parameters | **Unique for ARIMA** | `anofox` covers ETS only. `scirs2` is an RLS approximation. Others refit. |
| Allocation-free hot path | Unique only together with the two above | Nobody advertises it. It only matters to someone who also needs `no_std` or a hard real-time budget. |

**Chosen lead edge:** ARIMA for embedded and edge devices. The crate is `no_std` (with `alloc` for fitting) and has a
heap-free fixed-capacity streaming filter whose per-sample update and forecast match statsmodels `extend()`.

The use case is: fit offline or at boot, then run `update(y)` at sample rate on a microcontroller, PLC, wasm worker or
real-time thread. It yields one-step innovations for anomaly detection, plus h-step forecasts with intervals.

**Why `foresight` cannot copy this by adding `#![no_std]`:**

- Its exact likelihood uses the innovations algorithm over the full history (`arma.rs`). It has no state object that can be carried forward, so streaming needs a different filter design.
- Its whole API is built around `Vec`-returning `Series` and `Model`/`Fitted` traits.
- Its backtest and intervals rely on `std::thread::scope`.

Our core *is* the state-space filter. Batch likelihood, forecasting and streaming are the same code. `no_std` and
heap-free updates fall out of that design rather than being retrofitted.

Be honest about which part is defensible. `no_std` + `alloc` alone is cheap for `foresight` to copy: it would
feature-gate its threads and add `libm`. The part that cannot be copied by adding an attribute is the **heap-free
filter**. It is built from coefficients alone, so a model fitted offline (on a PC, from statsmodels or R) runs on a
device with no allocator at all. Lead with that.

**Main risk of this choice: the niche may be small.** Embedded forecasting demand is unproven. The mitigation is that
phase 1 is also a correct, statsmodels-checked, light general ARIMA, so desktop users lose nothing.

## 3. Roadmap

### Conventions fixed before writing any test

These were checked locally with statsmodels 0.14.0:

- **Difference first, then fit ARMA on the differenced series.**
  - `SARIMAX(y, order=(1,1,1), simple_differencing=True)` and `ARIMA(np.diff(y), order=(1,0,1))` gave identical params and log-likelihood (`-351.78243271333855` both).
  - The default `ARIMA(y, order=(1,1,1))` (diffuse state for the integrated part) gives llf `-423.513337` with `nobs=300`. ARMA on `np.diff(y)` gives `-423.513354` with `nobs=299`. They differ by 1.7e-5, and the AIC differs too.
  - So reference vectors must come from `SARIMAX(..., simple_differencing=True)` or from ARMA on the pre-differenced series. Document that `nobs = n - d - D*s`, like R's `arima`.
- **The constant means the mean, not the intercept.** `ARIMA(x, order=(1,0,1), trend='c')` returns `const = 5.10` for a series with mean ≈ 5. With d=1, `trend='t'` is the drift. Match those meanings.
- **AIC counts σ² as a parameter.** ARMA(1,1): AIC 853.027 = 2·423.513 + 2·3, the same as R `arima`.
- **Verification tiers.** The local run showed statsmodels `statespace` and `innovations_mle` optimizers agreeing only to about 1e-5 on params. Therefore:
  - **Tier A (1e-9 relative):** deterministic quantities at *fixed reference parameters*. These are log-likelihood, residuals/innovations, forecasts, forecast variances, Yule–Walker, Burg, Hannan–Rissanen, differencing, and Ljung–Box at given residuals. Example: statsmodels `model.loglike([0.6, 0.3, 1.0])` = `-424.6940795311283`.
  - **Tier B (1e-4 on params, 1e-6 relative on llf):** optimised estimates. Our llf at our optimum must also be ≥ statsmodels' llf minus 1e-6, so we never converge to a worse point.
  - **Streaming** is compared against `results.extend(new_obs)` at fixed params. It is never compared against a refit: the local run gave extend forecasts `[0.4728, 0.3210, 0.2180]` against refit `[0.4511, 0.3081, 0.2105]`.
- **Reference generation.** `scripts/generate_reference.py` writes checked-in Rust `const` arrays, keeping dev-dependencies minimal. It records the statsmodels version and seed in a header comment. Do not regenerate in CI.

### Phase 1: usable ARIMA that owns the edge (target 0.1 under the new name)

**Scope:**

1. **Crate setup.**
   - Add `#![no_std]` with `extern crate alloc`, a default `std` feature, and an optional `libm` dependency used when `std` is off.
   - This is required because on rustc 1.100 `f64::sqrt`, `ln`, `exp`, `tanh`, `powi` and `mul_add` do **not** resolve in `no_std` core (local probe: `E0599`). Even `powi` fails.
   - Keep one private `math` module with `sqrt`, `ln`, `exp`, `tanh`, `fma` and `powi` wrappers under `cfg(feature = "std")`.
   - `libm` has no `powi`. Write it as a small square-and-multiply loop, or use `libm::pow` for non-integer powers. `mul_add` maps to `libm::fma`.
   - Default build: 0 deps. `no_std` build: 1 dep (`libm`, which has no deps of its own).
   - Port the existing code: `std::cell::Cell` becomes `core::cell::Cell`. `std::error::Error` becomes `core::error::Error`, so MSRV is 1.81. `TimeSeries::std()` routes through `math::sqrt`.
   - CI: `cargo build --no-default-features --target thumbv7em-none-eabihf`, plus `wasm32-unknown-unknown`.
2. **Differencing.** `difference(x, d, D, s)` and `integrate` for forecasts. Reference: Box, Jenkins, Reinsel & Ljung (2015) §4.1.
3. **AR estimation.**
   - Yule–Walker reuses the existing Durbin–Levinson (biased acov, matching statsmodels `yule_walker(method="mle")`).
   - Burg follows Brockwell & Davis (2016, *Introduction to Time Series and Forecasting*) §5.1.2 and statsmodels `regression/linear_model.py::burg`.
   - Local reference values for an AR(2) fit on the test series: YW `[0.85349694, -0.12325685]`, Burg `[0.8540284, -0.1215598]`.
4. **ARMA start values: Hannan–Rissanen.** Long AR by Yule–Walker, then OLS of y on lagged y and lagged residuals. Reference: Hannan & Rissanen (1982), and statsmodels `tsa/arima/estimators/hannan_rissanen.py`. The OLS goes through an in-crate Cholesky or QR solve, about 60 lines.
5. **Exact Gaussian MLE.**
   - State-space model in Harvey form, dimension `r = max(p, q+1)`. Run the Kalman filter with the σ²-concentrated likelihood.
   - Get the initial covariance by solving the discrete Lyapunov equation with the doubling algorithm, about 30 lines, capped at 64 iterations.
   - References: Durbin & Koopman (2012) §3.4 and §7.2; Gardner, Harvey & Phillips (1980) AS154; statsmodels `tsa/statespace/tools.py::solve_discrete_lyapunov`.
   - Optimise over unconstrained parameters through the PACF → Durbin–Levinson map (Monahan 1984, Jones 1980). This keeps AR stationary and MA invertible. See statsmodels `tools.py::constrain_stationary_univariate`.
   - Optimiser: in-crate BFGS with central-difference gradients and a backtracking Armijo line search, about 150 lines (Nocedal & Wright 2006, Alg. 6.1). Fall back to Nelder–Mead only if BFGS fails in testing.
6. **Mean and drift** are estimated jointly as an extra parameter, using the statsmodels and R meanings above.
7. **Forecasts and intervals.**
   - The h-step Kalman prediction gives the mean and variance. Map them back through `integrate`. The variance accumulates through the integration polynomial, which is equivalent to ψ-weights of the full ARIMA (B&D §6.4).
   - Intervals at any level use the normal quantile from Wichura's AS241 `PPND16` (about 50 lines, rational approximation, accurate to 1e-16).
8. **Fit statistics.** Log-likelihood, σ², residuals, AIC, AICc and BIC, with statsmodels definitions.
9. **Streaming `Filter`.**
   - Fixed capacity, no heap, `Copy`-able, built from a fitted model.
   - It holds the state vector, the state covariance and a ring buffer of the last `d + D*s` raw values.
   - It carries the covariance (not just the steady-state gain) so it matches statsmodels `extend` exactly.
   - It has two constructors:
     - `Filter::new(&FilterSpec)` builds it from coefficients alone, heap-free. It computes the stationary initial covariance with the same doubling algorithm on fixed `[f64; R*R]` arrays. This is the "fit offline, run on the device" path.
     - `Fitted::filter()` continues from the end of the fitted sample.

**API sketch** (signatures only; the filter is the one generic type):

```rust
pub struct Order { pub p: usize, pub d: usize, pub q: usize }

pub enum Trend { None, Constant /* mean if d == 0, drift if d == 1 */ }

pub fn difference(x: &[f64], d: usize) -> Vec<f64>;
pub fn yule_walker(x: &[f64], p: usize) -> Option<ArCoefficients>;   // phi, sigma2
pub fn burg(x: &[f64], p: usize) -> Option<ArCoefficients>;

pub struct Arima { order: Order, trend: Trend }
impl Arima {
    pub fn new(p: usize, d: usize, q: usize) -> Self;
    pub fn trend(self, trend: Trend) -> Self;
    pub fn fit(&self, y: &[f64]) -> Result<Fitted, FitError>;                // needs alloc
    pub fn loglike(&self, y: &[f64], params: &Params) -> Result<f64, FitError>; // tier-A test hook
}

pub struct Params { pub ar: Vec<f64>, pub ma: Vec<f64>, pub trend: Option<f64>, pub sigma2: f64 }

pub struct Fitted { /* params, llf, nobs, residuals, final filter state */ }
impl Fitted {
    pub fn params(&self) -> &Params;
    pub fn loglike(&self) -> f64;
    pub fn aic(&self) -> f64;  pub fn aicc(&self) -> f64;  pub fn bic(&self) -> f64;
    pub fn residuals(&self) -> &[f64];
    pub fn forecast(&self, h: usize, level: f64) -> Forecast;              // mean, lower, upper
    pub fn filter<const R: usize>(&self) -> Result<Filter<R>, CapacityError>;
}

/// Everything a device needs; borrowed slices, no allocation.
pub struct FilterSpec<'a> { pub ar: &'a [f64], pub ma: &'a [f64], pub d: usize, pub trend: f64, pub sigma2: f64 }

/// Heap-free. R bounds max(p, q + 1) + d; returns CapacityError if too small.
#[derive(Clone, Copy)]
pub struct Filter<const R: usize> { /* [f64; R], [[f64; R]; R], ring buffer, params */ }
impl<const R: usize> Filter<R> {
    pub fn new(spec: &FilterSpec<'_>) -> Result<Self, FilterError>;        // no alloc: offline-fit path
    pub fn warm_up(&mut self, history: &[f64]);                            // = update over history
    pub fn update(&mut self, y: f64) -> Innovation;                         // { value, variance }
    pub fn forecast_into(&self, mean: &mut [f64], variance: &mut [f64]);
}
```

**Verification for phase 1:**

- Tier A: llf, residuals, forecasts and variances at fixed params for ARIMA(1,0,0), (0,0,1), (2,0,1), (1,1,1) and (0,1,1)+drift, from `SARIMAX(..., simple_differencing=True).loglike / .filter / get_forecast`. Also YW and Burg (`statsmodels.regression.linear_model.yule_walker/burg`) and Hannan–Rissanen (`ARIMA(...).fit(method="hannan_rissanen")`).
- Tier B: `ARIMA(...).fit()` params and llf on 3 or 4 simulated series plus one public series (sunspots, from `statsmodels.datasets`). Assert our llf ≥ statsmodels llf − 1e-6.
- Streaming on the differenced scale (checked locally):
  - Recipe: `res = ARIMA(np.diff(y[:250]), order, trend='n').fit()`, then `res.extend(np.diff(y)[249:]).get_forecast(h)`.
  - Feeding the raw chunk into `SARIMAX(y[:250], simple_differencing=True).fit().extend(y[250:])` gave the same forecasts `[0.47237887, 0.32019304, 0.21703676]`, but with `nobs` 49 instead of 50. Use the pre-differenced recipe so the boundary difference `y[250]-y[249]` is explicit.
  - Compare innovations, mean and `var_pred_mean` at 1e-9.
- Streaming on the original scale (checked locally):
  - Recipe: `SARIMAX(y, order=(p,1,q), trend='n').filter(params).get_forecast(h)`, with `params` taken from the ARMA-on-diff fit.
  - It reproduced `y[-1] + cumsum(diff forecast)` and the integrated ψ-weight variance exactly. For ARIMA(1,1,1) on the test series: forecasts `[36.1411992, 36.44850461, 36.65806286]` and variances `[0.99198702, 4.46209851, 10.48391644]`.
  - This is the reference for original-scale intervals.
- Offline-fit path: build `Filter::new(&FilterSpec { .. })` from statsmodels coefficients, run `warm_up(y)`, and compare against `SARIMAX(...).filter(params)` (tier A).
- Build proof:
  - CI builds for `thumbv7em-none-eabihf` and wasm32.
  - A unit test asserts `size_of::<Filter<8>>()` stays small.
  - One integration test installs a `#[global_allocator]` that panics. It then runs only `Filter::new`, `update` and `forecast_into`, which proves "no heap" for the whole device path.

**Rough size:** about 1,500–2,000 lines of library code on top of the existing 550, plus about 600 lines of tests and
generated vectors. Expect two to four focused weeks.

### Phase 2: seasonal ARIMA

- **API.** `Arima::seasonal(self, p: usize, d: usize, q: usize, s: usize) -> Self`. `difference` gains `D` and `s`.
- **Algorithm.** Multiply the seasonal and non-seasonal AR/MA polynomials into one ARMA of order `p + P·s` and `q + Q·s`, then reuse the phase-1 filter unchanged. Reference: Box et al. (2015) ch. 9, and statsmodels `sarimax.py` polynomial expansion.
- **Lyapunov cost.** With s = 12 the state dimension reaches about 26, so doubling costs about 26³ per iteration. That is acceptable. If it is too slow, switch to the Gardner AS154 initialisation or start from the stationary autocovariances.
- **Embedded cost.** `Filter<R>` grows to R ≈ 26–50 (covariance 5–20 KB). Document this, and offer an optional steady-state-gain mode (`Filter::steady`) that drops the covariance once it has converged. Steady state loses exact `extend` parity, so it is opt-in.
- **Verification.** Airline model `SARIMAX(log airpass, (0,1,1), (0,1,1,12), simple_differencing=True)` at tier A and tier B. This also cross-checks `foresight`'s R-matched airline numbers.
- **Size:** about 400 lines.

### Phase 3: exogenous regressors (regression with ARIMA errors)

- **API.** `Arima::fit_exog(&self, y: &[f64], x: &Matrix<'_>) -> Result<Fitted, FitError>`, where `Matrix` is a borrowed row-major `&[f64]` with a column count. `Fitted::forecast_exog(h, x_future, level)`. `Filter::update_exog(y, x_row)`.
- **Algorithm.** Put β into the optimiser vector, starting from OLS. The likelihood runs on `y - Xβ`, differenced, with differencing applied to X as well. This matches statsmodels `SARIMAX(exog=..., simple_differencing=True)` and R `arima(xreg=)`.
- **Verification.** Tier A/B against SARIMAX with exog.
- **Size:** about 250 lines.

### Phase 4: diagnostics

- **API.** `ljung_box(resid: &[f64], lags: usize, df_adjust: usize) -> Test`, where `Test` holds the statistic and p-value. Also `jarque_bera`, `kpss(x, Trend)` and `ndiffs`.
- **Algorithm.** The χ² survival function needs a regularised incomplete gamma: series plus continued fraction, about 60 lines (Numerical Recipes 3e §6.2, or Cephes `igamc`). The KPSS p-value uses the statsmodels lookup table with interpolation (`stattools.py::kpss`).
- **Verification.** `acorr_ljungbox`, `jarque_bera` and `kpss`, all tier A. A local Ljung–Box reference: lag 10 stat `12.69814`, p `0.241042`.
- **Size:** about 300 lines.

### Phase 5: auto-ARIMA (`alloc` only)

- **API.** `AutoArima::new().max_order(..).seasonal(s).criterion(Criterion::Aicc).fit(y) -> Result<Fitted, FitError>`.
- **Algorithm.** Hyndman & Khandakar (2008) stepwise search, with KPSS for `d` and seasonal strength or OCSB for `D`. Add a CSS pre-fit for speed during the search (R `armaCSS`), keeping exact MLE for the final model.
- **Verification problem.** statsmodels has no auto-ARIMA, and neither R nor `pmdarima` is installed locally. Generate R `forecast::auto.arima` order picks once on another machine or in CI (a Docker `r-base` job), check them in, and assert identical orders. `foresight` and `anofox` already set that bar.
- **Size:** about 400 lines.

### Phase 6: benchmarks and proof of the edge

- **Benchmarks.**
  - Use `criterion` as a dev-dependency only.
  - Compare fit time and `update` throughput against `foresight`, `anofox-forecast`, `chronos-ts` and `augurs`. `augurs` has no ARIMA, so compare it on ETS only, or skip it.
  - Use identical series and orders, and report llf reached as well as time.
  - Add an embedded table: `.text` size and stack use of `Filter::update` on `thumbv7em`, and cycles per update under QEMU or on a real board.
- **README headline:**
  - The bare-metal build badge.
  - The statsmodels parity table, with tiers spelled out.
  - A copy-paste MCU example.
- **Size:** about 300 lines of benches and docs.

### Phase 2+ research ideas (arXiv notes, not yet verified)

Source: research notes from a separate session. They come from abstracts and search snippets only, not the full papers. Read each paper before committing to it.

- **Conformal prediction intervals (recommended right after phase 1).**
  - **Why:** a model-agnostic wrapper that gives valid coverage when the innovations are not Gaussian. It pairs with `Filter::forecast_into` and sharpens the pitch: streaming forecasts with valid intervals, without std.
  - **Shape:** a rolling buffer of absolute h-step residuals, with the quantile taken as the interval half-width. A fixed-size const-generic buffer keeps it heap-free.
  - **References:** ARIMA-based benchmark https://arxiv.org/abs/2601.18509, online multi-step https://arxiv.org/abs/2410.13115, EnbPI https://arxiv.org/abs/2010.09107.
  - **Verification:** coverage tests on simulated series with skewed and heavy-tailed noise. No statsmodels parity is possible.
  - **Done (#40):** `Conformal<H, N>`; the papers were read first. Coverage 0.90 at level 0.9 under Gaussian, skewed and t3 noise.
- **Online ARIMA by gradient or Newton steps (a second mode next to `Filter`, not a replacement).**
  - **What:** Anava et al., ICML 2013, "Online learning for time series prediction". The citation is from memory and unchecked.
  - **How it differs:** it learns an approximating AR(m+k) model one sample at a time instead of filtering exact MLE coefficients. It handles drifting parameters, but it gives up the 1e-9 `extend()` parity.
  - **Pitch:** offer it as an adaptive option (`OnlineAr<const M>`) and keep exact `Filter` as the headline.
  - **Done (#42):** `OnlineAr<M>` (ARMA-ONS, arXiv:1302.6927, read first; the venue is unconfirmed). It beats a stale filter on drift and switches, and a periodic refit after a switch.
- **PMM2 estimator for skewed innovations (later differentiator).**
  - **What:** reported big gains under moderate asymmetry with N >= 200. No Rust crate is known to have it.
  - **Fit:** add it as an alternative `Method` after exact MLE exists, so it can be compared against MLE on the same reference series.
  - **Reference:** https://arxiv.org/abs/2511.07059.
  - **Done (#43):** the paper was read and the gain reproduced against exact MLE (not only CSS): 1.4-2x lower MSE for AR coefficients with Gamma(2,1) noise, none lost with Gaussian noise. Shipped as `Arima::fit_pmm2`; MA terms use the paper's linearised design and gain less.
- **Weak fit, deferred:**
  - **Galerkin-ARIMA** (https://arxiv.org/abs/2507.07469): splines replace the AR term and the fit is closed-form 2SLS with no optimizer. It is not standard ARIMA, so it cannot be checked against statsmodels.
  - **Distributed ARIMA** (https://arxiv.org/abs/2007.09577): split, fit in parallel, combine. It needs rayon, so it only works behind an optional `std` feature and conflicts with the lead edge.
  - **Bayesian order selection by nested sampling** (https://arxiv.org/abs/2512.01929): heavy, and it does not fit a lean crate. AICc stays the default.

## 4. Crate names

Free names, per the crates.io API returning 404 on 2026-11-22. crates.io treats `-` and `_` as the same, so the
underscore variants are also free.

- **Recommended:**
  - `backshift`: the B operator of Box–Jenkins. Short, memorable, and does not box the crate into ARIMA only.
  - `sarima`: best search-engine match.
- **Also free:**
  - Descriptive: `arima-core`, `arima-rs`, `arimax`, `sarimax`, `arma`, `armax`, `boxjenkins`, `box-jenkins`.
  - Edge-signalling: `tinyarima`, `tiny-arima`, `microarima`, `micro-arima`, `arima-lite`, `nostd-arima`, `arima-nostd`, `embedded-arima`, `arima-embedded`, `edge-arima`.
  - Streaming-signalling: `arima-stream`, `stream-arima`, `streaming-arima`, `arima-online`.
  - Other: `arimalib`, `arima-kit`, `lagwise`, `innovations`, `ararma`.
- **Taken:** `pdq`, `armature`, `autoregressive`, `arma-rs`.

Re-check the chosen name right before `cargo publish`, because names can be claimed at any time.

## 5. Risks

1. **The niche may be small (demand risk).**
   - No hard evidence shows embedded users want ARIMA. The large "recent" download count of the stale `arima` crate is unexplained, because the reverse dependencies hit HTTP 429.
   - Mitigation: phase 1 is also a clean general-purpose ARIMA, and the edge costs little (one `cfg` module plus one const-generic type).
   - Before investing past phase 1, check reverse dependencies of `arima` and search embedded and IoT forums or issues.
2. **Numerical stability of exact MLE.**
   - Near-unit-root AR and near-non-invertible MA make the Lyapunov solution ill-conditioned, and doubling converges slowly when the spectral radius is close to 1.
   - The concentrated likelihood is flat near the boundary.
   - Mitigations:
     - Use the PACF reparameterisation, which keeps the AR stationary and the MA invertible.
     - Cap the doubling iterations, and return an error rather than a silent bad value.
     - Clamp the transform, as statsmodels does (`|pacf| < 1 - 1e-8`).
     - Symmetrise P after every Kalman step and use the Joseph form.
     - Test explicitly at φ = 0.99 and θ = −0.98.
   - Seasonal state dimensions (r ≈ 26–50) amplify rounding. Watch tier-A tolerances there and loosen them to 1e-8 only with a written reason.
3. **The optimiser without a heavy dependency.**
   - Only the MLE refinement needs an optimiser, plus the CSS pre-fit in auto-ARIMA. Yule–Walker, Burg, Hannan–Rissanen, filtering, forecasting and diagnostics are closed-form or recursive.
   - Evidence that an in-crate optimiser is enough:
     - `foresight` reaches R-matching ARIMA ML fits with a 179-line Nelder–Mead (`src/optimize.rs`) over a PACF reparameterisation.
     - `anofox-forecast` ports R's `optim` BFGS (`vmmin`) with central differences to reproduce R's trajectory.
   - The risk is convergence to a worse local optimum than statsmodels. The tier-B check ("our llf ≥ statsmodels llf − 1e-6") catches it.
   - Starting from Hannan–Rissanen, rather than from zeros, is the main defence.
4. **`no_std` math.**
   - Because the `core` float methods are not available (tested), the `no_std` build depends on `libm`.
   - `libm` results can differ from the platform `libm` in the last ulp, so tier-A tests must run in both configurations.
   - Do not claim "zero dependencies" for the `no_std` build. Claim "zero dependencies by default; `no_std` needs only `libm`".
5. **Pace of other crates.** `foresight` (same zero-dep philosophy, R-validated) and `anofox-forecast` (76 releases, very active) ship features fast. A faster SARIMA/auto-ARIMA alone would not stand out next to them. Only the embedded and streaming design is defensible, so ship phase 1 before widening scope.

## 6. Could not verify

- **Reverse dependencies of `arima`.** crates.io returned HTTP 429 repeatedly, so I could not say who drives its 54k recent downloads.
- **Unverified feature cells.** Cells marked "kw" for `scirs2-series`, `inferust` (SARIMA, exog), `oxidiviner`, `so-tsa`, `solow-tsa` and `rusty-series` rest on keyword counts and module headers, not on running their code. Neither did I verify that their prediction intervals are analytically correct.
- **Untested builds.** The `wasm32` and `no_std` builds of `anofox-forecast`, `scirs2-series`, `chronos-ts` and other heavy crates were not compiled. Their `std`-only dependencies (rayon, chrono, ndarray, faer, nalgebra with std) make `no_std` very unlikely.
- **R checks.** R `forecast::auto.arima` reference orders could not be generated locally, because neither R nor `pmdarima` is installed.
- **Full `no_std` sweep.** A crates.io-wide sweep for `no_std` ARIMA crates was not possible beyond keyword searches. "Unique" means unique among the 13 ARIMA crates inspected plus the generic Kalman crates found by search.
- **No other crate ran.** I did not run the other crates' test suites or benchmarks. Claims of "matches R" come from their test files and READMEs.

## 7. Beyond ARIMA: which model families, in which order

Written 2026-11-24, after the speed work in #67 and #68 put backshift first in its own lane. foresight and
anofox-forecast cover more ground: foresight 0.7.3 ships ETS, Holt–Winters, Theta, TBATS, Croston, a Prophet-style
model, ensembles and backtesting; anofox-forecast 0.21.0 adds GARCH, VAR, MFLES, intermittent-demand models, anomaly and
changepoint detection and hierarchical reconciliation, and updates its exponential-smoothing models incrementally.

### The bar a family has to clear

A family is worth adding only where backshift can bring the same edge it has for ARIMA, not merely be present:

1. **Exact and checked.** The likelihood or estimator of a reference implementation (R `forecast`, statsmodels, or
   Nixtla's statsforecast), matched to stated tolerances in a conformance suite like the ARIMA one.
2. **Streams heap-free.** A fitted model carries forward one value at a time in a `Copy` struct, as `Filter<R, M>`
   does, so it runs on a device and feeds `Conformal` intervals. No competitor streams on bare metal.
3. **Zero dependencies, `no_std`.** The offline fit may allocate; the device path may not.
4. **Not already covered.** If an existing combination does the job, document the recipe instead of adding a model.

### The families, ranked

| Family | Verdict | Why |
|---|---|---|
| **ETS** (Hyndman et al. 2008: SES, Holt, damped, Holt–Winters, additive and multiplicative) | **Add first** | The most used family after ARIMA, and R's `ets` is the reference everyone compares with. Its innovations state-space form is a recursion over a few states with no Kalman covariance, so the device filter is smaller and faster than ARIMA's and fits the `Filter` shape directly. Both friends have it. |
| **Theta** (Assimakopoulos & Nikolopoulos 2000; Hyndman & Billah 2003) | **Add, right after ETS** | SES with drift, so nearly free once ETS exists, and a strong M3/M4 benchmark. References: R `thetaf`, statsmodels `ThetaModel`. |
| **Intermittent demand** (Croston, SBA, TSB) | **Add** | Tiny estimators with obvious device use (counts of rare events, spare-part demand). References: statsforecast `CrostonClassic`, `CrostonSBA`, `TSB`. Streaming is a two-state update. |
| **Backtesting and cross-family selection** (rolling origin, MASE, sMAPE; ARIMA vs ETS vs Theta) | **Add, after ETS** | Information criteria do not compare across families (ETS and differenced ARIMA likelihoods are on different data), so choosing among them needs out-of-sample error, which is foresight's selling point. Infrastructure, not a model. |
| **GARCH(1,1)**, alone or as ARIMA errors | **Done** (variants: #76–#81) | A different question (volatility), but a natural streaming companion: the innovation variance per step is what anomaly scoring wants. Reference: R `rugarch`, Python `arch`. |
| **Streaming changepoint detection** (CUSUM, BOCPD on the innovations) | **Done** | Fits the device story with the innovations the filters already produce. anofox has batch detection only. |
| **TBATS** | **Skip** | Multiple and non-integer seasonality is covered by MSTL+ARIMA and by Fourier regressors with ARIMA errors, both already checked against statsmodels and R. TBATS is heavy, slow to fit, and has no exact-likelihood reference to match. Benchmark the existing recipes against it instead. |
| **Prophet-style** (piecewise trend, Fourier, holidays) | **Recipe, not a model** | Regression with ARIMA errors plus Fourier terms and holiday dummies already is this (see `examples/holidays_outages.rs`). At most a small helper for trend-changepoint regressors. |
| **VAR, VARX, online VAR** (lag order by AIC, BIC or HQ) | **Add, after ETS** | Least squares checked against statsmodels `VAR` and R `vars`. A heap-free `OnlineVar<K, P>` streams `K` sensors jointly and flags anomalies only their co-movement shows; neither friend streams a multivariate model, let alone on bare metal. VARX reuses the regressor handling of `fit_exog`. |
| **Hierarchical reconciliation (bottom-up, OLS, WLS, MinT), MFLES** | **Skip** | Reconciliation is a platform feature across many series, outside a forecasting crate; MFLES is anofox-specific with no external reference. ETS covers the ground MFLES would. |

### Shape of the code

Each family follows the ARIMA layout, so the edge carries over without a framework:

- a `…Spec` of coefficients (no allocation), validated like `ArimaSpec`;
- a heap-free `…Filter` built from the spec, with `update`, `forecast_into` and `forecast_interval_into`;
- an allocating fit (`Ets::new(…).fit(y)`, `AutoEts`) that reuses the in-crate BFGS, with bounded parameters mapped
  through transforms as the ARIMA fit does;
- a conformance generator in `scripts/` and data checked in under `tests/it/`.

A shared trait appears only when backtesting needs to run several families through one loop, and then holds exactly
what that loop calls (fit, forecast), nothing speculative.

The families are not feature-gated one by one. The linker drops what a binary does not call, so a firmware that
uses only `Filter` carries no ETS or VAR code; the split that matters for running everywhere is the one that exists,
heap-free filters without `alloc` and fits behind it. Each feature would multiply the CI matrix. A family gets a
feature only if it needs a dependency, as the parallel search does with `rayon`, or measurably slows a clean build.

### Order of work

1. **ETS**, in two pull requests: the additive-error models (linear, the bulk of real use) with `AutoEts` and the
   streaming filter; then multiplicative errors and seasonality. Conformance against R `forecast::ets` (likelihood,
   parameters, forecasts, intervals) and statsmodels `ETSModel`. Benchmark against foresight's and anofox's ETS.
2. **Theta** on top of ETS, checked against R `thetaf` and statsmodels `ThetaModel`. (Simple exponential smoothing,
   Holt and Holt–Winters are ETS(A,N,N), (A,A,N) and (A,A,A), so they arrive with step 1.)
3. **VAR, VARX and a heap-free online VAR** with automatic lag selection, checked against statsmodels `VAR` and
   R `vars`.
4. **Croston, SBA, TSB**, checked against statsforecast.
5. **Backtesting and cross-family selection.**
6. **GARCH, streaming changepoint detection and the Prophet-style recipe.**

Each step is done when its conformance suite passes, its filter passes the no-allocation test and the firmware link,
and `ci/compare` shows where it stands against both friends, in speed and in how close each gets to the reference.

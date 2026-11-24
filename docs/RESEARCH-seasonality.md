# Research: seasonality hints, period detection and multiple seasonality

Research date: 2026-11-22. Companion to [ROADMAP.md](ROADMAP.md), whose comparison table of other crates (section 1) is reused here and extended only for the features below.

Evidence sources: the crate's own source (cited as `file:line` at the current `HEAD`, 47376a8); R `forecast` master and `wch/r-source` trunk from raw.githubusercontent.com (fetched 2026-11-22; line numbers on master can drift); the forecast 8.2 CRAN archive; `tidyverts/fable`, `fabletools` and `feasts` master; the local statsmodels 0.14.0 install (cited by its GitHub `v0.14.0` path, the same file and line); statsforecast, coreforecast, pmdarima, sktime, Nixtla tsfeatures, R tsfeatures and Prophet at their default-branch HEAD on 2026-11-22; `.crate` files from static.crates.io (latest versions on 2026-11-22), extracted and read but never built; the papers named below (arXiv PDF for MSTL, JSS PDF for Hyndman and Khandakar, Crossref metadata for the TBATS and Prophet papers); FPP3 at otexts.com. No code from the other libraries was run.

URL prefixes used below:

- `F` = `https://raw.githubusercontent.com/robjhyndman/forecast/master/R/`
- `RS` = `https://raw.githubusercontent.com/wch/r-source/trunk/src/library/stats/`
- `SM` = `https://github.com/statsmodels/statsmodels/blob/v0.14.0/statsmodels/`
- `SF` = `https://github.com/Nixtla/statsforecast/blob/98ce313b089b6e3d29490d2764c9a366d7256d05/python/statsforecast/`
- `CF` = `https://github.com/Nixtla/coreforecast/blob/ae3e773224ceb3c8a5349868c159de63bea6de20/`
- `PM` = `https://github.com/alkaline-ml/pmdarima/blob/4c2dfccb28f64d2c00a5e10b59c1d1a3e16576a9/pmdarima/`
- `SK` = `https://github.com/sktime/sktime/blob/fe61df959f717efdccf3257ee031bfb0015b9930/sktime/`
- `PR` = `https://github.com/facebook/prophet/blob/5cf15c0f6e5d5259d9dd8d511fb4e15a45884cdc/python/prophet/forecaster.py`
- `FB` = `https://raw.githubusercontent.com/tidyverts/fable/master/R/`, `FT` = `https://raw.githubusercontent.com/tidyverts/fabletools/master/R/`, `FE` = `https://raw.githubusercontent.com/tidyverts/feasts/master/R/`
- `TSF` = `https://github.com/Nixtla/tsfeatures/blob/d459002dc44a8bb883998670dd1f4c239579b729/tsfeatures/`, `RTSF` = `https://github.com/robjhyndman/tsfeatures/blob/8d5208169465d34d520e4fbd555f301632ccd67e/R/`
- forecast NEWS: `https://raw.githubusercontent.com/robjhyndman/forecast/master/NEWS.md`; Hyndman and Khandakar (2008): `https://www.jstatsoft.org/article/view/v027i03`; M4 benchmarks: `https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Benchmarks%20and%20Evaluation.R`; fpp2 §11.1: `https://otexts.com/fpp2/complexseasonality.html`; MSTL paper: `https://arxiv.org/abs/2107.13462`.
- Rust crates are cited as `crate version path:line` inside the extracted `.crate` from `https://static.crates.io/crates/<name>/<name>-<version>.crate`.

## 0. Summary

- **What exists today.** backshift fits one seasonal period. The period must be supplied, and nothing in the crate detects it. Hints for `d` and `D` already exist. Neither initial values nor fixed parameters can be supplied, except through the warm start of a refit.
- **Hints.** No primary source shows that a supplied period, `d`/`D` or start values improve accuracy over the automatic search. The evidence is maintainer comments and this crate's own airline case (section 1).
- **Period detection** is a single-period heuristic everywhere:
  - R uses an AR spectrum peak plus an arbitrary threshold.
  - coreforecast uses the ACF of first differences, confirmed by STL strength above 0.64.
  - sktime, augurs and anofox use Welch periodogram thresholds.
  - Only the Welch and ACF-ranking variants return several candidates.
- **Multiple seasonality.** Every implementation read (R `stlf`/`stlm`, statsforecast, augurs, anofox, foresight) forecasts MSTL's seasonal components by repeating the last cycle (seasonal naive), and models only the adjusted series. Dynamic harmonic regression (Fourier terms plus ARIMA errors) is FPP3's recommended route for long periods.
- **In Rust, the field is crowded:**
  - foresight 0.7.3 has STL, MSTL checked against R `mstl()` to 1e-10, a decomposition forecaster generic over any model (its ARIMA included), TBATS and Fourier regressors.
  - anofox-forecast 0.21.0 has MSTL, TBATS, Fourier terms and two period detectors.
  - augurs has MSTL (through `stlrs`) and Welch period detection.
  - scirs2-series has all of these.
- **Where backshift can lead.** None of these crates builds `no_std` (ROADMAP §1). Fourier regressors feeding the existing heap-free `Filter::update_exog` give streaming dynamic harmonic regression on a device, which nobody has. One prerequisite: today the regressor count sets the filter's `R`, so the covariance grows with the square of the number of Fourier columns (section 5). MSTL, MSTL+ARIMA and period detection are catch-up features, not an edge.
- **Next step.** Ship a heap-free Fourier regressor generator (about 80 lines plus tests) that works with the existing `fit_exog` and `forecast_exog_into`. Then add AICc selection of K. For devices, first give `Filter` a separate regressor capacity.

## 1. Seeding and guided fitting

### What the crate already accepts

| Hint | Where | Status |
|---|---|---|
| Season length | `Arima::seasonal(p, d, q, period)` (src/fit.rs:136), `AutoArima::seasonal(period)` (src/fit/auto.rs:147) | one period only; required, never detected |
| Fixed `d` | `AutoArima::d` (src/fit/auto.rs:173) | exists; otherwise `ndiffs` by KPSS (src/diagnostics.rs:222) |
| Fixed `D` | `AutoArima::seasonal_d` (src/fit/auto.rs:179) | exists; otherwise `nsdiffs` by STL strength ≥ 0.64 (src/fit/auto.rs:674-685) |
| Mean or drift on/off | `Arima::mean` (src/fit.rs:154) | exists |
| Initial values | `Arima::start` (src/fit.rs:301) is private. Hannan–Rissanen gives the non-seasonal part; "the seasonal parts start at zero" (src/fit.rs:297-300) | no public API |
| Warm start | `ArimaSpec::refit` / `refit_exog` start "from the model's current coefficients" (src/fit/refit.rs:3-5, :120, :138); `Refitter` holds the buffers inline (src/fit/refit.rs:361) | exists, but only for an already-fitted spec with the same orders |
| Fixed parameters | none | missing |

### How the reference tools take hints

- **R `stats::arima`.**
  - `fixed` is a vector over all coefficients, with "NA if that coefficient should not be fixed, but estimated" (RS `man/arima.Rd` L66-L69). The code masks the fixed entries out with `mask <- is.na(fixed)` (RS `R/arima.R` L179).
  - Fixing any AR coefficient turns off the stationarity transform: `"some AR parameters were fixed: setting transform.pars = FALSE"` (RS `R/arima.R` L181-L187).
  - `init` gives "initial parameter values. Missing values will be filled in, by zeroes except for regression coefficients" (RS `man/arima.Rd` L78-L81).
  - `forecast::Arima` has no `fixed` or `init` arguments of its own. It passes them on through `...` (F `arima.R` L592), and re-uses a model with `fixed = model$coef` (F `arima.R` L809, L819).
- **statsmodels.**
  - `MLEModel.fit(start_params=..., includes_fixed=False)` (SM `tsa/statespace/mlemodel.py` L535-L545).
  - `fix_params({name: value})` is a context manager (L465-L523), and `fit_constrained` wraps it (L741-L767).
  - SARIMAX refuses partial AR or MA fixing while stationarity is enforced: `'Cannot fix individual %s parameters when %s. Must either fix all %s parameters or none.'` (SM `tsa/statespace/sarimax.py` L1491-L1512). This is the same problem R works around by dropping the transform.
- **pmdarima.** `auto_arima(start_params=None)` (PM `arima/auto.py` L355) passes the same `start_params` to every candidate (L428, L673). The docs describe it only as "Starting parameters for ``ARMA(p,q)``" (PM `arima/_doc.py` L142-L144).
- **R `auto.arima` hints** (F `newarima2.R` L87-L121):
  - Defaults: `d = NA, D = NA`, `start.p = 2, start.q = 2, start.P = 1, start.Q = 1`, `seasonal.test = c("seas","ocsb","hegy","ch")`.
  - The period is rounded: `m <- round(m)  # Avoid non-integer seasonal periods` (L187-L191).
- **foresight 0.7.3.**
  - Fixes `d` and `D` (`src/models/arima/auto.rs:40-43`).
  - Warm-starts each stepwise neighbour from the current best model (`auto.rs:193`, `src/models/arima/mod.rs:274-275`).
  - A cold start uses three points, "The likelihood of a mixed model can have more than one peak" (`mod.rs:102-103`). This is the author's assertion; no data is given.
  - No public fixed or initial coefficients were found.
- **anofox-forecast 0.21.0.**
  - `SARIMA::with_coefficients` skips optimisation entirely (`src/models/arima/model.rs:2160-2174`, :211).
  - Its AutoARIMA has no option to fix `d` or `D`; only the bounds can be set (`auto_arima.rs:106`).

### Does a hint improve results, or only save time?

No primary source measures this. What exists:

- Hyndman and Khandakar (2008, JSS 27(3), §3.2 p. 10) motivate the stepwise search by feasibility only: "it is often not feasible to simply fit every potential model". The paper gives no accuracy comparison of stepwise against exhaustive search.
- The same paper (§3.1 p. 10) says "over-differencing harms forecasts (Smith and Yadav 1994) and widens prediction intervals". So a correct `D` matters for accuracy, not only for time.
- The forecast docs recommend `stepwise = FALSE, approximation = FALSE` "if you are analysing just one time series" (F `newarima2.R` L7-L9). This implies the defaults trade accuracy for speed, but no numbers are given.
- The 0.64 seasonal-strength threshold is "based on minimizing MASE when forecasting using auto.arima on M3 and M4 data" (F `unitRoot.R` L173-L174). A second comment in the same file says "seasonal M3" (L279). These are maintainer comments, not a published experiment.
  - The switch from OCSB happened in forecast 8.3: "Changed method for choosing D in `auto.arima()` to a measure of seasonal strength" (forecast `NEWS.md` L228). The 8.2 default had been `seasonal.test=c("ocsb","ch")` (forecast 8.2 `R/newarima2.R:98`).
- This crate's own evidence (ROADMAP.md:28, :30):
  - pmdarima's OCSB missed the seasonal unit root of a simulated airline series.
  - foresight and anofox-forecast both picked (0,2,2)(1,0,1,12) on it, so they also got `D` wrong.
  - So the choice of `D`, whether from a test or a hint, changes the chosen model, not only the run time.
- Starting values: neither the R `arima` docs nor the statsmodels `fit` docs say anything about local optima or accuracy. There is no primary evidence that `init` or `start_params` change the optimum found. ROADMAP's tier-B check ("our llf ≥ statsmodels llf − 1e-6") is the in-crate guard.

**Conclusion.** Treat period and `d`/`D` hints as correctness inputs. A wrong `D` gives a different model, and there is evidence that over-differencing hurts. Treat start values as a speed and robustness feature.

Two gaps are worth closing:

1. **Public initial values** through `Arima::start_from(&ArimaSpec)`. This reuses the `refit` unconstrain path (src/fit/refit.rs). It is small, because the warm-start code already exists.
2. **Fixed parameters**, but only fixing whole parts, following statsmodels' rule, so that the PACF transform stays valid for the free parts. R's approach of fixing one AR coefficient and dropping the transform is not worth copying.

## 2. Automatic period detection

| Tool | Algorithm (source) | Output | Weaknesses (documented where cited; otherwise this document's reading of the code) |
|---|---|---|---|
| R `forecast::findfrequency` | Removes a linear trend with `tslm` (F `findfrequency.R` L30). Then `spec.ar` with `n.freq = 500` on the longest non-NA stretch (L32-L33). That is a Yule–Walker AR fit with order chosen by AIC, up to `floor(10 log10 n)` (RS `R/spectrum.R` L52-L61, RS `R/ar.R` L62-L63). The period counts only if `max(spec) > 10`, "Arbitrary threshold chosen by trial and error" (L34-L35). It is `floor(1/f_peak + 0.5)` (L36). A peak at frequency 0 moves on to the next local maximum (L37-L43). Every failure returns `1L` (L44-L52). | one integer | rounds non-integer periods; scale-dependent absolute threshold; a 500-point grid limits resolution for long periods; one period only; no failure cases documented |
| fable / fabletools | No data-driven detection. Periods come from the time index: `"day" = c("year" = 365.25, "week" = 7)` (fabletools `R/frequency.R` L46). feasts `feat_spectral` gives spectral entropy only (feasts `R/features.R` L511-L512). | calendar candidates | needs a tsibble index |
| statsmodels 0.14.0 | None. `freq_to_period` is a lookup from the pandas frequency (SM `tsa/tsatools.py` L784-L825). MSTL without periods uses it (SM `tsa/stl/mstl.py` L229-L256). | – | – |
| coreforecast (Nixtla) | `find_season_length(x, max_season_length)`: "The lag in [2, max_lag] with the largest autocovariance of the first differences" (CF `include/coreforecast/seasonal.h` L54-L76). It is kept only if STL seasonal strength > 0.64 (CF `include/coreforecast/diff.h` L103-L105). | one integer, or 0 | integer lags only; one period; the argmax of the ACF can pick a multiple of the true period |
| statsforecast | No detector in `python/statsforecast`. AutoARIMA takes `season_length: int = 1` (SF `models.py` L377). | – | – |
| sktime `SeasonalityACF` | Lags 2…`min(10 log10 n, n−1)`. Significant where the lower ACF confidence bound is > 0 at α = 0.05, ranked by that bound (SK `param_est/seasonality/_acf.py` L137-L175). "this should be applied to stationary series" (L21). | `sp_` plus `sp_significant_` | short default lag range (30 at n = 1000), so long periods are missed |
| sktime `SeasonalityACFqstat` | Ljung–Box p-values per lag, with Benjamini–Yekutieli correction (SK `_acf_qstat.py` L36, L150-L194) | ranked list | as above |
| sktime `SeasonalityPeriodogram` | Welch spectrum. `nperseg = min(2·max_period, n/2)`, `max_period = min(n/3, 512)`. Periods are rounded with `int(round(1/f))`. Bins above `0.1 × max power` are kept, "not by statistical testing" (SK `_periodogram.py` L22, L71-L76, L124-L132, L176-L192). | ranked list | 512 cap; rounding; no local-max test |
| augurs-seasons 0.10.3 | Welch through `welch-sde`, with the same defaults as sktime except `threshold: 0.9` (`src/periodogram.rs:7-25`, :74, :165). | `Vec<u32>` | Every bin ≥ threshold counts, with no local-maximum test, so adjacent bins of one peak all pass. Results are sorted by **ascending** power (`periodogram.rs:117`), so `periods[0]` is the weakest peak. This is a reading of the code, untested. |
| anofox-forecast 0.21.0 | Default: Welch "with spectral-leakage suppression". With the optional `seasonal-detection` feature: SAZED from `fdars-core` (`src/detection/mod.rs:10-13`). The feature is not default (`Cargo.toml:50`, :59), and `fdars-core` is pinned at 0.9 while 0.47.0 is the latest. Config: `max_periods: 5, min_power_ratio: 3.0, min_cycles: 2` (`src/detection/period.rs:65-72`). A separate Laplace-model detector scans `{7, 12, 24, 30, 52, 4}` with an ACF floor of 0.35 (`src/models/laplace/forecaster.rs:175-194`). | up to 5 periods | Welch internals unverified; no evidence found that detection feeds its AutoARIMA (`seasonal_period` is user-given, `auto_arima.rs:31`) |
| scirs2-series 0.6.5 | `detect_periods` with ACF, FFT or combined methods (`src/detection/mod.rs:40-45`, :113), plus Bartlett, Welch and multitaper periodograms (`src/features/frequency.rs`) | list | not read in depth |
| Nixtla tsfeatures, R tsfeatures | No estimator. `frequency` echoes the input (Nixtla `tsfeatures.py` L227-L243). R reads `frequency(x)` (R tsfeatures `R/thiyanga.R` L14). | – | – |

**What fails, across the board:**

1. **Rounding.** Every detector returns integers, so 365.25 or 52.18 cannot come back.
2. **Resolution.** Long periods are limited by spectral resolution: a grid of 500 points (R), a 512-period cap (sktime and augurs Welch defaults), or a default lag range near 10 log10 n (sktime ACF). With defaults, none of these would return 8766 on hourly data. coreforecast could, since `max_season_length` is caller-supplied, at the cost of an O(n·max) autocovariance scan. This is an inference from the cited code, not a test.
3. **Harmonics.** Harmonics and multiples can confuse argmax rules. With the ACF, lag 2s can score close to lag s. This is an inference, not documented by any of the tools.
4. **Thresholds.** They are ad hoc: R's "10", 0.1 or 0.9 of the maximum power, and the STL-strength 0.64.
5. **Multiple periods.** Only the ranked-list detectors (sktime, augurs, anofox, scirs2) try, and none confirms the candidates jointly. A sound way to confirm them would be to run MSTL on the candidates and keep those whose per-period strength exceeds 0.64. That combination is a proposal, not an existing method.

**Implication for backshift.** A detector is cheap, because `acorr` (src/lib.rs:197), `stl` and `seasonal_strength` (src/stl.rs:35) already exist. Port coreforecast's rule: the ACF argmax of first differences over 2..max, then confirm with STL strength. It is deterministic, has a reference implementation to test against, and yields a value that `AutoArima::seasonal` accepts directly. It is a convenience, not an edge. Return candidates rather than a verdict, because the literature gives no reason to trust a single answer.

## 3. Multiple seasonality

### MSTL

- **Paper.** Bandara, Hyndman and Bergmeir (2021), arXiv 2107.13462.
  - Periods are sorted ascending "to minimise such seasonal confounding" (p. 4).
  - STL is applied in turn to each period, inside an outer loop (Algorithm 1, p. 5).
  - Optional Box-Cox; the trend comes from the last fit.
  - Reported speed: 7 s, against TBATS 2521 s and Prophet 936 s on 100 bootstrapped electricity series (Table 5, p. 16; that row is labelled "STL" in the extracted text).
  - Reported accuracy: "significantly outperforms STR, TBATS, and PROPHET in estimating all components" (p. 15).
  - The text on p. 6 says that periods "smaller than half of the length" are ignored. This contradicts Algorithm 1 and the code, which keep the periods below n/2 and drop the rest.
- **R `forecast::mstl`.**
  - `iterate = 2, s.window = 7 + 4 * seq_len(6)`, so windows 11, 15, …, 31 (F `mstl.R` L24-L29).
  - Drops periods ≥ n/2 with a warning (L37-L39) and sorts the rest ascending (L43).
  - A plain `ts` gets one iteration (L44-L46).
  - Non-integer periods reach `stats::stl`, which truncates them with `as.integer(period)` (RS `R/stl.R` L68-L69). So 365.25 becomes 365.
  - `stats::stl` needs `n > 2 * period` (RS `R/stl.R` L46-L48).
- **statsmodels `MSTL`.**
  - Same defaults: windows `7 + 4 * i` (SM `tsa/stl/mstl.py` L278-L279), `iterate=2`, and 1 for a single period (L136).
  - Drops periods ≥ nobs/2 (L216-L224).
  - `lmbda="auto"` uses the scipy Box-Cox MLE (L139-L143).
  - Each pass calls `STL(..., seasonal=self.windows[i])` (L159-L170).
- **statsforecast `mstl()`.**
  - Same windows and iterations (SF `mstl.py` L15, L32-L34), but with `seasonal_deg: 0` as the default (L47).
  - Box-Cox is "not implemented yet" (L42-L46).
  - It does **not** drop long periods.
- **feasts `STL`.** Takes several `season(period=)` terms with `iterations = 2` and windows `7 + 4*order(period)` (feasts `R/stl.R` L28-L72).

**Consequence for backshift: MSTL is not just "a loop over `stl`".** Three changes are needed:

1. `stl` hard-codes the statsmodels default seasonal window of 7 (src/stl.rs:52, :104-105). MSTL needs window 11 for the first period, 15 for the second, and so on. The trend window is derived from the seasonal window (src/stl.rs:106), so it changes too. `stl` therefore needs a variant that takes the seasonal window.
2. statsforecast's default `seasonal_deg = 0` differs from statsmodels' default. Parity has to name one reference; statsmodels is the natural one, since the STL port already matches it to 1e-9 (ROADMAP.md:28).
3. `stl` is behind `alloc` and returns `Vec`s (src/stl.rs:17-18, :67). An MSTL built on it is a batch, alloc-only tool, not a fixed-memory one.

The boundaries differ by one. `stl` accepts `n = 2*period` (`n < 2 * period` returns `None`, src/stl.rs:69), as statsmodels' `STL` does: a local statsmodels 0.14.0 run fits a 24-value series with `period=12`. R's `stl` needs `n > 2 * period` (RS `R/stl.R` L46-L48). Both MSTLs drop periods with `period >= n/2`, so an MSTL that matches statsmodels must apply that strict rule itself, before calling `stl`.

### STL plus ARIMA forecasting (`stlf`, `stlm`, statsforecast `MSTL`)

All the implementations read (R, statsforecast, augurs, anofox, foresight) work the same way:

1. Fit a non-seasonal model to the seasonally adjusted series (trend plus remainder).
2. Forecast each seasonal component by repeating its last cycle.
3. Add the forecasts together.

The evidence:

- R comment: "# Forecast seasonal series with seasonal naive", then `rep(object$stl[n - rev(seq_len(mp)) + 1, colname], ...)` (F `mstl.R` L619-L629). The components are summed into the mean and the bounds (L631-L638).
- `stlm` fits `method = c("ets","arima")` on `seasadj(stld)`. ETS must be non-seasonal (L527-L531). The ARIMA branch is `auto.arima(x, xreg = xreg, seasonal = FALSE, ...)` (L541-L543).
- statsforecast `MSTL` uses `trend_forecaster=AutoETS(model="ZZN")` (SF `models.py` L6071), the same `np.tile(...[-mp:])` seasonal forecast (L6037-L6039), and adds it to the trend model's output (L6148-L6150).
- augurs-mstl repeats the last cycle with `.skip(component.len() - period).cycle()` (augurs-mstl 0.10.3 `src/lib.rs:202-212`). Its trend models plug in through `TrendModel` (`src/trend.rs:16-20`), and the only ready-made one is AutoETS (augurs-ets 0.10.3 `src/trend.rs:2`). No ARIMA plug-in exists (`augurs-arima` returns 404 on crates.io).
- anofox-forecast's MSTL forecaster offers AutoETS, SES, Linear or Naive trends, with no ARIMA (`src/models/mstl_forecaster.rs:23-41`).
- foresight's `Decomposed<M: Model>` "Runs a model on the seasonally adjusted series and adds the seasonal patterns of the last cycle to its forecasts" (foresight 0.7.3 `src/models/decomposed.rs:8-12`, :30). Any foresight model fits inside, its ARIMA included.

Two consequences follow:

- **The intervals ignore seasonal uncertainty.** They come from the trend model alone (statsforecast L6148-L6150; R adds the same seasonal offset to both bounds).
- **The seasonal component is frozen.** Streaming it costs a ring buffer of one cycle per period. For an hourly yearly period that is 8766 values, about 70 KB, against 2K coefficients for a Fourier fit (but see the filter-size caveat in section 5).

### Dynamic harmonic regression (Fourier terms plus ARIMA errors)

**R `forecast::fourier(x, K, h)`:**

- The terms are `sinpi(2 * p * t)` and `cospi(2 * p * t)` with `p = k/period`, for `t = 1..n`, and `n + 1..n + h` for the future (F `season.R` L242-L298).
- The period is used unrounded (L259), so 365.25 works.
- Each K is checked with `2 * K > period`, which raises "`K` must not be greater than half the seasonal period" (L261-L265).
- Frequencies duplicated across periods are dropped (L284-L286).
- The identically-zero sine at K = period/2 is dropped (L289-L290).
- The docs say: "In practice, K should be chosen to minimize the AICc" (L222).
- fable implements the same algorithm (fable `R/00_specials.R` L96-L110).

**FPP3, Hyndman and Athanasopoulos:**

- §10.5 "Dynamic harmonic regression" (https://otexts.com/fpp3/dhr.html):
  - "When there are long seasonal periods, a dynamic regression with Fourier terms is often better than other models".
  - "The ARIMA() function will allow a seasonal period up to m = 350, but in practice will usually run out of memory whenever the seasonal period is more than about 200".
  - The one drawback it names is that "the seasonality is assumed to be fixed".
- §12.1 "Complex seasonality" (https://otexts.com/fpp3/complexseasonality.html):
  - It fits `ARIMA(sqrt(Calls) ~ PDQ(0, 0, 0) + pdq(d = 0) + fourier(period = 169, K = 10) + fourier(period = 5 * 169, K = 5))`.
  - It warns that AICc "for high seasonal periods ... tends to over-estimate the number of terms required".
- §13.1 (https://otexts.com/fpp3/weekly.html) uses Fourier terms for the 52.18-week year, because "most of the methods will not handle such a large seasonal period efficiently".

**Python:**

- statsmodels `Fourier(period: float, order)` checks `2 * order <= period` and starts in-sample time at t = 0 (SM `tsa/deterministic.py` L507-L547). Its docstring says `int`, but the code takes a float.
- pmdarima `FourierFeaturizer(m, k=None)`:
  - `k` defaults to `m // 2` (PM `preprocessing/exog/fourier.py` L138-L142).
  - It handles a single season only (L146).
  - It keeps the all-zero Nyquist sine (L148).
  - It produces future terms by `n_periods` (L199-L205).
- sktime `FourierFeatures` takes float periods but only checks K ≤ sp, not 2K ≤ sp (SK `transformations/fourier.py` L163-L171).

**Why this suits backshift:**

- The state count is `max(p + P s, q + Q s + 1) + d + D s` (src/filter.rs:120). That is why the wasm binding caps `MAX_PERIOD = 366` and `MAX_STATES = 128` (bindings/wasm/src/lib.rs:37, :42). With Fourier terms the seasonality moves into regressors, so the states stay at `max(p, q + 1) + d`.
- `Filter` needs capacity `max(states(), exog.len())` (src/filter.rs:128-132, :310-315).
- A daily-plus-weekly-plus-yearly hourly model with K = (10, 5, 10) has 50 regressors and an ARMA(2,2) error. It fits in `Filter<3, 50>`, where a seasonal ARIMA at s = 8766 cannot run at all.
- One `R` used to size everything, including the covariance `p: [[f64; R]; R]` and `beta: [f64; R]`, so `Filter<50>` carried a 50×50 covariance (about 22 KB) where the ARMA(2,2) error needs a 3×3 block. Since #32 the regressors have their own capacity `M`: `Filter<3, 50>` takes 640 bytes.

### TBATS

- **Paper.** De Livera, Hyndman and Snyder (2011), "Forecasting Time Series With Complex Seasonal Patterns Using Exponential Smoothing", JASA 106(496):1513-1527, doi:10.1198/jasa.2011.tm09771 (checked via Crossref).
- **R `tbats`.** Box-Cox, damped trend, ARMA errors chosen by AIC, and trigonometric seasonality `<m_i, k_i>` (F `tbats.R` L4-L24). Its K search uses `floor((m - 1)/2)` (L221), so non-integer periods work. fpp2 §11.1 says it is "flexible enough to handle non-integer frequencies".
- **Drawbacks**, also from fpp2 §11.1: "slow to estimate", prediction intervals "much too wide ... quite often", and "TBATS models do not allow for covariates".
- **fable** has no TBATS (fable `R/` listing).
- **statsforecast `AutoTBATS`** takes `int` periods only (SF `models.py` L1827-L1850; `tbats.py` L19-L32 uses `rolling(window=2*m)`).
- **Rust.** foresight (`src/models/tbats.rs`, "several periods, not necessarily whole", `models/mod.rs:15`), anofox-forecast (`src/models/tbats/mod.rs`) and scirs2-series (`src/tbats.rs`) all ship TBATS.
- **Assessment.** It is a different model family, slow, and already covered by three Rust crates. Skip it.

### Prophet-style calendar terms

- Prophet's seasonality is a Fourier series on time in days since 1970 (PR L543-L551).
- The defaults (PR L1202-L1244):

  | Seasonality | Period | Order | Switched on automatically when |
  |---|---|---|---|
  | yearly | 365.25 | 10 | at least 730 days of history |
  | weekly | 7 | 3 | at least 2 weeks of history and spacing under 7 days |
  | daily | 1 | 4 | – |

- Holidays are 0/1 indicator columns, one per day offset in the window `[lower_window, upper_window]` (PR L676-L688).
- Reference: Taylor and Letham (2018), *The American Statistician* 72(1):37-45, doi:10.1080/00031305.2017.1380080 (checked via Crossref).
- For backshift, holiday dummies are just more `fit_exog` columns. A generator needs a calendar, and that does not belong in a `no_std` core.

### Data-length requirements

| Method | Minimum data | Long or non-integer periods |
|---|---|---|
| Seasonal ARIMA (backshift) | The differenced series needs `p + q + P + Q + columns + 2` non-missing values (src/fit.rs:186-187); `nsdiffs` needs `n ≥ 2s` for STL | integer `s` only. States grow with s (src/filter.rs:120), and the wasm binding refuses s > 366. R rounds the period (F `newarima2.R` L187-L191) and caps `max.P` at `floor(n/3/m)` (L193-L196) |
| STL / MSTL | `n > 2·period` for each period kept (backshift `stl` and statsmodels `STL` accept `n = 2·period`, src/stl.rs:69; RS `R/stl.R` L46-L48; SM `mstl.py` L216-L224). R drops periods ≥ n/2 (F `mstl.R` L37-L39) | integer only (R truncates 365.25 to 365); for 8766-hourly data that means more than 2 years of history |
| STL + ARIMA (`stlm`) | as STL, plus the ARIMA minimum on the adjusted series | as STL |
| Fourier + ARIMA errors | `2K ≤ period` for each period (F `season.R` L261-L265); the regression needs n well above the number of columns | real-valued period; 365.25 and 8766 cost 2K columns, not s states |
| TBATS | not documented; the K search needs `floor((m-1)/2)` | non-integer in R; integer in statsforecast |
| Prophet | yearly term auto-on at ≥ 730 days, weekly at ≥ 2 weeks | real-valued |

## 4. Feature table of other libraries (extends ROADMAP §1)

Versions are as in ROADMAP §1 unless they are new here.

| Tool | Period detection | MSTL | STL + model forecast | Fourier / DHR | TBATS | Fixed / initial params | `no_std` |
|---|---|---|---|---|---|---|---|
| backshift 0.2.1 | no | no (single `stl`, window fixed at 7) | no | regressors yes (`fit_exog`); no generator | no | warm start via `refit` only | yes (filter, refit) |
| foresight 0.7.3 | no (period from `Series::new`, `src/series.rs:27`) | yes, robust option (`src/decompose.rs:1-3`, :148); checked against R to 1e-10 (`tests/against_r.rs:744`) | yes, any model incl. ARIMA (`src/models/decomposed.rs:8-30`) | `regressors::fourier(period: f64, order, rows)` (`src/regressors.rs:51`) | yes | fixes `d`/`D`; internal warm start | no |
| anofox-forecast 0.21.0 | yes: Welch, SAZED behind a feature | yes (`src/seasonality/mstl.rs`) | yes, ETS/SES/Linear/Naive trend, no ARIMA | Prophet-style (`src/seasonality/fourier.rs:1`) | yes | `with_coefficients` (no re-fit) | no |
| augurs 0.10.3 | `augurs-seasons` (Welch) | `augurs-mstl` through `stlrs` 0.4.0, f32 (`src/lib.rs:103`) | yes, ETS trend; no ARIMA | inside `augurs-prophet` only | no | – | no |
| stlrs 0.4.0 | no | yes, robust, zero deps | no | no | no | – | not tested |
| scirs2-series 0.6.5 | yes (ACF/FFT/combined) | yes | not read | not read | yes | not read | no |
| statsmodels 0.14.0 | no (`freq_to_period` lookup) | yes | no built-in forecaster | `Fourier`, `CalendarFourier`, `DeterministicProcess` | no | `start_params`, `fix_params` | – |
| statsforecast | no; coreforecast `find_season_length` | yes, no Box-Cox | yes (`MSTL(trend_forecaster=...)`) | no generator found | `AutoTBATS` (int periods) | not checked | – |
| pmdarima | no | no | no | `FourierFeaturizer` (one season) | no | `start_params` | – |
| sktime | ACF, ACF-qstat, Welch | wraps statsmodels | via pipelines (not checked) | `FourierFeatures` | wraps others (not checked) | – | – |
| R forecast | `findfrequency` | `mstl` | `stlf`, `stlm` (ETS/ARIMA) | `fourier()` | `tbats` | `fixed`, `init` via `stats::arima` | – |
| R fable / feasts | calendar periods only | `feasts::STL` multi-season | via decomposition models (not checked) | `fourier()` special | no | named `fixed` (fable `R/arima.R` L487-L488) | – |

## 5. Where backshift could be first or best

- **MSTL plus streaming ARIMA in fixed memory: weak.**
  - MSTL itself is everywhere: foresight, anofox, augurs/stlrs and scirs2 in Rust alone.
  - MSTL plus ARIMA is not new in Rust either: foresight's `Decomposed<M>` takes any of its models (`src/models/decomposed.rs:30`).
  - The seasonal forecast is a frozen last cycle in every implementation. Streaming it is a ring buffer of `Σ period_i` values, plus a `Filter` on the adjusted series.
  - That is genuinely fixed-memory and `no_std`-compatible at run time, and nobody offers it. It is also easy to copy, and costly on a device for long periods (70 KB at 8766).
  - The decomposition remains a batch, alloc-only step (src/stl.rs:67).
  - R `stlm(method="arima")` and statsforecast `MSTL(trend_forecaster=...)` already do this outside Rust.
- **Fourier-regressor streaming forecasts on devices: strongest.**
  - The pieces exist: `fit_exog` (src/fit.rs:189), `Filter::update_exog` (src/filter.rs:376) and `forecast_exog_into` (src/filter.rs:437).
  - The only missing piece is a heap-free generator of the sin/cos row for time t and the next h rows.
  - It removes the `MAX_PERIOD` and `MAX_STATES` limits for long and non-integer periods, and it handles several periods at once.
  - Nobody else is `no_std` (ROADMAP §1), and no other ARIMA crate streams exactly (ROADMAP §1 notes).
  - Caveats:
    - foresight already has the batch half (`regressors::fourier`).
    - As built today, `Filter<R>` sizes its covariance by the regressor count (section 3, DHR). The device pitch needs a separate regressor capacity first; otherwise 50 Fourier columns cost about 22 KB per filter.
    - The device must compute `sin` and `cos` through `libm`. Compute each term from `t mod period` rather than a rotation recurrence, so it does not drift over millions of samples.
- **MSTL validated against statsmodels to 1e-9: owned already, for R.**
  - foresight 0.7.3 checks `Mstl` against R `forecast::mstl()` with one period (12) and two (7, 30), using `exactly`, which asserts a relative error below 1e-10 (foresight 0.7.3 `tests/against_r.rs:624-630`, :744-:792).
  - statsmodels parity would be credible, because the STL port already matches (ROADMAP.md:28). It would differ from foresight's check only in the reference library. That is a selling point for Python porters at most, the same verdict ROADMAP §2 gives reference validation.
- **Period detection feeding auto-ARIMA: catch-up.**
  - anofox and scirs2 already detect periods.
  - coreforecast uses the exact "detect, then confirm with STL strength" design.
  - Nothing found wires detection into an auto-ARIMA by default. That is a small convenience, not an edge, and section 2 shows that the detectors are heuristic.

## 6. Recommended order

| # | Piece | Reuses | Rough size |
|---|---|---|---|
| 0 | Separate regressor capacity from state capacity in `Filter` (for example `Filter<R, M>`), so 2K Fourier columns do not inflate the R×R covariance. This is a breaking API change, so do it before 1.0. | `Filter` (src/filter.rs:260-287), `ArimaSpec::capacity` (src/filter.rs:131) | ~100 lines, touches the bindings |
| 1 | `fourier(period: f64, k, t0, rows, out: &mut [f64])`: heap-free, writes rows into a caller buffer, follows R's column order and Nyquist drop. Add an alloc convenience that returns a `Vec`. Check against R `fourier()` and statsmodels `Fourier` values (mind the t = 1 vs t = 0 origin). | `fit_exog`, `update_exog`, `forecast_exog_into`; `libm` already in the `no_std` build | ~80 lines + tests |
| 2 | DHR selection: loop K = 1..Kmax per period, fit `AutoArima` with `seasonal_d(0)`, no seasonal part, and the Fourier regressors, and keep the lowest AICc (FPP3 §10.5, §12.1). Document the over-estimation warning for long periods. | `AutoArima::fit_exog` (src/fit/auto.rs:221), `Fitted::aicc` (src/fit.rs:407) | ~100 lines |
| 3 | Public initial values: `Arima::start_from(&ArimaSpec)`, optionally with fixing of whole parts | the unconstrain path in `refit` | ~80 lines |
| 4 | `stl_with(values, period, seasonal_window)` plus `mstl(values, periods, iterate)`, with statsmodels defaults (windows `7 + 4i`, iterate 2, drop periods ≥ n/2, ascending order). Box-Cox later. Tier-A against statsmodels `MSTL`. | `stl` internals (src/stl.rs:67, `Windows::defaults` :104) | ~120 lines + references |
| 5 | MSTL forecast (parity with foresight and R `stlm`, not an edge): ARIMA on the adjusted series plus last-cycle seasonals, documented as "intervals ignore seasonal uncertainty". Optional fixed-memory streaming wrapper (ring buffers plus `Filter`). | items 4, `AutoArima`, `Filter` | ~150 lines |
| 6 | `find_period(x, max)`: the coreforecast rule (ACF argmax of first differences, then STL strength > 0.64), returning ranked candidates. Test against coreforecast outputs. | `acorr` (src/lib.rs:197), `stl`, `seasonal_strength` | ~60 lines |
| – | Skip: TBATS (slow, covered by three Rust crates); holiday calendars (no calendar in a `no_std` core); detecting multiple periods jointly (no primary method found) | – | – |

Items 0–2 are the edge. Items 3 and 6 are small conveniences. Items 4–5 are parity features, worth doing only if users ask for decomposition.

## 7. Could not verify

- **Evidence on hints.** No published experiment compares supplied and searched period, `d`/`D` or start values for forecast accuracy, nor stepwise against exhaustive search. The 0.64 threshold rests on maintainer comments (F `unitRoot.R` L173-L174, L279), which disagree on M3 versus M3 plus M4.
- **The M4 paper.** Makridakis et al. 2020 (IJF) was not read (HTTP 403). The M4 benchmark code only shows that the frequency is taken as given (`Benchmarks and Evaluation.R` L106).
- **R behaviour with non-integer periods.** How `seq_len(365.25)` behaves inside `forecast.stlm` and `forecast.mstl` is unverified, because R was not run.
- **TBATS limits.** That R `tbats` formally accepts non-integer periods is implied by the code and by fpp2 only. That statsforecast `AutoTBATS` fails on float periods was inferred from the code, not run.
- **The augurs-seasons ordering.** The ascending-power ordering is a reading of the code; its tests cover single periods only.
- **anofox internals.** Its Welch leakage suppression, its SAZED scoring, and whether any detector feeds its AutoARIMA were not read.
- **Absence claims.** That utilsforecast has no period detection rests on its module tree only. Some cells for scirs2-series, sktime and fable pipelines are marked "not read" or "not checked".
- **Paper text.** The Prophet defaults are cited from code, not from the paper text. The MSTL paper's own default windows and iterations are not stated in its text; the defaults are cited from code.
- **Untested builds.** None of the other libraries was built or run. `stlrs` was not tested for `no_std`.

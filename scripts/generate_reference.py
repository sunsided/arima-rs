"""Generate statsmodels reference vectors for the integration tests.

Writes `tests/it/reference_data.rs` and `tests/it/arma_data.rs`. Run from the repository root:

    python3 -I scripts/generate_reference.py

Requires numpy and statsmodels (tested with statsmodels 0.14).
"""

import pathlib
import warnings

import numpy as np
import statsmodels
from statsmodels.regression.linear_model import burg, yule_walker
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import acf, acovf, kpss, pacf

TESTS = pathlib.Path(__file__).resolve().parent.parent / "tests" / "it"
OUTPUT = TESTS / "reference_data.rs"
ARMA_OUTPUT = TESTS / "arma_data.rs"
STL_OUTPUT = TESTS / "stl_data.rs"
MAX_LAGS = 10
AR_ORDER = 3


def returns_series():
    """Ten small daily returns, the series used in the unit tests."""
    return np.array([0.05, 0.01, -0.02, 0.03, -0.04, 0.06, 0.02, -0.01, -0.03, 0.04])


def ar2_series():
    """200 samples of an AR(2) process with a fixed seed."""
    rng = np.random.default_rng(20261122)
    noise = rng.standard_normal(260)
    y = np.zeros_like(noise)
    for t in range(2, len(y)):
        y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + noise[t]
    return y[60:]


def sine_series():
    """1000 samples of two overlaid sines, the series used in the unit tests."""
    n = 1000
    t = 2.0 * np.arange(n) / (n - 1)
    return 4.0 * np.sin(2 * np.pi * 2.5 * t) + 0.5 * np.sin(2 * np.pi * 16 * t)


def trend_series():
    """A noisy linear trend, which is far from stationary."""
    rng = np.random.default_rng(7)
    return 0.5 * np.arange(50) + rng.standard_normal(50)


SERIES = [
    ("RETURNS", "Ten small daily returns.", returns_series()),
    ("AR2", "200 samples of `y_t = 0.6 y_(t-1) - 0.3 y_(t-2) + e_t`, seed 20261122.", ar2_series()),
    ("SINE", "1000 samples of two overlaid sines (2.5 and 16 cycles).", sine_series()),
    ("TREND", "50 samples of a noisy linear trend with slope 0.5, seed 7.", trend_series()),
]


def rust_float(value):
    """Format a float as a Rust literal that round-trips exactly."""
    value = float(value)
    return "f64::NAN" if np.isnan(value) else repr(value)


def rust_array(values):
    """Format floats as a Rust array literal that round-trips exactly."""
    if len(values) == 0:
        return "&[]"
    body = ",\n".join(f"        {rust_float(v)}" for v in values)
    return f"&[\n{body},\n    ]"


def case(name, doc, y):
    """Render one `Case` constant."""
    # statsmodels only computes PACF for lags below half the sample size.
    lags = min(MAX_LAGS, len(y) // 2 - 1)
    fields = {
        "values": y,
        "acf": acf(y, nlags=lags, fft=False),
        "pacf": pacf(y, nlags=lags, method="ldb"),
        "acovf_biased": acovf(y, adjusted=False, fft=False)[: lags + 1],
        "acovf_unbiased": acovf(y, adjusted=True, fft=False)[: lags + 1],
    }
    order = min(AR_ORDER, lags)
    yw_mle, sigma_mle = yule_walker(y, order, method="mle")
    yw_adjusted, sigma_adjusted = yule_walker(y, order, method="adjusted")
    burg_phi, burg_sigma2 = burg(y, order)
    fields |= {
        "yw_mle_phi": yw_mle,
        "yw_adjusted_phi": yw_adjusted,
        "burg_phi": burg_phi,
    }
    ljung = acorr_ljungbox(y, lags=[lags], model_df=0)
    jb_stat, jb_p, _, _ = jarque_bera(y)
    kpss_lags = min(int(np.ceil(12.0 * (len(y) / 100.0) ** 0.25)), len(y) - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the table's p-value clipping warns
        level_stat, level_p, _, _ = kpss(y, regression="c", nlags=kpss_lags)
        trend_stat, trend_p, _, _ = kpss(y, regression="ct", nlags=kpss_lags)
    scalars = {
        "ljung_box_statistic": rust_float(ljung["lb_stat"].iloc[0]),
        "ljung_box_p_value": rust_float(ljung["lb_pvalue"].iloc[0]),
        "jarque_bera_statistic": rust_float(jb_stat),
        "jarque_bera_p_value": rust_float(jb_p),
        "kpss_level_statistic": rust_float(level_stat),
        "kpss_level_p_value": rust_float(level_p),
        "kpss_trend_statistic": rust_float(trend_stat),
        "kpss_trend_p_value": rust_float(trend_p),
        "ar_order": str(order),
        # yule_walker returns sigma, not sigma2, and NaN when sigma2 <= 0.
        "yw_mle_sigma2": rust_float(sigma_mle**2),
        "yw_adjusted_sigma2": rust_float(sigma_adjusted**2),
        "burg_sigma2": rust_float(burg_sigma2),
    }
    lines = [f"/// {doc}", f"pub const {name}: Case = Case {{", f"    max_lags: {lags},"]
    for field, values in fields.items():
        lines.append(f"    {field}: {rust_array(values)},")
    lines += [
        f"    var_biased: {float(np.var(y, ddof=0))!r},",
        f"    var_unbiased: {float(np.var(y, ddof=1))!r},",
    ]
    lines += [f"    {field}: {value}," for field, value in scalars.items()]
    lines += ["};"]
    return "\n".join(lines)


ARMA_LENGTH = 60
ARMA_SPLIT = 40
ARMA_HORIZON = 5

# (name, doc, ar, d, ma, mean, sigma2, missing index). Seeds follow the list order.
ARMA_CASES = [
    ("AR1", "AR(1).", [0.6], 0, [], 0.0, 1.0, None),
    ("MA1", "MA(1).", [], 0, [0.5], 0.0, 1.3, None),
    ("ARMA21", "ARMA(2, 1).", [0.5, -0.2], 0, [0.4], 0.0, 1.3, None),
    ("ARMA31", "ARMA(3, 1): more AR than MA lags, so the selection vector is padded.", [0.3, 0.2, -0.25], 0, [0.5], 0.0, 0.8, None),
    ("ARMA12", "ARMA(1, 2): more MA than AR lags, so the transition column is padded.", [0.7], 0, [0.4, -0.3], 0.0, 2.0, None),
    ("WHITE", "ARMA(0, 0): white noise with one state.", [], 0, [], 0.0, 1.5, None),
    ("MISSING", "ARMA(1, 1) with the value at index 10 missing.", [0.5], 0, [0.3], 0.0, 1.0, 10),
    ("MEAN", "ARMA(1, 1) with mean 3: statsmodels ARIMA `trend=\"c\"`.", [0.5], 0, [0.3], 3.0, 1.0, None),
    ("I1", "ARIMA(1, 1, 1).", [0.5], 1, [0.3], 0.0, 1.0, None),
    ("DRIFT", "ARIMA(1, 1, 1) with drift 0.2: statsmodels ARIMA `trend=\"t\"`.", [0.5], 1, [0.3], 0.2, 1.0, None),
    ("I2", "ARIMA(2, 2, 0).", [0.4, -0.3], 2, [], 0.0, 0.7, None),
    ("RANDOM_WALK", "ARIMA(0, 1, 0): a random walk.", [], 1, [], 0.0, 1.0, None),
]

# Seasonal cases: the same fields plus (seasonal ar, D, seasonal ma, period).
SEASONAL_CASES = [
    ("SAR4", "ARIMA(1, 0, 0)(1, 0, 0, 4) with mean 2.", [0.4], 0, [], 2.0, 1.0, None, ([0.5], 0, [], 4)),
    ("SMA4", "ARIMA(0, 0, 1)(0, 0, 1, 4).", [], 0, [0.3], 0.0, 0.8, None, ([], 0, [0.6], 4)),
    ("AIRLINE4", "The airline model ARIMA(0, 1, 1)(0, 1, 1, 4).", [], 1, [-0.4], 0.0, 1.0, None, ([], 1, [-0.6], 4)),
    ("SARIMA4", "ARIMA(1, 1, 0)(1, 1, 0, 4).", [0.3], 1, [], 0.0, 1.0, None, ([-0.4], 1, [], 4)),
    ("SAR12", "ARIMA(1, 0, 0)(1, 0, 0, 12) with mean -1.", [0.5], 0, [], -1.0, 1.0, None, ([0.6], 0, [], 12)),
]

NO_SEASON = ([], 0, [], 0)

# Regression with ARIMA errors: the same fields plus the seasonal part and beta.
EXOG_CASES = [
    ("EXOG_MEAN", "ARIMA(1, 0, 0) with mean 3 and two regressors (noise and a trend).", [0.5], 0, [], 3.0, 1.0, None, NO_SEASON, [1.5, -2.0]),
    ("EXOG_I1", "ARIMA(1, 1, 0) with one regressor.", [0.4], 1, [], 0.0, 1.0, None, NO_SEASON, [1.5]),
    ("EXOG_AIRLINE4", "ARIMA(0, 1, 1)(0, 1, 1, 4) with one regressor.", [], 1, [-0.4], 0.0, 1.0, None, ([], 1, [-0.6], 4), [0.8]),
]


def regressors(beta, seed, rows, offset):
    """`rows` regressor rows for `beta`: standard normal columns, the second one a trend.

    The trend runs from 0 to 1 over the history and the forecast horizon together;
    `offset` is the index of the first row on it.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, len(beta)))
    if len(beta) >= 2:
        x[:, 1] = np.linspace(0.0, 1.0, ARMA_LENGTH + ARMA_HORIZON)[offset : offset + rows]
    return x


def seasonal_series(ar, d, ma, mean, sigma2, seed, seasonal):
    """Simulate ARMA_LENGTH raw values of a seasonal ARIMA model.

    The ARMA part uses the expanded polynomials; the raw series then integrates the
    differencing polynomial from k starting values of 10.
    """
    sar, seasonal_d, sma, s = seasonal
    ar_poly = np.convolve(np.r_[1.0, -np.array(ar, float)], lag_polynomial(sar, s, -1.0))
    ma_poly = np.convolve(np.r_[1.0, np.array(ma, float)], lag_polynomial(sma, s, 1.0))
    delta = differencing_polynomial(d, seasonal_d, s)
    k = len(delta) - 1
    w = arma_series(list(-ar_poly[1:]), list(ma_poly[1:]), sigma2, seed, ARMA_LENGTH - k) + mean
    y = np.full(ARMA_LENGTH, 10.0)
    for t in range(k, ARMA_LENGTH):
        y[t] = w[t - k] - np.dot(delta[1:], y[t - 1 :: -1][:k]) if k else w[t]
    return y


def lag_polynomial(coefficients, period, sign):
    """`1 + sign * sum c_j B^(j period)` as a coefficient array."""
    poly = np.zeros(len(coefficients) * period + 1)
    poly[0] = 1.0
    for j, c in enumerate(coefficients):
        poly[(j + 1) * period] = sign * c
    return poly


def differencing_polynomial(d, seasonal_d, period):
    """The coefficients of `(1 - B)^d (1 - B^s)^D`."""
    delta = np.array([1.0])
    for _ in range(d):
        delta = np.convolve(delta, [1.0, -1.0])
    for _ in range(seasonal_d):
        delta = np.convolve(delta, lag_polynomial([1.0], period, -1.0))
    return delta


def simple_difference(y, d, seasonal_d, period):
    """`(1 - B)^d (1 - B^s)^D y` as statsmodels' simple differencing computes it."""
    w = np.diff(y, n=d)
    for _ in range(seasonal_d):
        w = w[period:] - w[:-period]
    return w


def arima_series(ar, d, ma, mean, sigma2, seed):
    """Simulate ARMA_LENGTH raw values whose `d`-th difference is the ARMA series plus `mean`."""
    w = arma_series(ar, ma, sigma2, seed, ARMA_LENGTH - d) + mean
    for _ in range(d):
        w = np.concatenate([[10.0], 10.0 + np.cumsum(w)])
    return w


def arma_series(ar, ma, sigma2, seed, length=ARMA_LENGTH):
    """Simulate `length` ARMA values after a burn-in, so the start is near stationarity."""
    rng = np.random.default_rng(seed)
    burn = 200
    e = rng.standard_normal(length + burn) * np.sqrt(sigma2)
    y = np.zeros_like(e)
    for t in range(len(y)):
        y[t] = e[t]
        y[t] += sum(phi * y[t - j - 1] for j, phi in enumerate(ar) if t - j - 1 >= 0)
        y[t] += sum(theta * e[t - j - 1] for j, theta in enumerate(ma) if t - j - 1 >= 0)
    return y[burn:]


def exact_filter(model, params):
    """Filter without statsmodels' steady-state shortcut.

    By default statsmodels freezes the state covariance once it changes by less
    than 1e-19 and reuses it, which moves later innovations by up to about 1e-9.
    The crate always runs the full recursion, so the references must too.
    """
    model.ssm.tolerance = 0
    return model.filter(params)


def exact_extend(head, tail, order, params, seasonal_order=(0, 0, 0, 0)):
    """`head.extend(tail)` without the steady-state shortcut.

    `extend` clones the model, which resets the tolerance, so this repeats what it
    does: a new model on `tail`, known-initialised from `head`'s last predicted state.
    """
    model = SARIMAX(tail, order=order, seasonal_order=seasonal_order, trend="n")
    model.ssm.initialize_known(head.predicted_state[:, -1], head.predicted_state_cov[:, :, -1])
    return exact_filter(model, params)


def forecast_model(y, ar, d, ma, mean, seasonal, x=None):
    """The statsmodels model whose forecasts are on the original scale, with its trend parameters."""
    sar, seasonal_d, sma, s = seasonal
    order = (len(ar), d, len(ma))
    seasonal_order = (len(sar), seasonal_d, len(sma), s)
    if mean == 0.0:
        return SARIMAX(y, exog=x, order=order, seasonal_order=seasonal_order, trend="n"), []
    trend = {0: "c", 1: "t"}[d]
    return ARIMA(y, exog=x, order=order, seasonal_order=seasonal_order, trend=trend), [mean]


def arma_case(index, name, doc, ar, d, ma, mean, sigma2, missing, seasonal=NO_SEASON, beta=()):
    """Render one `ArmaCase` constant from statsmodels filters at fixed parameters.

    Innovations and the likelihood come from SARIMAX on `diff(y, d) - mean` (simple
    differencing). Forecasts come from statsmodels' own original-scale model.
    """
    sar, seasonal_d, sma, s = seasonal
    if seasonal == NO_SEASON:
        y = arima_series(ar, d, ma, mean, sigma2, seed=100 + index)
    else:
        y = seasonal_series(ar, d, ma, mean, sigma2, 100 + index, seasonal)
    beta = np.array(beta, float)
    x = regressors(beta, 1000 + index, ARMA_LENGTH, 0)
    future_x = regressors(beta, 2000 + index, ARMA_HORIZON, ARMA_LENGTH)
    y = y + x @ beta
    if missing is not None:
        y[missing] = np.nan
    k = d + seasonal_d * s
    raw_w = simple_difference(y - x @ beta, d, seasonal_d, s)
    w = raw_w - mean
    order = (len(ar), 0, len(ma))
    stationary_seasonal = (len(sar), 0, len(sma), s)
    params = [*ar, *ma, *sar, *sma, sigma2]

    def arma_model(series):
        return SARIMAX(series, order=order, seasonal_order=stationary_seasonal, trend="n")

    full = exact_filter(arma_model(w), params)
    head = exact_filter(arma_model(w[: ARMA_SPLIT - k]), params)
    extended = exact_extend(head, w[ARMA_SPLIT - k :], order, params, stationary_seasonal)

    fit_with_mean = (d == 0 and seasonal_d == 0) or mean != 0.0
    raw_y = simple_difference(y, d, seasonal_d, s)
    raw_x = np.column_stack([simple_difference(x[:, j], d, seasonal_d, s) for j in range(len(beta))]) if len(beta) else None
    fit = ARIMA(raw_y, exog=raw_x, order=order, seasonal_order=stationary_seasonal, trend="c" if fit_with_mean else "n").fit()
    fit_params = dict(zip(fit.model.param_names, fit.params))

    exog = x if len(beta) else None
    model, trend_params = forecast_model(y, ar, d, ma, mean, seasonal, exog)
    filtered = exact_filter(model, [*trend_params, *beta, *params])
    forecast = filtered.get_forecast(ARMA_HORIZON, exog=future_x if len(beta) else None)

    arrays = {
        "ar": ar,
        "ma": ma,
        "values": y,
        "innovation_values": full.filter_results.forecasts_error[0],
        "innovation_variances": full.filter_results.forecasts_error_cov[0, 0],
        "extend_values": extended.filter_results.forecasts_error[0],
        "extend_variances": extended.filter_results.forecasts_error_cov[0, 0],
        "forecast_mean": forecast.predicted_mean,
        "forecast_variance": forecast.var_pred_mean,
        "forecast_lower_80": forecast.conf_int(alpha=0.2)[:, 0],
        "forecast_upper_80": forecast.conf_int(alpha=0.2)[:, 1],
        "forecast_lower_95": forecast.conf_int(alpha=0.05)[:, 0],
        "forecast_upper_95": forecast.conf_int(alpha=0.05)[:, 1],
    }
    arrays |= {
        "fit_ar": [fit_params[f"ar.L{i + 1}"] for i in range(len(ar))],
        "fit_ma": [fit_params[f"ma.L{i + 1}"] for i in range(len(ma))],
        "seasonal_ar": sar,
        "seasonal_ma": sma,
        "fit_seasonal_ar": [fit_params[f"ar.S.L{(i + 1) * s}"] for i in range(len(sar))],
        "fit_seasonal_ma": [fit_params[f"ma.S.L{(i + 1) * s}"] for i in range(len(sma))],
        "exog": beta,
        "regressors": x.ravel(),
        "future_regressors": future_x.ravel(),
        "fit_exog": [fit_params[f"x{j + 1}"] for j in range(len(beta))],
    }
    if np.isnan(raw_w).any() or seasonal != NO_SEASON or len(beta):
        hr_lines = ["    hannan_rissanen: None,"]
    else:
        centered = raw_w - (raw_w.mean() if fit_with_mean else 0.0)
        start, _ = hannan_rissanen(centered, ar_order=len(ar), ma_order=len(ma), unbiased=False)
        hr_ar = start.ar_params if len(ar) else []
        hr_ma = start.ma_params if len(ma) else []
        hr_lines = [f"    hannan_rissanen: Some(({rust_array(hr_ar)}, {rust_array(hr_ma)})),"]
    lines = [f"/// {doc}", f"pub const {name}: ArmaCase = ArmaCase {{"]
    lines += [f"    {field}: {rust_array(values)}," for field, values in arrays.items()]
    lines += hr_lines
    lines += [
        f"    fit_with_mean: {str(fit_with_mean).lower()},",
        f"    fit_mean: {rust_float(fit_params.get('const', 0.0))},",
        f"    fit_sigma2: {rust_float(fit_params['sigma2'])},",
        f"    fit_loglike: {rust_float(fit.llf)},",
        f"    fit_aic: {rust_float(fit.aic)},",
        f"    fit_aicc: {rust_float(fit.aicc)},",
        f"    fit_bic: {rust_float(fit.bic)},",
        f"    d: {d},",
        f"    seasonal_d: {seasonal_d},",
        f"    period: {s},",
        f"    mean: {rust_float(mean)},",
        f"    sigma2: {rust_float(sigma2)},",
        f"    loglike: {rust_float(full.llf)},",
        "};",
    ]
    return "\n".join(lines)


def write_arma():
    """Write the ARMA filter reference module."""
    header = f'''//! statsmodels reference vectors for the ARMA filter at fixed parameters.
//!
//! Generated by `scripts/generate_reference.py` with statsmodels {statsmodels.__version__}.
//! Do not edit by hand; rerun the script instead.

/// Where the streaming cases split the raw series: the filter sees `values[..SPLIT]`,
/// then the innovations of `values[SPLIT..]` are compared with `extend`.
pub const SPLIT: usize = {ARMA_SPLIT};

/// One simulated series with statsmodels' filter output at fixed parameters.
///
/// Innovations and the likelihood are `SARIMAX(diff(values, d) - mean, order=(p, 0, q))`.
/// Forecasts are statsmodels' original-scale `ARIMA`/`SARIMAX(values, order=(p, d, q))`.
pub struct ArmaCase {{
    /// The AR coefficients.
    pub ar: &'static [f64],
    /// The differencing order.
    pub d: usize,
    /// The seasonal AR coefficients.
    pub seasonal_ar: &'static [f64],
    /// The seasonal differencing order.
    pub seasonal_d: usize,
    /// The seasonal MA coefficients.
    pub seasonal_ma: &'static [f64],
    /// The season length; 0 for non-seasonal cases.
    pub period: usize,
    /// The mean of the differenced series.
    pub mean: f64,
    /// The MA coefficients.
    pub ma: &'static [f64],
    /// The innovation variance.
    pub sigma2: f64,
    /// The series, {ARMA_LENGTH} values. `NaN` marks a missing value.
    pub values: &'static [f64],
    /// `results.llf`.
    pub loglike: f64,
    /// `filter_results.forecasts_error[0]`, one per differenced value.
    pub innovation_values: &'static [f64],
    /// `filter_results.forecasts_error_cov[0, 0]`, one per differenced value.
    pub innovation_variances: &'static [f64],
    /// `forecasts_error[0]` of `filter(values[..SPLIT]).extend(values[SPLIT..])`.
    pub extend_values: &'static [f64],
    /// `forecasts_error_cov[0, 0]` of the same `extend` call.
    pub extend_variances: &'static [f64],
    /// `get_forecast({ARMA_HORIZON}).predicted_mean` after the whole series.
    pub forecast_mean: &'static [f64],
    /// `get_forecast({ARMA_HORIZON}).var_pred_mean` after the whole series.
    pub forecast_variance: &'static [f64],
    /// `hannan_rissanen(w, p, q, unbiased=False)` AR and MA estimates on `w = diff(values, d)`,
    /// demeaned when `fit_with_mean`. `None` when a value is missing.
    pub hannan_rissanen: Option<(&'static [f64], &'static [f64])>,
    /// Whether the fit estimates a mean: always for `d = 0`, and for `d = 1` with drift.
    pub fit_with_mean: bool,
    /// AR estimates of `ARIMA(diff(values, d), order=(p, 0, q), trend=...).fit()`.
    pub fit_ar: &'static [f64],
    /// MA estimates of the same fit.
    pub fit_ma: &'static [f64],
    /// Seasonal AR estimates of the same fit.
    pub fit_seasonal_ar: &'static [f64],
    /// Seasonal MA estimates of the same fit.
    pub fit_seasonal_ma: &'static [f64],
    /// Regression coefficients; empty without regressors.
    pub exog: &'static [f64],
    /// One row of `exog.len()` regressors per value, row-major.
    pub regressors: &'static [f64],
    /// The regressor rows of the forecast horizon, row-major.
    pub future_regressors: &'static [f64],
    /// Regression estimates of the fit, on the differenced regressors.
    pub fit_exog: &'static [f64],
    /// The `const` estimate of the same fit, or 0.
    pub fit_mean: f64,
    /// The `sigma2` estimate of the same fit.
    pub fit_sigma2: f64,
    /// `llf` of the same fit.
    pub fit_loglike: f64,
    /// `aic` of the same fit.
    pub fit_aic: f64,
    /// `aicc` of the same fit. statsmodels counts missing values in `n`.
    pub fit_aicc: f64,
    /// `bic` of the same fit. statsmodels counts missing values in `n`.
    pub fit_bic: f64,
    /// `get_forecast(...).conf_int(alpha=0.2)[:, 0]`: lower 80% bounds.
    pub forecast_lower_80: &'static [f64],
    /// Upper 80% bounds.
    pub forecast_upper_80: &'static [f64],
    /// `get_forecast(...).conf_int(alpha=0.05)[:, 0]`: lower 95% bounds.
    pub forecast_lower_95: &'static [f64],
    /// Upper 95% bounds.
    pub forecast_upper_95: &'static [f64],
}}

/// All ARMA reference cases.
pub const ALL: &[&ArmaCase] = &[{", ".join(f"&{case[0]}" for case in ARMA_CASES + SEASONAL_CASES + EXOG_CASES)}];
'''
    cases = ARMA_CASES + SEASONAL_CASES + EXOG_CASES
    body = "\n\n".join(arma_case(i, *case) for i, case in enumerate(cases))
    ARMA_OUTPUT.write_text(header + "\n" + body + "\n")


def stl_series(period, cycles, seed):
    """A seasonal pattern on a trend, with noise: `cycles` seasons of length `period`."""
    rng = np.random.default_rng(seed)
    n = period * cycles
    t = np.arange(n)
    pattern = rng.standard_normal(period) * 2.0
    return 10.0 + 0.05 * t + pattern[t % period] + 0.5 * rng.standard_normal(n)


STL_CASES = [
    ("QUARTERLY", "Twelve years of a quarterly pattern on a trend.", 4, 12, 31),
    ("WEEKLY", "Five weeks of a daily pattern (period 7).", 7, 5, 32),
    ("MONTHLY", "Six years of a monthly pattern.", 12, 6, 33),
    ("ELEVEN", "Five cycles of period 11, whose trend window minimum is exactly 21.", 11, 5, 34),
]


def write_stl():
    """Write the STL reference module."""
    header = f'''//! statsmodels `STL` reference decompositions.
//!
//! Generated by `scripts/generate_reference.py` with statsmodels {statsmodels.__version__}.
//! Do not edit by hand; rerun the script instead.

/// A series with `STL(values, period=period).fit()` from statsmodels.
pub struct StlCase {{
    /// The season length.
    pub period: usize,
    /// The series.
    pub values: &'static [f64],
    /// The seasonal component.
    pub seasonal: &'static [f64],
    /// The trend component.
    pub trend: &'static [f64],
}}

/// All STL reference cases.
pub const ALL: &[&StlCase] = &[{", ".join(f"&{c[0]}" for c in STL_CASES)}];
'''
    blocks = []
    for name, doc, period, cycles, seed in STL_CASES:
        y = stl_series(period, cycles, seed)
        result = STL(y, period=period).fit()
        blocks.append("\n".join([
            f"/// {doc}",
            f"pub const {name}: StlCase = StlCase {{",
            f"    period: {period},",
            f"    values: {rust_array(y)},",
            f"    seasonal: {rust_array(result.seasonal)},",
            f"    trend: {rust_array(result.trend)},",
            "};",
        ]))
    STL_OUTPUT.write_text(header + "\n" + "\n\n".join(blocks) + "\n")


def main():
    """Write the reference module."""
    header = f'''//! statsmodels reference vectors.
//!
//! Generated by `scripts/generate_reference.py` with statsmodels {statsmodels.__version__}.
//! Do not edit by hand; rerun the script instead.

/// One input series with the statistics statsmodels computes for it.
pub struct Case {{
    /// The number of lags each vector covers, `min(10, values.len() / 2 - 1)`.
    pub max_lags: usize,
    /// The input series.
    pub values: &'static [f64],
    /// Autocorrelation for lags `0..=max_lags`: `acf(values, nlags=max_lags, fft=False)`.
    pub acf: &'static [f64],
    /// Partial autocorrelation for lags `0..=max_lags`: `pacf(values, method="ldb")`.
    pub pacf: &'static [f64],
    /// Autocovariance divided by `n`: `acovf(values, adjusted=False)`.
    pub acovf_biased: &'static [f64],
    /// Autocovariance divided by `n - lag`: `acovf(values, adjusted=True)`.
    pub acovf_unbiased: &'static [f64],
    /// Variance divided by `n`: `np.var(values, ddof=0)`.
    pub var_biased: f64,
    /// Variance divided by `n - 1`: `np.var(values, ddof=1)`.
    pub var_unbiased: f64,
    /// `acorr_ljungbox(values, lags=[max_lags], model_df=0)` statistic.
    pub ljung_box_statistic: f64,
    /// Its p-value.
    pub ljung_box_p_value: f64,
    /// `jarque_bera(values)` statistic.
    pub jarque_bera_statistic: f64,
    /// Its p-value.
    pub jarque_bera_p_value: f64,
    /// `kpss(values, regression="c", nlags=legacy)` statistic.
    pub kpss_level_statistic: f64,
    /// Its p-value.
    pub kpss_level_p_value: f64,
    /// `kpss(values, regression="ct", nlags=legacy)` statistic.
    pub kpss_trend_statistic: f64,
    /// Its p-value.
    pub kpss_trend_p_value: f64,
    /// The AR order of the fits below, `min(3, max_lags)`.
    pub ar_order: usize,
    /// `yule_walker(values, ar_order, method="mle")` coefficients.
    pub yw_mle_phi: &'static [f64],
    /// The square of the sigma that `yule_walker(..., method="mle")` returns.
    pub yw_mle_sigma2: f64,
    /// `yule_walker(values, ar_order, method="adjusted")` coefficients.
    pub yw_adjusted_phi: &'static [f64],
    /// The square of the sigma that `yule_walker(..., method="adjusted")` returns;
    /// `NaN` where statsmodels found `sigma2 <= 0`.
    pub yw_adjusted_sigma2: f64,
    /// `burg(values, ar_order)` coefficients.
    pub burg_phi: &'static [f64],
    /// `burg(values, ar_order)` innovation variance.
    pub burg_sigma2: f64,
}}

/// All reference cases.
pub const ALL: &[&Case] = &[{", ".join(f"&{name}" for name, _, _ in SERIES)}];
'''
    body = "\n\n".join(case(name, doc, y) for name, doc, y in SERIES)
    OUTPUT.write_text(header + "\n" + body + "\n")
    write_arma()
    write_stl()


if __name__ == "__main__":
    main()

"""Generate statsmodels reference vectors for the integration tests.

Writes `tests/it/reference_data.rs` and `tests/it/arma_data.rs`. Run from the repository root:

    python3 -I scripts/generate_reference.py

Requires numpy and statsmodels (tested with statsmodels 0.14).
"""

import pathlib

import numpy as np
import statsmodels
from statsmodels.regression.linear_model import burg, yule_walker
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, acovf, pacf

TESTS = pathlib.Path(__file__).resolve().parent.parent / "tests" / "it"
OUTPUT = TESTS / "reference_data.rs"
ARMA_OUTPUT = TESTS / "arma_data.rs"
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
    scalars = {
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

ARMA_CASES = [
    ("AR1", "AR(1).", [0.6], [], 1.0, None),
    ("MA1", "MA(1).", [], [0.5], 1.3, None),
    ("ARMA21", "ARMA(2, 1).", [0.5, -0.2], [0.4], 1.3, None),
    ("ARMA31", "ARMA(3, 1): more AR than MA lags, so the selection vector is padded.", [0.3, 0.2, -0.25], [0.5], 0.8, None),
    ("ARMA12", "ARMA(1, 2): more MA than AR lags, so the transition column is padded.", [0.7], [0.4, -0.3], 2.0, None),
    ("WHITE", "ARMA(0, 0): white noise with one state.", [], [], 1.5, None),
    ("MISSING", "ARMA(1, 1) with the value at index 10 missing.", [0.5], [0.3], 1.0, 10),
]


def arma_series(ar, ma, sigma2, seed):
    """Simulate ARMA_LENGTH values after a burn-in, so the start is near stationarity."""
    rng = np.random.default_rng(seed)
    burn = 200
    e = rng.standard_normal(ARMA_LENGTH + burn) * np.sqrt(sigma2)
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


def exact_extend(head, tail, order, params):
    """`head.extend(tail)` without the steady-state shortcut.

    `extend` clones the model, which resets the tolerance, so this repeats what it
    does: a new model on `tail`, known-initialised from `head`'s last predicted state.
    """
    model = SARIMAX(tail, order=order, trend="n")
    model.ssm.initialize_known(head.predicted_state[:, -1], head.predicted_state_cov[:, :, -1])
    return exact_filter(model, params)


def arma_case(index, name, doc, ar, ma, sigma2, missing):
    """Render one `ArmaCase` constant from a statsmodels SARIMAX filter at fixed parameters."""
    y = arma_series(ar, ma, sigma2, seed=100 + index)
    if missing is not None:
        y[missing] = np.nan
    order = (len(ar), 0, len(ma))
    params = [*ar, *ma, sigma2]
    full = exact_filter(SARIMAX(y, order=order, trend="n"), params)
    head = exact_filter(SARIMAX(y[:ARMA_SPLIT], order=order, trend="n"), params)
    extended = exact_extend(head, y[ARMA_SPLIT:], order, params)
    forecast = full.get_forecast(ARMA_HORIZON)

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
    }
    lines = [f"/// {doc}", f"pub const {name}: ArmaCase = ArmaCase {{"]
    lines += [f"    {field}: {rust_array(values)}," for field, values in arrays.items()]
    lines += [f"    sigma2: {rust_float(sigma2)},", f"    loglike: {rust_float(full.llf)},", "};"]
    return "\n".join(lines)


def write_arma():
    """Write the ARMA filter reference module."""
    header = f'''//! statsmodels reference vectors for the ARMA filter at fixed parameters.
//!
//! Generated by `scripts/generate_reference.py` with statsmodels {statsmodels.__version__}.
//! Do not edit by hand; rerun the script instead.

/// Where the streaming cases split the series: `filter(values[..SPLIT])`, then `extend(values[SPLIT..])`.
pub const SPLIT: usize = {ARMA_SPLIT};

/// One simulated series with what `SARIMAX(values, order=(p, 0, q), trend="n").filter(params)` returns.
pub struct ArmaCase {{
    /// The AR coefficients.
    pub ar: &'static [f64],
    /// The MA coefficients.
    pub ma: &'static [f64],
    /// The innovation variance.
    pub sigma2: f64,
    /// The series, {ARMA_LENGTH} values. `NaN` marks a missing value.
    pub values: &'static [f64],
    /// `results.llf`.
    pub loglike: f64,
    /// `filter_results.forecasts_error[0]`, one per value.
    pub innovation_values: &'static [f64],
    /// `filter_results.forecasts_error_cov[0, 0]`, one per value.
    pub innovation_variances: &'static [f64],
    /// `forecasts_error[0]` of `filter(values[..SPLIT]).extend(values[SPLIT..])`.
    pub extend_values: &'static [f64],
    /// `forecasts_error_cov[0, 0]` of the same `extend` call.
    pub extend_variances: &'static [f64],
    /// `get_forecast({ARMA_HORIZON}).predicted_mean` after the whole series.
    pub forecast_mean: &'static [f64],
    /// `get_forecast({ARMA_HORIZON}).var_pred_mean` after the whole series.
    pub forecast_variance: &'static [f64],
}}

/// All ARMA reference cases.
pub const ALL: &[&ArmaCase] = &[{", ".join(f"&{case[0]}" for case in ARMA_CASES)}];
'''
    body = "\n\n".join(arma_case(i, *case) for i, case in enumerate(ARMA_CASES))
    ARMA_OUTPUT.write_text(header + "\n" + body + "\n")


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


if __name__ == "__main__":
    main()

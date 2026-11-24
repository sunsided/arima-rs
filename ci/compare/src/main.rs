//! Fits the same models with backshift, foresight and anofox-forecast and prints a
//! Markdown table: fit time, and the exact Gaussian log-likelihood (backshift's
//! `loglike`, simple differencing) at each crate's own estimates, so every crate is
//! scored by the same yardstick.
//!
//! Each repeat runs every crate back to back, so they share the machine's load at the
//! time, and the table reports the fastest repeat: the one least disturbed by it.

use anofox_forecast::models::Forecaster;
mod ets;
mod garch;
mod intermittent;
mod theta;
mod var;

use backshift::{Arima, ArimaSpec, AutoArima, Method, Seasonal, loglike};
use std::time::Instant;

/// One benchmark model.
struct Case {
    /// A label for the table.
    name: &'static str,
    /// The series.
    values: Vec<f64>,
    /// `(p, d, q)`.
    order: (usize, usize, usize),
    /// `(P, D, Q, s)`; `s = 0` for none.
    seasonal: (usize, usize, usize, usize),
}

/// Estimates in backshift's conventions.
struct Estimate {
    /// AR coefficients.
    ar: Vec<f64>,
    /// MA coefficients.
    ma: Vec<f64>,
    /// Seasonal AR coefficients.
    seasonal_ar: Vec<f64>,
    /// Seasonal MA coefficients.
    seasonal_ma: Vec<f64>,
    /// Mean of the differenced series.
    mean: f64,
    /// Innovation variance.
    sigma2: f64,
}

/// A crate's fit of one case, in backshift's conventions.
type Fit = fn(&Case) -> Option<Estimate>;

/// How often each fit is repeated; the table reports the fastest repeat.
const REPEATS: usize = 15;

fn main() {
    let cases = [
        Case {
            name: "ARMA(2,1), n=300",
            values: simulate(&[0.5, -0.3], &[0.4], 0, 300, 1),
            order: (2, 0, 1),
            seasonal: (0, 0, 0, 0),
        },
        Case {
            name: "ARIMA(1,1,1), n=300",
            values: simulate(&[0.6], &[0.3], 1, 300, 2),
            order: (1, 1, 1),
            seasonal: (0, 0, 0, 0),
        },
        Case {
            name: "airline (0,1,1)(0,1,1,12), n=144",
            values: airline(144, 3),
            order: (0, 1, 1),
            seasonal: (0, 1, 1, 12),
        },
    ];

    println!("| Model | Crate | Fit time | Exact log-likelihood at its estimates |");
    println!("|---|---|---|---|");
    let crates: [(&str, Fit); 4] = [
        ("backshift", |case| fit_backshift(case, Method::Exact)),
        ("backshift (approximate)", |case| {
            fit_backshift(case, Method::Css)
        }),
        ("foresight 0.7.3", fit_foresight),
        ("anofox-forecast 0.21.0", fit_anofox),
    ];
    for case in &cases {
        let mut fastest = [f64::INFINITY; 4];
        let mut estimates: [Option<Estimate>; 4] = [None, None, None, None];
        for _ in 0..REPEATS {
            for (k, (_, fit)) in crates.iter().enumerate() {
                let start = Instant::now();
                estimates[k] = fit(case);
                fastest[k] = fastest[k].min(start.elapsed().as_secs_f64());
            }
        }

        for (k, (name, _)) in crates.iter().enumerate() {
            let llf = estimates[k]
                .as_ref()
                .map_or(f64::NAN, |e| exact_loglike(case, e));
            println!(
                "| {} | {name} | {} | {llf:.4} |",
                case.name,
                milliseconds(fastest[k])
            );
        }
    }

    println!();
    println!("| Series | Crate | Search time | Selected order |");
    println!("|---|---|---|---|");
    for case in &cases {
        auto_report(case);
    }

    ets::report();
    var::report();
    theta::report();
    intermittent::report();
    garch::report();
    streaming();
}

/// backshift's own fit, exact or by conditional sum of squares only.
fn fit_backshift(case: &Case, method: Method) -> Option<Estimate> {
    let (p, d, q) = case.order;
    let (sp, sd, sq, s) = case.seasonal;
    let model = Arima::new(p, d, q);
    let model = if s > 0 {
        model.seasonal(sp, sd, sq, s)
    } else {
        model
    };
    let fitted = model
        .mean(with_constant(case))
        .method(method)
        .fit(&case.values)
        .ok()?;
    Some(Estimate {
        ar: fitted.ar,
        ma: fitted.ma,
        seasonal_ar: fitted.seasonal_ar,
        seasonal_ma: fitted.seasonal_ma,
        mean: fitted.mean,
        sigma2: fitted.sigma2,
    })
}

/// foresight's `Arima::estimate`.
fn fit_foresight(case: &Case) -> Option<Estimate> {
    let (p, d, q) = case.order;
    let (sp, sd, sq, s) = case.seasonal;
    let series = foresight::Series::new(&case.values, s.max(1));
    let model = foresight::models::Arima::new(p, d, q)
        .seasonal(sp, sd, sq)
        .constant(with_constant(case));
    let fit = model.estimate(series)?;
    Some(Estimate {
        ar: fit.ar,
        ma: fit.ma,
        seasonal_ar: fit.seasonal_ar,
        seasonal_ma: fit.seasonal_ma,
        mean: fit.constant.unwrap_or(0.0),
        sigma2: fit.sigma2,
    })
}

/// Whether the model includes a mean or drift: for `d + D <= 1`, as anofox-forecast
/// does by default and R's `auto.arima` allows, so all three crates fit the same model.
fn with_constant(case: &Case) -> bool {
    case.order.1 + case.seasonal.1 <= 1
}

/// anofox-forecast's `ARIMA` or `SARIMA` (with its default intercept).
fn fit_anofox(case: &Case) -> Option<Estimate> {
    let (p, d, q) = case.order;
    let (sp, sd, sq, s) = case.seasonal;
    let series = anofox_series(&case.values);
    if s == 0 {
        let mut model = anofox_forecast::models::arima::ARIMA::new(p, d, q);
        model.fit(&series).ok()?;
        return Some(Estimate {
            ar: model.ar_coefficients().to_vec(),
            ma: model.ma_coefficients().to_vec(),
            seasonal_ar: vec![],
            seasonal_ma: vec![],
            mean: model.intercept(),
            sigma2: model.sigma2()?,
        });
    }

    let mut model = anofox_forecast::models::arima::SARIMA::new(p, d, q, sp, sd, sq, s);
    model.fit(&series).ok()?;
    Some(Estimate {
        ar: model.ar_coefficients().to_vec(),
        ma: model.ma_coefficients().to_vec(),
        seasonal_ar: model.seasonal_ar_coefficients().to_vec(),
        seasonal_ma: model.seasonal_ma_coefficients().to_vec(),
        mean: 0.0,
        sigma2: model.sigma2()?,
    })
}

/// A daily-stamped anofox series.
pub(crate) fn anofox_series(values: &[f64]) -> anofox_forecast::core::TimeSeries {
    let start = chrono::DateTime::from_timestamp(0, 0).expect("epoch");
    let stamps = (0..values.len())
        .map(|i| start + chrono::Duration::days(i as i64))
        .collect();
    anofox_forecast::core::TimeSeries::univariate(stamps, values.to_vec()).expect("valid series")
}

/// backshift's exact log-likelihood of the case at the given estimates.
fn exact_loglike(case: &Case, e: &Estimate) -> f64 {
    let (_, d, _) = case.order;
    let (_, sd, _, s) = case.seasonal;
    let spec = ArimaSpec {
        ar: &e.ar,
        d,
        ma: &e.ma,
        seasonal: Seasonal {
            ar: &e.seasonal_ar,
            d: sd,
            ma: &e.seasonal_ma,
            period: s,
        },
        mean: e.mean,
        sigma2: e.sigma2,
        ..ArimaSpec::default()
    };
    loglike(&case.values, &spec).unwrap_or(f64::NAN)
}

/// Times the three crates' fully automatic searches (each chooses its own `d` and `D`),
/// interleaved, and prints the fastest repeat and the selected model of each.
fn auto_report(case: &Case) {
    let s = case.seasonal.3;
    let mut fastest = [f64::INFINITY; 4];
    let mut orders = [String::new(), String::new(), String::new(), String::new()];
    let format = |p, d, q, sp, sd, sq| format!("({p},{d},{q})({sp},{sd},{sq},{s})");
    for _ in 0..5 {
        for (k, approximation) in [(0, false), (3, true)] {
            let start = Instant::now();
            let search = if s > 0 {
                AutoArima::new().seasonal(s)
            } else {
                AutoArima::new()
            };
            let f = search
                .approximation(approximation)
                .fit(&case.values)
                .expect("search")
                .fitted;
            fastest[k] = fastest[k].min(start.elapsed().as_secs_f64());
            orders[k] = format(
                f.ar.len(),
                f.d,
                f.ma.len(),
                f.seasonal_ar.len(),
                f.seasonal_d,
                f.seasonal_ma.len(),
            );
        }

        let start = Instant::now();
        let series = foresight::Series::new(&case.values, s.max(1));
        let best = foresight::models::AutoArima::new().select(series);
        fastest[1] = fastest[1].min(start.elapsed().as_secs_f64());
        orders[1] = best.map_or("failed".into(), |f| {
            let ((p, d, q), (sp, sd, sq)) = (f.order(), f.seasonal_order());
            format(p, d, q, sp, sd, sq)
        });

        let start = Instant::now();
        let mut model = if s > 0 {
            anofox_forecast::models::arima::AutoARIMA::seasonal(s)
        } else {
            anofox_forecast::models::arima::AutoARIMA::new()
        };
        let fitted = model.fit(&anofox_series(&case.values));
        fastest[2] = fastest[2].min(start.elapsed().as_secs_f64());
        orders[2] = match (fitted, model.selected_full_order()) {
            (Ok(()), Some(o)) => format(o.p, o.d, o.q, o.cap_p, o.cap_d, o.cap_q),
            _ => "failed".into(),
        };
    }

    let names = [
        "backshift",
        "foresight 0.7.3",
        "anofox-forecast 0.21.0",
        "backshift (approximate)",
    ];
    for k in [0, 3, 1, 2] {
        println!(
            "| {} | {} | {} | {} |",
            case.name,
            names[k],
            milliseconds(fastest[k]),
            orders[k]
        );
    }
}

/// Streaming throughput of the heap-free filter, which the other crates do not have.
fn streaming() {
    let values = airline(10_000, 4);
    let spec = ArimaSpec {
        d: 1,
        ma: &[-0.4],
        seasonal: Seasonal {
            ar: &[],
            d: 1,
            ma: &[-0.6],
            period: 12,
        },
        ..ArimaSpec::default()
    };
    let mut filter = backshift::Filter::<27>::new(&spec).expect("airline fits in 27 states");
    let start = Instant::now();
    let mut sum = 0.0;
    for &y in &values {
        if let Some(innovation) = filter.update(y) {
            sum += innovation.value;
        }
    }
    let per_update = start.elapsed().as_secs_f64() / values.len() as f64;
    std::hint::black_box(sum);
    println!();
    println!(
        "Streaming: `Filter::<27>::update` for the airline model takes {:.2} µs per value ({} bytes of state).",
        per_update * 1e6,
        std::mem::size_of::<backshift::Filter<27>>()
    );
    println!(
        "Filter sizes: R=4 {} bytes, R=8 {} bytes, R=16 {} bytes, R=27 {} bytes.",
        std::mem::size_of::<backshift::Filter<4>>(),
        std::mem::size_of::<backshift::Filter<8>>(),
        std::mem::size_of::<backshift::Filter<16>>(),
        std::mem::size_of::<backshift::Filter<27>>()
    );
}

/// Seconds as milliseconds with two decimals.
pub(crate) fn milliseconds(seconds: f64) -> String {
    format!("{:.2} ms", seconds * 1e3)
}

/// Standard normal draws from a fixed linear congruential generator (Box–Muller).
pub(crate) fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    (0..n)
        .map(|_| (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos())
        .collect()
}

/// An ARIMA(p, d, q) series after a burn-in, integrated `d` times.
fn simulate(ar: &[f64], ma: &[f64], d: usize, n: usize, seed: u64) -> Vec<f64> {
    let burn = 300;
    let e = normals(n + burn, seed);
    let mut w = vec![0.0; n + burn];
    for t in 0..w.len() {
        w[t] = e[t];
        for (j, phi) in ar.iter().enumerate() {
            if t > j {
                w[t] += phi * w[t - j - 1];
            }
        }
        for (j, theta) in ma.iter().enumerate() {
            if t > j {
                w[t] += theta * e[t - j - 1];
            }
        }
    }

    let mut y = w[burn..].to_vec();
    for _ in 0..d {
        let mut sum = 10.0;
        for v in &mut y {
            sum += *v;
            *v = sum;
        }
    }
    y
}

/// An airline-model series: `(1 - B)(1 - B^12) y_t = (1 - 0.4 B)(1 - 0.6 B^12) e_t`.
fn airline(n: usize, seed: u64) -> Vec<f64> {
    let e = normals(n, seed);
    let mut y = vec![0.0; n];
    for t in 0..n {
        let lag = |k: usize| if t >= k { e[t - k] } else { 0.0 };
        let past = |k: usize| if t >= k { y[t - k] } else { 0.0 };
        let w = e[t] - 0.4 * lag(1) - 0.6 * lag(12) + 0.24 * lag(13);
        y[t] = w + past(1) + past(12) - past(13);
    }
    y.iter().map(|v| v + 100.0).collect()
}

/// The fastest of `repeats` runs of `f`, in seconds, with the last result.
pub(crate) fn fastest<T>(repeats: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut result = f();
    for _ in 0..repeats {
        let start = Instant::now();
        result = f();
        best = best.min(start.elapsed().as_secs_f64());
    }
    (best, result)
}

/// The root mean squared error of `forecast` against `actual`, to three decimals.
pub(crate) fn rmse(forecast: &[f64], actual: &[f64]) -> String {
    let n = f64::from(u32::try_from(actual.len()).expect("short horizon"));
    let sse: f64 = forecast
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    format!("{:.3}", (sse / n).sqrt())
}

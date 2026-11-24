//! ETS fits and searches of backshift, foresight and anofox-forecast, scored on held-out values.
//!
//! The crates parameterise their initial states differently, so a common likelihood is not
//! available; the forecast error on the last `h` values, which no crate sees, is.

use crate::{fastest, milliseconds, normals, rmse};
use anofox_forecast::models::Forecaster;

/// One ETS benchmark series and the structure to fit to it.
struct Case {
    /// The row label.
    name: &'static str,
    /// The training values.
    train: Vec<f64>,
    /// The held-out values that follow them.
    test: Vec<f64>,
    /// The season length; 1 without a season.
    period: usize,
    /// Whether the fixed-structure fit has a trend, and whether it is damped.
    trend: (bool, bool),
}

/// How often each fit is repeated; the table reports the fastest repeat.
const REPEATS: usize = 15;

/// Prints the fixed-structure and automatic ETS tables.
pub fn report() {
    let cases = cases();
    println!();
    println!("| ETS model | Crate | Fit time | RMSE on held-out values |");
    println!("|---|---|---|---|");
    for case in &cases {
        for (name, fit) in fits() {
            let (seconds, forecast) = fastest(REPEATS, || fit(case));
            println!(
                "| {} | {name} | {} | {} |",
                case.name,
                milliseconds(seconds),
                score(forecast.as_deref(), &case.test)
            );
        }
    }

    println!();
    println!("| Series | Crate | Automatic ETS time | Selected | RMSE on held-out values |");
    println!("|---|---|---|---|---|");
    for case in &cases {
        for (name, search) in searches() {
            let (seconds, result) = fastest(5, || search(case));
            let (label, forecast) =
                result.map_or(("failed".to_string(), None), |(l, f)| (l, Some(f)));
            println!(
                "| {} | {name} | {} | {label} | {} |",
                case.name,
                milliseconds(seconds),
                score(forecast.as_deref(), &case.test)
            );
        }
    }
}

/// The benchmark series: a damped trend, and a trend with an additive monthly season.
fn cases() -> Vec<Case> {
    let split = |values: Vec<f64>, h: usize| {
        let cut = values.len() - h;
        (values[..cut].to_vec(), values[cut..].to_vec())
    };
    let e = normals(130, 21);
    let mut level = 20.0;
    let mut slope = 0.8;
    let damped: Vec<f64> = e
        .iter()
        .map(|noise| {
            slope *= 0.97;
            level += slope + 0.3 * noise;
            level + noise
        })
        .collect();
    let (train, test) = split(damped, 10);
    let damped = Case {
        name: "damped trend, n=120",
        train,
        test,
        period: 1,
        trend: (true, true),
    };

    let e = normals(156, 22);
    let seasonal: Vec<f64> = (0..156)
        .map(|t| {
            let t_f = f64::from(u32::try_from(t).expect("short series"));
            50.0 + 0.25 * t_f + 6.0 * (std::f64::consts::TAU * t_f / 12.0).sin() + e[t]
        })
        .collect();
    let (train, test) = split(seasonal, 12);
    let monthly = Case {
        name: "trend + monthly season, n=144",
        train,
        test,
        period: 12,
        trend: (true, false),
    };
    vec![damped, monthly]
}

/// A crate's fixed-structure fit: its forecasts of the held-out values.
type Fit = fn(&Case) -> Option<Vec<f64>>;

/// A crate's automatic search: the selected model and its forecasts.
type Search = fn(&Case) -> Option<(String, Vec<f64>)>;

/// The fixed-structure fits of the three crates.
fn fits() -> [(&'static str, Fit); 3] {
    [
        ("backshift", |case| {
            let model = match case.trend {
                (false, _) => backshift::Ets::new(),
                (true, false) => backshift::Ets::new().trend(),
                (true, true) => backshift::Ets::new().damped(),
            };
            let fit = model.seasonal(case.period).fit(&case.train).ok()?;
            Some(backshift_forecast(&fit, case.test.len()))
        }),
        ("foresight 0.7.3", |case| {
            use foresight::models::{ErrorKind, Season, Trend};
            let trend = match case.trend {
                (false, _) => Trend::None,
                (true, false) => Trend::Additive,
                (true, true) => Trend::Damped,
            };
            let season = if case.period > 1 {
                Season::Additive
            } else {
                Season::None
            };
            let model = foresight::models::Ets::new(ErrorKind::Additive, trend, season);
            let fit = model.estimate(foresight::Series::new(&case.train, case.period))?;
            Some(foresight::Fitted::forecast(&fit, case.test.len()))
        }),
        ("anofox-forecast 0.21.0", |case| {
            use anofox_forecast::models::exponential::{ETS, ETSSpec};
            let spec = match (case.trend, case.period > 1) {
                ((true, true), false) => ETSSpec::aadn(),
                ((true, false), true) => ETSSpec::aaa(),
                ((true, true), true) => ETSSpec::aada(),
                ((true, false), false) => ETSSpec::aan(),
                ((false, _), true) => ETSSpec::ana(),
                ((false, _), false) => ETSSpec::ann(),
            };
            let mut model = ETS::new(spec, case.period);
            model.fit(&crate::anofox_series(&case.train)).ok()?;
            Some(model.predict(case.test.len()).ok()?.primary().to_vec())
        }),
    ]
}

/// The automatic searches of the three crates, each over its own default model set.
fn searches() -> [(&'static str, Search); 3] {
    [
        ("backshift", |case| {
            let fit = backshift::AutoEts::new()
                .seasonal(case.period)
                .fit(&case.train)
                .ok()?;
            Some((
                fit.label().to_string(),
                backshift_forecast(&fit, case.test.len()),
            ))
        }),
        ("foresight 0.7.3", |case| {
            let fit = foresight::models::AutoEts::new()
                .select(foresight::Series::new(&case.train, case.period))?;
            Some((
                fit.model().code(),
                foresight::Fitted::forecast(&fit, case.test.len()),
            ))
        }),
        ("anofox-forecast 0.21.0", |case| {
            use anofox_forecast::models::exponential::AutoETS;
            let mut model = if case.period > 1 {
                AutoETS::with_period(case.period)
            } else {
                AutoETS::non_seasonal()
            };
            model.fit(&crate::anofox_series(&case.train)).ok()?;
            let label = model
                .selected_spec()
                .map_or_else(|| "?".to_string(), |s| s.short_name());
            Some((
                label,
                model.predict(case.test.len()).ok()?.primary().to_vec(),
            ))
        }),
    ]
}

/// backshift's forecasts from the fit's final states.
fn backshift_forecast(fit: &backshift::EtsFit, h: usize) -> Vec<f64> {
    let filter = backshift::EtsFilter::<24>::new(&fit.spec()).expect("fitted models are valid");
    let (mut mean, mut variance) = (vec![0.0; h], vec![0.0; h]);
    filter.forecast_into(&mut mean, &mut variance);
    mean
}

/// The RMSE of a forecast that may have failed.
fn score(forecast: Option<&[f64]>, actual: &[f64]) -> String {
    forecast.map_or_else(|| "failed".to_string(), |f| rmse(f, actual))
}

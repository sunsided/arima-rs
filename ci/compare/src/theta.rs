//! Theta forecasts of backshift, foresight and anofox-forecast, scored on held-out values.

use crate::{fastest, milliseconds, normals, rmse};
use anofox_forecast::models::Forecaster;
use foresight::Model;

/// Prints the Theta table.
pub fn report() {
    println!();
    println!("| Theta | Crate | Fit and forecast time | RMSE on held-out values |");
    println!("|---|---|---|---|");
    for (name, period, values) in [
        ("trend, n=120", 1, series(132, 1, 51)),
        ("monthly season, n=144", 12, series(156, 12, 52)),
    ] {
        let h = 12;
        let (train, test) = values.split_at(values.len() - h);

        let (seconds, forecast) = fastest(15, || {
            let fit = backshift::Theta::new()
                .seasonal(period)
                .fit(train)
                .expect("fit");
            let filter = backshift::ThetaFilter::<12>::new(&fit.spec()).expect("valid");
            let (mut mean, mut variance) = (vec![0.0; h], vec![0.0; h]);
            filter.forecast_into(&mut mean, &mut variance);
            mean
        });
        println!(
            "| {name} | backshift | {} | {} |",
            milliseconds(seconds),
            rmse(&forecast, test)
        );

        let (seconds, forecast) = fastest(15, || {
            foresight::models::Theta
                .forecast(foresight::Series::new(train, period), h)
                .expect("forecast")
        });
        println!(
            "| {name} | foresight 0.7.3 | {} | {} |",
            milliseconds(seconds),
            rmse(&forecast, test)
        );

        let (seconds, forecast) = fastest(15, || {
            let mut model = if period > 1 {
                anofox_forecast::models::theta::Theta::seasonal(period)
            } else {
                anofox_forecast::models::theta::Theta::new()
            };
            model.fit(&crate::anofox_series(train)).expect("fit");
            model.predict(h).expect("forecast").primary().to_vec()
        });
        println!(
            "| {name} | anofox-forecast 0.21.0 | {} | {} |",
            milliseconds(seconds),
            rmse(&forecast, test)
        );
    }
}

/// `n` positive values with a level, a slope and, for `period > 1`, a multiplicative season.
fn series(n: usize, period: usize, seed: u64) -> Vec<f64> {
    let e = normals(n, seed);
    let mut level = 100.0;
    (0..n)
        .map(|t| {
            level += 0.3 + 0.8 * e[t];
            let phase = std::f64::consts::TAU
                * f64::from(u32::try_from(t % period.max(1)).expect("small"))
                / f64::from(u32::try_from(period.max(1)).expect("small"));
            let season = if period > 1 {
                1.0 + 0.12 * phase.sin()
            } else {
                1.0
            };
            level * season
        })
        .collect()
}

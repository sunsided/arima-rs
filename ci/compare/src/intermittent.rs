//! Intermittent-demand forecasts of backshift, foresight and anofox-forecast, scored on
//! held-out periods.

use crate::{fastest, milliseconds, normals, rmse};
use anofox_forecast::models::Forecaster;
use foresight::Model;

/// The held-out periods.
const H: usize = 12;

/// A crate's forecast of the next `H` periods from the training values.
type Forecast = fn(&[f64]) -> Vec<f64>;

/// Prints the intermittent-demand table.
pub fn report() {
    let values = series(162);
    let (train, test) = values.split_at(values.len() - H);
    println!();
    println!(
        "| Intermittent demand, n=150 | Crate | Fit and forecast time | RMSE on held-out values |"
    );
    println!("|---|---|---|---|");
    let methods: [(&str, &str, Forecast); 6] = [
        ("SBA, optimised alpha", "backshift", |train| {
            let filter = backshift::CrostonFilter::optimised(train, backshift::CrostonVariant::Sba)
                .expect("fit");
            vec![filter.forecast(); H]
        }),
        ("SBA, optimised alpha", "foresight 0.7.3", |train| {
            let model = foresight::models::Croston::new()
                .variant(foresight::models::Intermittent::Sba)
                .optimised();
            model
                .forecast(foresight::Series::non_seasonal(train), H)
                .expect("forecast")
        }),
        ("SBA, optimised alpha", "anofox-forecast 0.21.0", |train| {
            let mut model = anofox_forecast::models::intermittent::Croston::new().sba_optimized();
            model.fit(&crate::anofox_series(train)).expect("fit");
            model.predict(H).expect("forecast").primary().to_vec()
        }),
        ("TSB", "backshift (optimised)", |train| {
            let filter = backshift::TsbFilter::optimised(train).expect("fit");
            vec![filter.forecast(); H]
        }),
        ("TSB", "foresight 0.7.3", |train| {
            let model =
                foresight::models::Croston::new().variant(foresight::models::Intermittent::Tsb);
            model
                .forecast(foresight::Series::non_seasonal(train), H)
                .expect("forecast")
        }),
        ("TSB", "anofox-forecast 0.21.0 (fixed 0.1)", |train| {
            let mut model = anofox_forecast::models::intermittent::TSB::new();
            model.fit(&crate::anofox_series(train)).expect("fit");
            model.predict(H).expect("forecast").primary().to_vec()
        }),
    ];
    for (method, name, forecast) in methods {
        let (seconds, values) = fastest(15, || forecast(train));
        println!(
            "| {method} | {name} | {} | {} |",
            milliseconds(seconds),
            rmse(&values, test)
        );
    }
}

/// `n` periods of intermittent demand: about one period in four has a demand of 1 to 8.
fn series(n: usize) -> Vec<f64> {
    let e = normals(2 * n, 71);
    (0..n)
        .map(|t| {
            if e[2 * t] > 0.67 {
                (3.0 + 2.0 * e[2 * t + 1]).clamp(1.0, 8.0).round()
            } else {
                0.0
            }
        })
        .collect()
}

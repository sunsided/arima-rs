//! VAR fits of backshift and anofox-forecast (foresight has none), scored on held-out rows.

use crate::{milliseconds, normals};
use std::time::Instant;

/// The number of series.
const K: usize = 3;

/// The lag order.
const P: usize = 2;

/// The held-out rows.
const H: usize = 12;

/// Prints the VAR table.
pub fn report() {
    let (train, test) = series(500);
    println!();
    println!(
        "| VAR(2), 3 series, n=500 | Crate | Fit and forecast time | RMSE on held-out values |"
    );
    println!("|---|---|---|---|");

    let (seconds, forecast) = fastest(15, || {
        let fit = backshift::Var::new(P).fit(&train, K).expect("fit");
        fit.forecast(H, &[]).mean
    });
    println!(
        "| | backshift | {} | {} |",
        milliseconds(seconds),
        rmse(&forecast, &test)
    );

    let columns: Vec<Vec<f64>> = (0..K)
        .map(|i| train.iter().skip(i).step_by(K).copied().collect())
        .collect();
    let (seconds, forecast) = fastest(15, || {
        let mut model = anofox_forecast::models::var::VAR::new(P);
        model.fit(&columns).expect("fit");
        let by_series = model.predict(H).expect("forecast");
        // Back to row-major.
        (0..H)
            .flat_map(|h| by_series.iter().map(move |s| s[h]))
            .collect::<Vec<f64>>()
    });
    println!(
        "| | anofox-forecast 0.21.0 | {} | {} |",
        milliseconds(seconds),
        rmse(&forecast, &test)
    );
}

/// A simulated VAR(2) of `n` training rows and `H` held-out rows, row-major.
fn series(n: usize) -> (Vec<f64>, Vec<f64>) {
    let a1 = [[0.5, 0.1, 0.0], [0.0, 0.4, 0.2], [0.1, 0.0, 0.3]];
    let a2 = [[-0.2, 0.0, 0.1], [0.1, -0.1, 0.0], [0.0, 0.1, -0.1]];
    let rows = n + H + 100;
    let e = normals(rows * K, 31);
    let mut y = vec![0.0; rows * K];
    for t in 2..rows {
        for i in 0..K {
            let mut value = 1.0 + e[t * K + i];
            for j in 0..K {
                value += a1[i][j] * y[(t - 1) * K + j] + a2[i][j] * y[(t - 2) * K + j];
            }
            y[t * K + i] = value;
        }
    }

    let start = 100 * K;
    let cut = start + n * K;
    (y[start..cut].to_vec(), y[cut..].to_vec())
}

/// The fastest of `repeats` runs of `f`, with the last result.
fn fastest<T>(repeats: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut result = f();
    for _ in 0..repeats {
        let start = Instant::now();
        result = f();
        best = best.min(start.elapsed().as_secs_f64());
    }
    (best, result)
}

/// The root mean squared error of `forecast` against `actual`.
fn rmse(forecast: &[f64], actual: &[f64]) -> String {
    let n = f64::from(u32::try_from(actual.len()).expect("short horizon"));
    let sse: f64 = forecast
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    format!("{:.3}", (sse / n).sqrt())
}

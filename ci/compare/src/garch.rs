//! GARCH(1,1) fits of backshift and anofox-forecast (foresight has none), scored by one
//! likelihood.

use crate::{fastest, milliseconds, normals};
use anofox_forecast::models::Forecaster;
use backshift::{GarchFilter, GarchSpec};

/// The simulated model: `omega`, `alpha` and `beta`.
const TRUE: [f64; 3] = [0.05, 0.1, 0.85];

/// Prints the GARCH table.
pub fn report() {
    let x = series(1000);
    println!();
    println!(
        "| GARCH(1,1), n=1000 | Crate | Fit time | Log-likelihood at its estimate (tseries' definition) | omega, alpha, beta |"
    );
    println!("|---|---|---|---|---|");

    let (seconds, fit) = fastest(15, || backshift::Garch::new().fit(&x).expect("fit"));
    row("backshift", seconds, &x, [fit.omega, fit.alpha, fit.beta]);

    let (seconds, model) = fastest(15, || {
        let mut model = anofox_forecast::models::GARCH::garch_1_1();
        model.fit(&crate::anofox_series(&x)).expect("fit");
        model
    });
    let estimate = [
        model.omega(),
        model.alpha_params()[0],
        model.beta_params()[0],
    ];
    row("anofox-forecast 0.21.0", seconds, &x, estimate);
    println!(
        "| (simulated) | | | {} | {:.4}, {:.4}, {:.4} |",
        format_loglike(loglike(&x, TRUE)),
        TRUE[0],
        TRUE[1],
        TRUE[2]
    );
}

/// Prints one crate's row.
fn row(name: &str, seconds: f64, x: &[f64], estimate: [f64; 3]) {
    println!(
        "| GARCH(1,1) | {name} | {} | {} | {:.4}, {:.4}, {:.4} |",
        milliseconds(seconds),
        format_loglike(loglike(x, estimate)),
        estimate[0],
        estimate[1],
        estimate[2]
    );
}

/// A log-likelihood with three decimals, or a dash where the estimate is not a valid model.
fn format_loglike(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.3}")
    } else {
        "-".to_owned()
    }
}

/// The log-likelihood of `x` at `[omega, alpha, beta]` as `tseries::garch` defines it: the
/// variance starts at the mean square, and the values from the second on count.
fn loglike(x: &[f64], [omega, alpha, beta]: [f64; 3]) -> f64 {
    let n = x.len() as f64;
    let spec = GarchSpec {
        omega,
        alpha,
        beta,
        variance: x.iter().map(|v| v * v).sum::<f64>() / n,
    };
    let Ok(mut filter) = GarchFilter::new(&spec) else {
        return f64::NAN;
    };

    filter.update(x[0]);
    let terms: f64 = x[1..]
        .iter()
        .map(|&v| {
            let h = filter.update(v).variance;
            h.ln() + v * v / h
        })
        .sum();
    -0.5 * terms - 0.5 * (n - 1.0) * core::f64::consts::TAU.ln()
}

/// `n` values of the GARCH(1,1) model [`TRUE`] with normal innovations.
fn series(n: usize) -> Vec<f64> {
    let [omega, alpha, beta] = TRUE;
    let mut h = omega / (1.0 - alpha - beta);
    normals(n, 81)
        .into_iter()
        .map(|e| {
            let x = h.sqrt() * e;
            h = omega + alpha * x * x + beta * h;
            x
        })
        .collect()
}

//! A Prophet-style model as regression with ARIMA errors: a piecewise-linear trend,
//! weekly and yearly Fourier seasonality, and holiday dummies.
//!
//! Prophet (Taylor and Letham 2018) adds a trend with slope changes at changepoints,
//! Fourier terms per season, and holiday effects, then fits them with a sparse prior on the
//! slope changes. Here the same regressors go into `Arima::fit_exog`, so the errors are an
//! AR(1) instead of white noise. The changepoints sit on a coarse grid in the first 80% of
//! the history, as Prophet places them; with no prior to shrink them, the grid is coarse,
//! and AICc shows the trend changes pay for themselves.
//!
//! The regressor rows come from the heap-free `trend_row` and `fourier_row`, so a device
//! builds the row of each new day, and of each forecast day, the same way.
//!
//! Run with `cargo run --example prophet_recipe`.

use backshift::{Arima, Filter, fourier_columns, fourier_row, trend_row};

/// Three years of daily values.
const DAYS: usize = 1095;

/// The forecast horizon in days.
const HORIZON: usize = 30;

/// The changepoint grid: every 120 days in the first 80% of the history.
const KNOTS: [u64; 7] = [120, 240, 360, 480, 600, 720, 840];

/// The true slope (per day) and its changes at days 360 and 720.
const SLOPE: f64 = 0.5;
const CHANGES: [(u64, f64); 2] = [(360, 1.0), (720, -1.2)];

/// The Fourier terms: weekly with 3 harmonics, yearly with 4.
const SEASONS: [(f64, usize); 2] = [(7.0, 3), (365.25, 4)];

/// The holidays: the same three days every year, and their effect.
const HOLIDAYS: [usize; 3] = [0, 120, 358];
const HOLIDAY_EFFECT: f64 = -80.0;

/// The regressor columns of a day: trend, Fourier terms, holiday.
const COLUMNS: usize = KNOTS.len() + 1 + 6 + 8 + 1;

fn main() {
    assert_eq!(
        COLUMNS,
        KNOTS.len()
            + 1
            + SEASONS
                .iter()
                .map(|&(p, k)| fourier_columns(p, k))
                .sum::<usize>()
            + 1
    );
    let history: Vec<f64> = (0..DAYS).flat_map(|t| row(t, &KNOTS)).collect();
    let y = simulate(&history);

    let model = Arima::new(1, 0, 0);
    let with = model
        .fit_exog(&y, &history, COLUMNS)
        .expect("the series is long enough");
    // The same model with a single straight trend: no changepoints.
    let straight: Vec<f64> = (0..DAYS).flat_map(|t| row(t, &[])).collect();
    let without = model
        .fit_exog(&y, &straight, COLUMNS - KNOTS.len())
        .expect("the series is long enough");

    println!("Regression with AR(1) errors on {DAYS} days\n");
    println!(
        "AICc with changepoints {:.1}, without {:.1}",
        with.aicc(),
        without.aicc()
    );
    println!("slope {:.3} per day (true {SLOPE})", with.exog[0]);
    for (knot, change) in KNOTS.iter().zip(&with.exog[1..]) {
        let truth = CHANGES
            .iter()
            .find(|(day, _)| day == knot)
            .map_or(0.0, |&(_, delta)| delta);
        println!("  change at day {knot:>3}: {change:>6.3} (true {truth})");
    }
    let holiday = with.exog[COLUMNS - 1];
    println!("holiday effect {holiday:.1} (true {HOLIDAY_EFFECT})\n");

    // The forecast on the heap-free filter; the trend continues at its last slope.
    let spec = with.spec();
    let mut filter = Filter::<1, COLUMNS>::new(&spec).expect("the fitted model is stationary");
    filter.warm_up_exog(&y, &history);
    let future: Vec<f64> = (DAYS..DAYS + HORIZON)
        .flat_map(|t| row(t, &KNOTS))
        .collect();
    let (mut mean, mut lower, mut upper) = ([0.0; HORIZON], [0.0; HORIZON], [0.0; HORIZON]);
    filter.forecast_interval_exog_into(0.95, &future, &mut mean, &mut lower, &mut upper);
    for h in [0, 6, 13, 29] {
        println!(
            "day {:>4}: {:>7.1} [{:>7.1}, {:>7.1}]",
            DAYS + h + 1,
            mean[h],
            lower[h],
            upper[h]
        );
    }

    assert!(
        with.aicc() < without.aicc(),
        "the changepoints pay for themselves"
    );
    assert!(
        (with.exog[3] - 1.0).abs() < 0.3,
        "the change at day 360 is recovered"
    );
    assert!(
        (with.exog[6] + 1.2).abs() < 0.3,
        "the change at day 720 is recovered"
    );
    assert!(
        (holiday - HOLIDAY_EFFECT).abs() < 20.0,
        "the holiday effect is recovered"
    );
}

/// The regressor row of day `t` with the changepoints `knots`: trend, Fourier terms, then
/// the holiday dummy.
fn row(t: usize, knots: &[u64]) -> Vec<f64> {
    let day = u64::try_from(t).expect("small");
    let mut row = vec![0.0; knots.len() + 1];
    trend_row(day, knots, &mut row);
    for (period, k) in SEASONS {
        let mut terms = vec![0.0; fourier_columns(period, k)];
        fourier_row(period, k, day, &mut terms).expect("valid terms");
        row.extend(terms);
    }
    row.push(if HOLIDAYS.contains(&(t % 365)) {
        1.0
    } else {
        0.0
    });

    row
}

/// The series: the trend with its changes, a weekly and a yearly pattern, the holidays,
/// and AR(1) noise with `phi = 0.6`, `sigma = 5`.
fn simulate(regressors: &[f64]) -> Vec<f64> {
    let mut state = 13_u64;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // The top 53 bits as a fraction in [0, 1).
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let mut normal = move || {
        // Box–Muller.
        let (u, v) = (uniform().max(f64::MIN_POSITIVE), uniform());
        (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
    };

    let mut noise = 0.0;
    (0..DAYS)
        .map(|t| {
            let row = &regressors[t * COLUMNS..(t + 1) * COLUMNS];
            let time = t as f64;
            let trend = 200.0
                + SLOPE * time
                + CHANGES
                    .iter()
                    .map(|&(day, delta)| delta * (time - day as f64).max(0.0))
                    .sum::<f64>();
            let week = 30.0 * row[KNOTS.len() + 1] + 10.0 * row[KNOTS.len() + 2];
            let year = 50.0 * row[KNOTS.len() + 8];
            noise = 0.6 * noise + 5.0 * normal();
            trend + week + year + HOLIDAY_EFFECT * row[COLUMNS - 1] + noise
        })
        .collect()
}

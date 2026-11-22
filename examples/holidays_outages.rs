//! Daily active users with a weekly rhythm, interrupted by bank holidays and server outages.
//!
//! Two dummy regressors mark the interruptions: `holiday` is known in advance from the
//! calendar, `outage` only after the fact. Fitting SARIMA(1, 0, 0)(0, 1, 1, 7) with them
//! (regression with ARIMA errors, as R's `arima(xreg=)`) recovers how many users each
//! interruption costs, and keeps the dips out of the weekly pattern. The plain model
//! without regressors has to explain the dips as noise: its AICc is worse, and its
//! forecast does not see the bank holiday coming.
//!
//! The forecast runs on the heap-free `Filter`, as a device would: the future regressor
//! rows say "bank holiday 11 days ahead, no outage planned".
//!
//! Run with `cargo run --example holidays_outages`.

use backshift::{Arima, Filter};

/// Two years of daily values.
const DAYS: usize = 730;

/// The forecast horizon in days.
const HORIZON: usize = 28;

/// The day of the horizon (0-based) that is a bank holiday.
const UPCOMING_HOLIDAY: usize = 10;

/// Users lost on a bank holiday, and during an outage.
const HOLIDAY_EFFECT: f64 = -250.0;
const OUTAGE_EFFECT: f64 = -400.0;

/// The bank holidays in the history, as day indices: about ten a year.
const HOLIDAYS: [usize; 20] = [
    0, 88, 91, 120, 129, 139, 275, 358, 359, 364, // first year
    365, 453, 456, 485, 494, 504, 640, 723, 724, 729, // second year
];

/// The outages in the history, as day indices; the last one spans two days.
const OUTAGES: [usize; 4] = [212, 401, 602, 603];

/// The regressor columns, in the order of the coefficients `beta`.
const COLUMNS: usize = 2;

fn main() {
    let regressors = history_regressors();
    let users = simulate(&regressors);
    let model = Arima::new(1, 0, 0).seasonal(0, 1, 1, 7);

    let with = model
        .fit_exog(&users, &regressors, COLUMNS)
        .expect("the series is long enough");
    let without = model.fit(&users).expect("the series is long enough");

    println!("SARIMA(1,0,0)(0,1,1,7) on {DAYS} days of active users\n");
    println!("                  with regressors   without");
    println!(
        "AICc              {:>15.1}   {:>7.1}",
        with.aicc(),
        without.aicc()
    );
    println!(
        "sigma             {:>15.1}   {:>7.1}",
        with.sigma2.sqrt(),
        without.sigma2.sqrt()
    );
    println!(
        "holiday effect    {:>15.1}   (true {HOLIDAY_EFFECT})",
        with.exog[0]
    );
    println!(
        "outage effect     {:>15.1}   (true {OUTAGE_EFFECT})\n",
        with.exog[1]
    );

    // The filter is sized at compile time: 15 states for the model, 2 regressors.
    let spec = with.spec();
    let mut filter = Filter::<16>::new(&spec).expect("the fitted model is stationary");
    filter.warm_up_exog(&users, &regressors);

    let mut plain = Filter::<16>::new(&without.spec()).expect("the fitted model is stationary");
    plain.warm_up(&users);

    let future = future_regressors();
    let (mut mean, mut lower, mut upper) = ([0.0; HORIZON], [0.0; HORIZON], [0.0; HORIZON]);
    filter.forecast_interval_exog_into(0.95, &future, &mut mean, &mut lower, &mut upper);
    let (mut plain_mean, mut plain_lower, mut plain_upper) =
        ([0.0; HORIZON], [0.0; HORIZON], [0.0; HORIZON]);
    plain.forecast_interval_into(0.95, &mut plain_mean, &mut plain_lower, &mut plain_upper);

    println!("day   with regressors (95%)        without (95%)");
    for h in UPCOMING_HOLIDAY - 3..=UPCOMING_HOLIDAY + 3 {
        let marker = if h == UPCOMING_HOLIDAY {
            "  <- bank holiday"
        } else {
            ""
        };
        println!(
            "{:>3}   {:>6.0} [{:>6.0}, {:>6.0}]     {:>6.0} [{:>6.0}, {:>6.0}]{marker}",
            h + 1,
            mean[h],
            lower[h],
            upper[h],
            plain_mean[h],
            plain_lower[h],
            plain_upper[h],
        );
    }

    assert!(
        (with.exog[0] - HOLIDAY_EFFECT).abs() < 50.0,
        "holiday effect recovered"
    );
    assert!(
        (with.exog[1] - OUTAGE_EFFECT).abs() < 50.0,
        "outage effect recovered"
    );
    assert!(
        with.aicc() < without.aicc(),
        "the regressors pay for themselves"
    );
}

/// The regressor rows of the history, row-major: `[holiday, outage]` per day.
fn history_regressors() -> Vec<f64> {
    let mut rows = vec![0.0; DAYS * COLUMNS];
    for day in HOLIDAYS {
        rows[day * COLUMNS] = 1.0;
    }
    for day in OUTAGES {
        rows[day * COLUMNS + 1] = 1.0;
    }

    rows
}

/// The regressor rows of the forecast horizon: the calendar knows the holiday, and no
/// outage is planned.
fn future_regressors() -> [f64; HORIZON * COLUMNS] {
    let mut rows = [0.0; HORIZON * COLUMNS];
    rows[UPCOMING_HOLIDAY * COLUMNS] = 1.0;

    rows
}

/// Active users: a weekly pattern (quiet weekends) carried by seasonal ARIMA noise,
/// `(1 - 0.5 B)(1 - B^7) u_t = (1 - 0.6 B^7) e_t` with `sigma = 15`, plus the effects of
/// the holidays and outages in `regressors`.
fn simulate(regressors: &[f64]) -> Vec<f64> {
    let mut state = 7_u64;
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

    // Monday to Sunday.
    let week = [1000.0, 1040.0, 1050.0, 1030.0, 980.0, 700.0, 650.0];
    let (phi, theta, sigma) = (0.5, -0.6, 15.0);

    let mut u = Vec::with_capacity(DAYS);
    let mut e = Vec::with_capacity(DAYS);
    let mut z_previous = 0.0;
    for t in 0..DAYS {
        let shock = sigma * normal();
        e.push(shock);
        if t < 7 {
            u.push(week[t] + shock);
            continue;
        }
        // z = (1 - B^7) u follows an ARMA(1, 0)(0, 1) model.
        let z = phi * z_previous + shock + theta * e[t - 7];
        z_previous = z;
        u.push(u[t - 7] + z);
    }

    u.iter()
        .enumerate()
        .map(|(t, u)| {
            let row = &regressors[t * COLUMNS..(t + 1) * COLUMNS];
            u + row[0] * HOLIDAY_EFFECT + row[1] * OUTAGE_EFFECT
        })
        .collect()
}

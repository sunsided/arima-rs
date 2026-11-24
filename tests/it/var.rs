//! Checks the VAR and VARX fits, forecasts and lag selection against statsmodels in
//! [`crate::var_data`], and the online filter against the fits.

use crate::var_data::{ALL, VarCase};
use backshift::{OnlineVar, Var, VarFit, select_var_order};

/// Relative tolerance against statsmodels' least squares.
const TIGHT: f64 = 1e-8;

/// Asserts that every value of `actual` matches `expected` within `TIGHT`, relative.
fn assert_all_close(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= TIGHT * e.abs().max(1.0),
            "{what}: {a} vs {e}"
        );
    }
}

/// This crate's fit of `case`.
fn fit(case: &VarCase) -> VarFit {
    Var::new(case.p)
        .fit_exog(case.values, case.k, case.exog, case.columns)
        .unwrap()
}

#[test]
fn fits_match_statsmodels() {
    for case in ALL {
        let fit = fit(case);
        assert_all_close(&fit.intercept, case.intercept, "intercept");
        assert_all_close(&fit.lags, case.lags, "lags");
        assert_all_close(&fit.beta, case.beta, "beta");
        assert_all_close(&fit.sigma, case.sigma, "sigma_u");
        assert_all_close(&fit.sigma_mle, case.sigma_mle, "sigma_u_mle");
        assert_all_close(&[fit.loglike], &[case.loglike], "llf");
        let criteria = [fit.aic(), fit.bic(), fit.hqic(), fit.fpe()];
        assert_all_close(&criteria, &case.criteria, "aic, bic, hqic, fpe");
    }
}

#[test]
fn forecasts_match_statsmodels() {
    for case in ALL {
        let fit = fit(case);
        let h = case.mean.len() / case.k;
        let forecast = fit.forecast(h, case.exog_future);
        assert_all_close(&forecast.mean, case.mean, "mean");
        assert_all_close(&forecast.covariance, case.forecast_cov, "forecast_cov");
        let (mut lower, mut upper) = (vec![0.0; h * case.k], vec![0.0; h * case.k]);
        forecast.interval_into(0.95, &mut lower, &mut upper);
        assert_all_close(&lower, case.lower, "lower");
        assert_all_close(&upper, case.upper, "upper");
    }
}

#[test]
fn lag_selection_matches_statsmodels() {
    for case in ALL {
        let orders = select_var_order(case.values, case.k, case.exog, case.columns, 6).unwrap();
        let picked = [orders.aic, orders.bic, orders.hqic, orders.fpe];
        assert_eq!(picked, case.selected, "k = {}, p = {}", case.k, case.p);
    }
}

#[test]
fn online_filter_forecasts_like_the_fit() {
    let case = ALL.iter().find(|c| c.columns == 0 && c.k == 2).unwrap();
    let fit = fit(case);
    let mut filter = OnlineVar::<2, 8>::new(&fit.spec()).unwrap();
    filter.warm_up(case.values);
    let h = case.mean.len() / 2;
    let (mut mean, mut variance) = (vec![0.0; 2 * h], vec![0.0; 2 * h]);
    filter.forecast_into(&mut mean, &mut variance);
    assert_all_close(&mean, case.mean, "online mean");
    let diagonal: Vec<f64> = (0..h)
        .flat_map(|step| (0..2).map(move |i| case.forecast_cov[step * 4 + i * 2 + i]))
        .collect();
    assert_all_close(&variance, &diagonal, "online variance");
}

#[test]
fn learning_without_forgetting_is_least_squares_on_all_values() {
    // Fit on all but the last 30 rows, learn those online, and compare with the fit on all.
    let case = ALL.iter().find(|c| c.columns == 0 && c.k == 3).unwrap();
    let (k, p) = (case.k, case.p);
    let cut = case.values.len() - 30 * k;
    let early = Var::new(p).fit(&case.values[..cut], k).unwrap();
    let mut filter = OnlineVar::<3, 4>::new(&early.spec())
        .unwrap()
        .with_learning(&early.gram_inverse, 1.0)
        .unwrap();
    filter.warm_up(&case.values[cut - p * k..cut]);
    filter.warm_up(&case.values[cut..]);

    let full = fit(case);
    let (mut intercept, mut lags) = (vec![0.0; k], vec![0.0; p * k * k]);
    filter.coefficients_into(&mut intercept, &mut lags);
    assert_all_close(&intercept, &full.intercept, "learned intercept");
    assert_all_close(&lags, &full.lags, "learned lags");
}

#[test]
fn mahalanobis_distance_scales_with_sigma() {
    // White noise in two series with unit variances and correlation 0.5: an error of (1, 1)
    // sits along the correlation, (1, -1) against it.
    let spec = backshift::VarSpec {
        intercept: &[0.0, 0.0],
        lags: &[],
        sigma: &[1.0, 0.5, 0.5, 1.0],
    };
    let mut filter = OnlineVar::<2, 1>::new(&spec).unwrap();
    let mut innovation = [0.0; 2];
    let along = filter.update(&[1.0, 1.0], &mut innovation).unwrap();
    let against = filter.update(&[1.0, -1.0], &mut innovation).unwrap();
    assert!((along - 4.0 / 3.0).abs() < 1e-12 && (against - 4.0).abs() < 1e-12);
}

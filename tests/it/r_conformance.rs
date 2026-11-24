//! Checks the crate against R 4.4 and the `forecast` package ([`crate::r_data`]).
//!
//! R's own STL (degree-0 seasonal smoothing, a different low-pass window) differs from
//! statsmodels' by design, so STL and MSTL are checked against statsmodels only.

use crate::r_data::{
    ACF, ARIMA, AUTO, ArimaCase, DIAGNOSTIC, FIXED_AR, FIXED_AR_AIC, FOURIER_HORIZON,
    FOURIER_IN_SAMPLE, FOURIER_MONTHLY, LJUNG_BOX, PACF,
};
use backshift::{
    Arima, ArimaSpec, AutoArima, Estimator, Filter, Fixed, Seasonal, TimeSeries, acorr, fourier,
    ljung_box, ndiffs, nsdiffs, pacf,
};

/// How far below R's maximised log-likelihood ours may end up without differencing.
const LOGLIKE_SLACK: f64 = 1e-6;

/// The same with differencing. R's `arima` approximates the diffuse prior of the
/// differenced states with a variance of `kappa = 1e6`, which moves its log-likelihood by
/// about `1e-5` for `d = 2`; the exact likelihood here differs by that much at R's own
/// coefficients.
const DIFFUSE_SLACK: f64 = 5e-5;

/// Tolerance on fitted coefficients: two optimisers stop at slightly different points.
const PARAMETER_TOLERANCE: f64 = 2e-3;

/// Tolerance on statistics both sides compute in closed form.
const TOLERANCE: f64 = 1e-9;

/// Asserts `|actual - expected| <= tolerance * max(1, |expected|)`.
fn assert_close(what: &str, actual: f64, expected: f64, tolerance: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance * scale,
        "{what}: got {actual}, R {expected}"
    );
}

/// The model of a case.
fn model(case: &ArimaCase) -> Arima {
    let [p, d, q] = case.order;
    let model = Arima::new(p, d, q).mean(case.include_mean);
    let [sp, sd, sq] = case.seasonal;
    if case.period > 0 {
        model
            .seasonal(sp, sd, sq, case.period)
            .mean(case.include_mean)
    } else {
        model
    }
}

/// R's fitted coefficients of a case as a spec.
fn r_spec(case: &ArimaCase) -> ArimaSpec<'static> {
    ArimaSpec {
        ar: case.ar,
        d: case.order[1],
        ma: case.ma,
        seasonal: Seasonal {
            ar: case.seasonal_ar,
            d: case.seasonal[1],
            ma: case.seasonal_ma,
            period: case.period,
        },
        mean: case.mean,
        sigma2: case.sigma2,
        ..ArimaSpec::default()
    }
}

#[test]
fn arima_reaches_the_r_maximum() {
    for case in ARIMA {
        let fitted = model(case).fit(case.values).unwrap();
        let what = format!("{:?}{:?}", case.order, case.seasonal);
        let differenced = case.order[1] + case.seasonal[1] > 0;
        let slack = if differenced {
            DIFFUSE_SLACK
        } else {
            LOGLIKE_SLACK
        };
        assert!(
            fitted.loglike >= case.loglike - slack,
            "{what}: llf {} below R {}",
            fitted.loglike,
            case.loglike
        );
        let pairs = fitted
            .ar
            .iter()
            .zip(case.ar)
            .chain(fitted.ma.iter().zip(case.ma))
            .chain(fitted.seasonal_ar.iter().zip(case.seasonal_ar))
            .chain(fitted.seasonal_ma.iter().zip(case.seasonal_ma))
            .chain([(&fitted.mean, &case.mean)]);
        for (actual, expected) in pairs {
            assert_close(&what, *actual, *expected, PARAMETER_TOLERANCE);
        }
        assert_close(&what, fitted.sigma2, case.sigma2, PARAMETER_TOLERANCE);
    }
}

#[test]
fn forecasts_at_r_coefficients_match_predict() {
    for case in ARIMA.iter().chain([&&FIXED_AR]) {
        let mut filter = Filter::<32>::new(&r_spec(case)).unwrap();
        filter.warm_up(case.values);
        let (mut mean, mut variance) = ([0.0; 5], [0.0; 5]);
        filter.forecast_into(&mut mean, &mut variance);
        let what = format!("{:?}{:?}", case.order, case.seasonal);
        for h in 0..5 {
            assert_close(&what, mean[h], case.forecast_mean[h], 1e-6);
            assert_close(&what, variance[h].sqrt(), case.forecast_se[h], 1e-6);
        }
    }
}

#[test]
fn fixed_parts_match_r_arima_fixed() {
    let case = &FIXED_AR;
    let start = ArimaSpec {
        ar: &[0.5],
        ma: &[0.0],
        ..ArimaSpec::default()
    };
    let fixed = Fixed {
        ar: true,
        ..Fixed::NONE
    };
    let fitted = model(case).fit_fixed(case.values, &start, fixed).unwrap();
    assert!(fitted.loglike >= case.loglike - LOGLIKE_SLACK);
    assert_close("ma", fitted.ma[0], case.ma[0], PARAMETER_TOLERANCE);
    assert_close("mean", fitted.mean, case.mean, PARAMETER_TOLERANCE);
    // R's AIC counts only the estimated parameters, as `Fitted::parameters` does.
    assert_close("aic", fitted.aic(), FIXED_AR_AIC, 1e-6);
}

#[test]
fn diagnostics_match_r() {
    let series = TimeSeries::try_new(DIAGNOSTIC, Estimator::Biased).unwrap();
    for (lag, &expected) in ACF.iter().enumerate() {
        assert_close("acf", acorr(&series, lag + 1).unwrap(), expected, TOLERANCE);
    }

    let partial = pacf(&series, PACF.len()).unwrap();
    for (lag, &expected) in PACF.iter().enumerate() {
        assert_close("pacf", partial[lag + 1], expected, TOLERANCE);
    }

    let test = ljung_box(DIAGNOSTIC, 10, 2).unwrap();
    assert_close("Q", test.statistic, LJUNG_BOX[0], TOLERANCE);
    assert_close("p", test.p_value, LJUNG_BOX[1], TOLERANCE);
}

#[test]
fn fourier_matches_r_forecast_fourier() {
    let terms = [(7.0, 2), (30.5, 3)];
    let in_sample = fourier(&terms, 1, 40).unwrap();
    let horizon = fourier(&terms, 41, 5).unwrap();
    let monthly = fourier(&[(12.0, 6)], 1, 24).unwrap();
    for (actual, expected) in [
        (in_sample, FOURIER_IN_SAMPLE),
        (horizon, FOURIER_HORIZON),
        (monthly, FOURIER_MONTHLY),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "{a} vs {e}");
        }
    }
}

#[test]
fn differencing_tests_match_r() {
    for case in AUTO {
        assert_eq!(ndiffs(case.values, 0.05, 2), case.ndiffs, "ndiffs");
        if case.period > 1 {
            assert_eq!(
                nsdiffs(case.values, case.period, 1),
                case.nsdiffs,
                "nsdiffs"
            );
        }
    }
}

#[test]
fn auto_arima_picks_r_orders() {
    for case in AUTO {
        let search = if case.period > 1 {
            AutoArima::new().seasonal(case.period)
        } else {
            AutoArima::new()
        };
        let best = search.fit(case.values).unwrap().fitted;
        let orders = [
            best.ar.len(),
            best.d,
            best.ma.len(),
            best.seasonal_ar.len(),
            best.seasonal_d,
            best.seasonal_ma.len(),
        ];
        assert_eq!(orders, case.orders, "period {}", case.period);
        assert_eq!(
            best.with_mean, case.constant,
            "constant, period {}",
            case.period
        );
    }
}

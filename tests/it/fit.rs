//! Checks Hannan–Rissanen and maximum-likelihood fitting against statsmodels.

use crate::arma_data::{ALL, ArmaCase};
use backshift::{Arima, FitError, Method, difference, hannan_rissanen};

/// Absolute tolerance on fitted coefficients and means, relative tolerance on `sigma2`.
///
/// Two optimisers stop at slightly different points of a likelihood this flat (60
/// values): parameters agree to about 1e-4 while `llf` agrees to about 1e-8. So
/// parameters are compared loosely and the likelihood tightly.
const PARAMETER_TOLERANCE: f64 = 5e-4;

/// How far below statsmodels' maximised log-likelihood ours may end up.
const LOGLIKE_SLACK: f64 = 1e-6;

/// The model a case was fitted with in statsmodels.
fn model(case: &ArmaCase) -> Arima {
    let model = Arima::new(case.ar.len(), case.d, case.ma.len());
    let model = if case.period > 0 {
        model.seasonal(
            case.seasonal_ar.len(),
            case.seasonal_d,
            case.seasonal_ma.len(),
            case.period,
        )
    } else {
        model
    };

    model.mean(case.fit_with_mean)
}

#[test]
fn hannan_rissanen_matches_statsmodels() {
    for case in ALL {
        let Some((ar, ma)) = case.hannan_rissanen else {
            continue;
        };

        let w = difference(case.values, case.d);
        let estimate = hannan_rissanen(&w, ar.len(), ma.len()).unwrap();
        for (actual, expected) in estimate
            .ar
            .iter()
            .chain(&estimate.ma)
            .zip(ar.iter().chain(ma))
        {
            assert!(
                (actual - expected).abs() <= 1e-9,
                "got {actual}, statsmodels {expected}"
            );
        }
    }
}

#[test]
fn fit_reaches_the_statsmodels_maximum() {
    for (index, case) in ALL.iter().enumerate() {
        let fitted = model(case)
            .fit_exog(case.values, case.regressors, case.exog.len())
            .unwrap();
        let context = format!("case {index}: {fitted:?}");

        assert!(fitted.converged, "{context}");
        assert!(
            fitted.loglike >= case.fit_loglike - LOGLIKE_SLACK,
            "{context}: llf {} below statsmodels {}",
            fitted.loglike,
            case.fit_loglike
        );

        let pairs = fitted
            .ar
            .iter()
            .zip(case.fit_ar)
            .chain(fitted.ma.iter().zip(case.fit_ma))
            .chain(fitted.seasonal_ar.iter().zip(case.fit_seasonal_ar))
            .chain(fitted.seasonal_ma.iter().zip(case.fit_seasonal_ma))
            .chain(fitted.exog.iter().zip(case.fit_exog));
        for (actual, expected) in pairs {
            assert!(
                (actual - expected).abs() <= PARAMETER_TOLERANCE,
                "{context}: {actual} vs {expected}"
            );
        }

        assert!(
            (fitted.mean - case.fit_mean).abs() <= PARAMETER_TOLERANCE,
            "{context}: mean"
        );
        let sigma2_error = (fitted.sigma2 - case.fit_sigma2).abs() / case.fit_sigma2;
        assert!(sigma2_error <= PARAMETER_TOLERANCE, "{context}: sigma2");
    }
}

#[test]
fn information_criteria_match_statsmodels() {
    for (index, case) in ALL.iter().enumerate() {
        let fitted = model(case)
            .fit_exog(case.values, case.regressors, case.exog.len())
            .unwrap();

        // Our llf sits up to about 1e-7 above statsmodels', so the criteria may differ by twice that.
        let close = |actual: f64, expected: f64| (actual - expected).abs() <= 1e-6;
        assert!(
            close(fitted.aic(), case.fit_aic),
            "case {index}: aic {} vs {}",
            fitted.aic(),
            case.fit_aic
        );

        // statsmodels counts missing values in n; this crate, like R, does not.
        if case.values.iter().any(|v| v.is_nan()) {
            assert!(
                fitted.bic() < case.fit_bic,
                "case {index}: fewer observations, smaller penalty"
            );
            continue;
        }

        assert!(
            close(fitted.aicc(), case.fit_aicc),
            "case {index}: aicc {} vs {}",
            fitted.aicc(),
            case.fit_aicc
        );
        assert!(
            close(fitted.bic(), case.fit_bic),
            "case {index}: bic {} vs {}",
            fitted.bic(),
            case.fit_bic
        );
    }
}

#[test]
fn fit_from_a_previous_fit_reaches_the_same_maximum_faster() {
    for (index, case) in ALL.iter().enumerate() {
        let model = model(case);
        let columns = case.exog.len();
        let first = model
            .fit_exog(case.values, case.regressors, columns)
            .unwrap();
        let again = model
            .fit_exog_from(case.values, case.regressors, columns, &first.spec())
            .unwrap();
        let context = format!("case {index}");

        assert!(again.converged, "{context}");
        assert!(again.iterations <= first.iterations, "{context}");
        assert!(
            again.loglike >= case.fit_loglike - LOGLIKE_SLACK,
            "{context}: llf {} below statsmodels {}",
            again.loglike,
            case.fit_loglike
        );
    }
}

#[test]
fn fit_from_rejects_other_orders() {
    let case = ALL[0];
    let fitted = model(case).fit(case.values).unwrap();
    let other = Arima::new(2, 0, 0).fit_from(case.values, &fitted.spec());
    assert_eq!(other, Err(FitError::StartMismatch));

    let with_regressor = model(case).fit_exog_from(case.values, case.values, 1, &fitted.spec());
    assert_eq!(with_regressor, Err(FitError::StartMismatch));
}

#[test]
fn css_fits_never_beat_the_exact_optimum() {
    for case in ALL {
        let model = model(case);
        let (Ok(exact), Ok(css)) = (
            model.fit(case.values),
            model.method(Method::Css).fit(case.values),
        ) else {
            continue;
        };
        // Both report the exact log-likelihood, so the exact fit's is never lower.
        assert!(
            css.loglike <= exact.loglike + LOGLIKE_SLACK,
            "{css:?} vs {exact:?}"
        );
    }
}

#[test]
fn css_falls_back_to_exact_with_missing_values() {
    let mut y: Vec<f64> = ALL[0].values.to_vec();
    y[10] = f64::NAN;
    let model = model(ALL[0]);
    assert_eq!(model.method(Method::Css).fit(&y), model.fit(&y));
}

#[test]
fn css_fits_accept_a_start() {
    let model = model(ALL[0]).method(Method::Css);
    let first = model.fit(ALL[0].values).unwrap();
    assert!(model.fit_from(ALL[0].values, &first.spec()).is_ok());
}

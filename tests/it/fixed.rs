//! Checks fits with fixed parts against statsmodels' `fix_params`.

use crate::fixed_data::{ALL, FixedCase};
use backshift::{Arima, ArimaSpec, Fixed, Seasonal};

/// How far below statsmodels' maximised log-likelihood ours may end up, as in `fit`.
const LOGLIKE_SLACK: f64 = 1e-6;

/// Tolerance on the estimated coefficients: two optimisers stop at slightly different
/// points of a flat likelihood.
const PARAMETER_TOLERANCE: f64 = 2e-3;

/// The model of a case.
fn model(case: &FixedCase) -> Arima {
    let model = Arima::new(case.ar.len(), 0, case.ma.len()).mean(case.with_mean);
    if case.period > 0 {
        model.seasonal(0, 0, case.seasonal_ma.len(), case.period)
    } else {
        model
    }
}

#[test]
fn fixed_parts_match_statsmodels_fix_params() {
    for case in ALL {
        // The start holds statsmodels' values; only the fixed parts must match them.
        let start = ArimaSpec {
            ar: case.ar,
            ma: case.ma,
            seasonal: Seasonal {
                ma: case.seasonal_ma,
                period: case.period,
                ..Seasonal::NONE
            },
            mean: case.mean,
            ..ArimaSpec::default()
        };
        let fixed = Fixed {
            ar: case.fixed_ar,
            seasonal_ma: case.fixed_seasonal_ma,
            mean: case.fixed_mean,
            ..Fixed::NONE
        };
        let fitted = model(case).fit_fixed(case.values, &start, fixed).unwrap();

        assert_eq!(fitted.fixed, 1);
        assert!(
            fitted.loglike >= case.loglike - LOGLIKE_SLACK,
            "llf {} vs {}",
            fitted.loglike,
            case.loglike
        );
        let pairs = fitted
            .ar
            .iter()
            .zip(case.ar)
            .chain(fitted.ma.iter().zip(case.ma))
            .chain(fitted.seasonal_ma.iter().zip(case.seasonal_ma))
            .chain([(&fitted.mean, &case.mean)]);
        for (actual, expected) in pairs {
            assert!(
                (actual - expected).abs() < PARAMETER_TOLERANCE,
                "{actual} vs {expected}"
            );
        }

        let sigma2_error = (fitted.sigma2 - case.sigma2).abs() / case.sigma2;
        assert!(
            sigma2_error < PARAMETER_TOLERANCE,
            "sigma2 {}",
            fitted.sigma2
        );

        // Fixed parts are exact, not merely close.
        if case.fixed_ar {
            assert_eq!(fitted.ar, case.ar);
        }
        if case.fixed_mean {
            assert_eq!(fitted.mean, case.mean);
        }
    }
}

#[test]
fn fixed_parts_are_not_counted_in_the_criteria() {
    let case = ALL[0];
    let start = ArimaSpec {
        ma: case.ma,
        seasonal: Seasonal {
            ma: case.seasonal_ma,
            period: case.period,
            ..Seasonal::NONE
        },
        ..ArimaSpec::default()
    };
    let free = model(case).fit_from(case.values, &start).unwrap();
    let fixed = Fixed {
        seasonal_ma: true,
        ..Fixed::NONE
    };
    let held = model(case).fit_fixed(case.values, &start, fixed).unwrap();
    assert_eq!(free.parameters(), held.parameters() + 1);
    assert!(held.loglike <= free.loglike + 1e-9);
}

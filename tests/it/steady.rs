//! Checks the filter's steady-state mode against statsmodels with its default tolerance.

use crate::steady_data::{ALL, SteadyCase};
use backshift::{ArimaSpec, Filter, Innovation, Seasonal};

/// Relative tolerance against statsmodels' default (steady-state) filter.
///
/// The full recursion differs from those references by about 1e-10, so this tolerance
/// tells the two modes apart.
const TOLERANCE: f64 = 1e-13;

/// The state capacity of the filters: larger than every case needs.
const CAPACITY: usize = 8;

/// Whether `actual` matches `expected` within [`TOLERANCE`], relative to `max(1, |expected|)`.
/// Two `NaN`s match.
fn close(actual: f64, expected: f64) -> bool {
    if expected.is_nan() {
        return actual.is_nan();
    }

    (actual - expected).abs() <= TOLERANCE * expected.abs().max(1.0)
}

/// A filter for the model of a case, with the steady-state mode if `steady`.
fn filter(case: &SteadyCase, steady: bool) -> Filter<CAPACITY> {
    let spec = ArimaSpec {
        ar: case.ar,
        d: case.d,
        ma: case.ma,
        seasonal: Seasonal {
            ma: case.seasonal_ma,
            period: case.period,
            ..Seasonal::NONE
        },
        mean: case.mean,
        sigma2: case.sigma2,
        ..ArimaSpec::default()
    };
    let filter = Filter::new(&spec).expect("reference models are stationary");
    if steady {
        filter.with_steady_state()
    } else {
        filter
    }
}

/// The innovations of streaming every value of a case through a filter.
fn innovations(filter: &mut Filter<CAPACITY>, case: &SteadyCase) -> Vec<Innovation> {
    case.values
        .iter()
        .filter_map(|&y| filter.update(y))
        .collect()
}

/// Whether every innovation matches statsmodels' errors and error variances.
fn matches(actual: &[Innovation], case: &SteadyCase) -> bool {
    actual.len() == case.innovation_values.len()
        && actual.iter().enumerate().all(|(t, innovation)| {
            close(innovation.value, case.innovation_values[t])
                && close(innovation.variance, case.innovation_variances[t])
        })
}

#[test]
fn steady_state_innovations_match_statsmodels_default() {
    for case in ALL {
        let actual = innovations(&mut filter(case, true), case);
        assert!(matches(&actual, case), "period {}", case.period_converged);
    }
}

#[test]
fn full_recursion_differs_from_statsmodels_default() {
    // Otherwise the test above would not show that the mode takes the shortcut.
    let differs = ALL
        .iter()
        .filter(|case| !matches(&innovations(&mut filter(case, false), case), case))
        .count();
    assert_eq!(differs, ALL.len());
}

#[test]
fn steady_state_forecasts_match_statsmodels_default() {
    for case in ALL.iter().filter(|case| !case.forecast_mean.is_empty()) {
        let mut filter = filter(case, true);
        filter.warm_up(case.values);
        let (mut mean, mut variance) = ([0.0; 5], [0.0; 5]);
        filter.forecast_into(&mut mean, &mut variance);
        for h in 0..mean.len() {
            assert!(close(mean[h], case.forecast_mean[h]), "mean[{h}]");
            assert!(
                close(variance[h], case.forecast_variance[h]),
                "variance[{h}]"
            );
        }
    }
}

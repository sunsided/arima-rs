//! Checks the seasonal period candidates against coreforecast's `find_season_length`.

use crate::period_data::ALL;
use backshift::find_period;

#[test]
fn first_candidate_matches_coreforecast() {
    for case in ALL {
        let candidates = find_period(case.values, case.max_period);
        let first = candidates.first().map(|c| c.period);
        if case.season_length > 0 {
            assert_eq!(first, Some(case.season_length), "{candidates:?}");
        } else {
            // coreforecast rejects its largest lag; it must not be a candidate here either.
            assert!(
                candidates.iter().all(|c| c.period != case.largest_lag),
                "lag {}: {candidates:?}",
                case.largest_lag
            );
        }
    }
}

#[test]
fn candidates_are_ranked_and_strong() {
    for case in ALL {
        let candidates = find_period(case.values, case.max_period);
        assert!(
            candidates
                .windows(2)
                .all(|w| w[0].autocovariance >= w[1].autocovariance)
        );
        assert!(candidates.iter().all(|c| c.strength > 0.64));
    }
}

#[test]
fn no_candidates_without_finite_data() {
    assert!(find_period(&[1.0, 2.0, f64::NAN, 4.0, 5.0], 2).is_empty());
    assert!(find_period(&[1.0, 2.0, f64::INFINITY, 4.0, 5.0], 2).is_empty());
    assert!(find_period(&[1.0, 2.0, 3.0], 2).is_empty());
}

#[test]
fn short_series_give_at_most_the_lags_they_have() {
    // Four values leave one difference lag to check; five leave two. Neither may panic,
    // and an empty result is a valid answer.
    for n in [4, 5] {
        let y: Vec<f64> = (0..n).map(|t| if t % 2 == 0 { 0.0 } else { 1.0 }).collect();
        assert!(find_period(&y, n).len() <= n - 3);
    }
}

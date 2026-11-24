//! Checks GARCH(1,1) against R's `tseries::garch` in [`crate::r_garch_data`].

use crate::r_garch_data::{ALL, GarchCase};
use backshift::{FitError, Garch, GarchFilter, GarchSpec};

/// The mean square of `x`, where tseries starts the variance.
fn mean_square(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64
}

/// A filter at the first value of `case` with the coefficients `[omega, alpha, beta]`.
fn filter_at_start(case: &GarchCase, coefficients: [f64; 3]) -> GarchFilter {
    let [omega, alpha, beta] = coefficients;
    GarchFilter::new(&GarchSpec {
        omega,
        alpha,
        beta,
        variance: mean_square(case.values),
    })
    .unwrap()
}

/// The log-likelihood of `case` at `coefficients`, from the filter's variances, as tseries
/// defines it: the values from the second on, with the normal constant.
fn loglike(case: &GarchCase, coefficients: [f64; 3]) -> f64 {
    let mut filter = filter_at_start(case, coefficients);
    filter.update(case.values[0]);
    let terms: f64 = case.values[1..]
        .iter()
        .map(|&x| {
            let h = filter.update(x).variance;
            h.ln() + x * x / h
        })
        .sum();
    -0.5 * terms - 0.5 * (case.values.len() - 1) as f64 * core::f64::consts::TAU.ln()
}

#[test]
fn filter_reproduces_tseries_volatility_and_likelihood() {
    for case in ALL {
        // tseries' fitted values start at the long-run variance, its likelihood (below) at
        // the mean square.
        let [omega, alpha, beta] = case.tseries;
        let mut filter = GarchFilter::new(&GarchSpec {
            omega,
            alpha,
            beta,
            variance: omega / (1.0 - alpha - beta),
        })
        .unwrap();
        filter.update(case.values[0]);
        for (t, (&x, &sigma)) in case.values[1..].iter().zip(case.sigma).enumerate() {
            let ours = filter.update(x).variance.sqrt();
            assert!(
                (ours - sigma).abs() < 1e-12 * sigma,
                "value {}: {ours} vs R {sigma}",
                t + 1
            );
        }

        let ours = loglike(case, case.tseries);
        assert!(
            (ours - case.tseries_loglike).abs() < 1e-9 * case.tseries_loglike.abs(),
            "{ours} vs R {}",
            case.tseries_loglike
        );
    }
}

#[test]
fn fit_reaches_the_optimum() {
    for case in ALL {
        let fit = Garch::new().fit(case.values).unwrap();
        assert!(
            fit.loglike >= case.optimum_loglike - 1e-6,
            "{} vs optimum {}",
            fit.loglike,
            case.optimum_loglike
        );
        assert!(fit.loglike >= case.tseries_loglike - 1e-9);
        assert!((fit.loglike - loglike(case, [fit.omega, fit.alpha, fit.beta])).abs() < 1e-9);

        let ours = [fit.omega, fit.alpha, fit.beta];
        for (ours, best) in ours.iter().zip(case.optimum) {
            assert!(
                (ours - best).abs() < 1e-3 * best.max(0.1),
                "{ours:?} vs optimum {:?}",
                case.optimum
            );
        }
    }
}

#[test]
fn filter_continues_from_the_fit() {
    let case = ALL[0];
    let fit = Garch::new().fit(case.values).unwrap();
    let mut filter = filter_at_start(case, [fit.omega, fit.alpha, fit.beta]);
    for (&x, &h) in case.values.iter().zip(&fit.variances) {
        assert_eq!(filter.update(x).variance, h);
    }
    assert_eq!(filter.variance(), fit.variance);
    assert_eq!(GarchFilter::new(&fit.spec()).unwrap(), filter);
}

#[test]
fn fit_is_scale_free() {
    let case = ALL[2];
    let fit = Garch::new().fit(case.values).unwrap();
    let scaled: Vec<f64> = case.values.iter().map(|x| 1000.0 * x).collect();
    let big = Garch::new().fit(&scaled).unwrap();
    assert!((big.omega / 1e6 - fit.omega).abs() < 1e-6 * fit.omega);
    assert!((big.alpha - fit.alpha).abs() < 1e-6);
    assert!((big.beta - fit.beta).abs() < 1e-6);
}

#[test]
fn white_noise_has_little_persistence_and_bad_input_fails() {
    let mut state = 3_u64;
    let noise: Vec<f64> = (0..500)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        })
        .collect();
    let fit = Garch::new().fit(&noise).unwrap();
    assert!(fit.loglike.is_finite() && fit.alpha < 0.1);

    assert_eq!(Garch::new().fit(&[0.0; 20]), Err(FitError::Degenerate));
    assert_eq!(
        Garch::new().fit(&[1.0; 5]),
        Err(FitError::TooShort {
            needed: 10,
            present: 5
        })
    );
    assert_eq!(
        Garch::new().fit(&[1.0, f64::NAN, 2.0]),
        Err(FitError::Unsupported)
    );
}

//! Compares PMM2 with exact maximum likelihood on simulated series with skewed innovations.
//!
//! The paper (arXiv:2511.07059) compares PMM2 with conditional sum of squares; these
//! tests compare it with this crate's exact maximum likelihood, on the paper's models.

use backshift::{Arima, FitError};

/// Simulated series per model.
const REPLICATIONS: usize = 200;

/// The length of each series.
const LENGTH: usize = 500;

/// A small xorshift generator, so the tests need no extra dependency.
struct Rng(u64);

impl Rng {
    /// A uniform value in `(0, 1)`.
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    /// A standard normal value (Box–Muller).
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// A Gamma(2, 1) value centred and scaled to unit variance: skewness `sqrt(2)`.
    fn skewed(&mut self) -> f64 {
        let gamma = -(self.uniform().ln() + self.uniform().ln());
        (gamma - 2.0) / 2.0_f64.sqrt()
    }
}

/// An ARIMA(p, d, q) series with the given innovations.
fn simulate(phi: &[f64], d: usize, theta: &[f64], rng: &mut Rng, skewed: bool) -> Vec<f64> {
    let burn = 200;
    let (mut w, mut e) = (vec![0.0; LENGTH + burn], vec![0.0; LENGTH + burn]);
    for t in 0..w.len() {
        e[t] = if skewed { rng.skewed() } else { rng.normal() };
        w[t] = e[t];
        for (i, p) in phi.iter().enumerate() {
            w[t] += p * t.checked_sub(i + 1).map_or(0.0, |s| w[s]);
        }
        for (j, q) in theta.iter().enumerate() {
            w[t] += q * t.checked_sub(j + 1).map_or(0.0, |s| e[s]);
        }
    }

    let mut y = w[burn..].to_vec();
    for _ in 0..d {
        let mut level = 0.0;
        for value in &mut y {
            level += *value;
            *value = level;
        }
    }

    y
}

/// The mean squared errors of the first coefficient by maximum likelihood and by PMM2,
/// the mean theoretical efficiency, and the share of series PMM2 rejected.
///
/// PMM2 can solve its equations outside the stationary and invertible region, for example
/// for an ARMA(1, 1) whose AR and MA roots nearly cancel; it then returns an error and the
/// caller keeps the maximum-likelihood fit. Those series are left out of both errors.
fn compare(phi: &[f64], d: usize, theta: &[f64], skewed: bool, seed: u64) -> (f64, f64, f64, f64) {
    let model = Arima::new(phi.len(), d, theta.len()).mean(false);
    let truth = phi.first().or(theta.first()).copied().unwrap();
    let first = |f: &backshift::Fitted| *f.ar.first().or(f.ma.first()).unwrap();
    let mut rng = Rng(seed);
    let (mut mle, mut pmm2, mut efficiency, mut used) = (0.0, 0.0, 0.0, 0_usize);
    for _ in 0..REPLICATIONS {
        let y = simulate(phi, d, theta, &mut rng, skewed);
        match model.fit_pmm2(&y) {
            Ok(fit) => {
                mle += (first(&fit.start) - truth).powi(2);
                pmm2 += (first(&fit.fitted) - truth).powi(2);
                efficiency += fit.efficiency();
                used += 1;
            }
            Err(error) => assert_eq!(error, FitError::Degenerate),
        }
    }

    let n = used as f64;
    let rejected = 1.0 - n / REPLICATIONS as f64;
    (mle / n, pmm2 / n, efficiency / n, rejected)
}

#[test]
fn pmm2_beats_maximum_likelihood_with_skewed_innovations() {
    // ARIMA(1, 1, 0) and ARMA(1, 1) from the paper, Gamma(2, 1) innovations (theory: 5/3).
    for (phi, d, theta, seed) in [(&[0.7][..], 1, &[][..], 1), (&[0.6][..], 0, &[-0.4][..], 2)] {
        let (mle, pmm2, efficiency, rejected) = compare(phi, d, theta, true, seed);
        println!(
            "phi {phi:?} theta {theta:?}: MSE ML {mle:.6}, PMM2 {pmm2:.6}, ratio {:.2}, \
             efficiency {efficiency:.2}, rejected {rejected:.3}",
            mle / pmm2
        );
        assert!(rejected <= 0.02, "rejected {rejected}");
        assert!(mle / pmm2 > 1.3, "ratio {}", mle / pmm2);
        assert!(
            (efficiency - 5.0 / 3.0).abs() < 0.25,
            "efficiency {efficiency}"
        );
    }
}

#[test]
fn pmm2_matches_maximum_likelihood_with_gaussian_innovations() {
    let (mle, pmm2, efficiency, rejected) = compare(&[0.7], 1, &[], false, 3);
    assert_eq!(rejected, 0.0);
    println!("Gaussian: MSE ML {mle:.6}, PMM2 {pmm2:.6}, efficiency {efficiency:.2}");
    assert!((mle / pmm2 - 1.0).abs() < 0.15, "ratio {}", mle / pmm2);
    assert!(efficiency < 1.1);
}

#[test]
fn pmm2_estimates_a_drift_on_differenced_series() {
    // ARIMA(1, 1, 0) with drift 0.3 and skewed innovations: the design, the innovations
    // and the intercept all live on the differenced series.
    let mut rng = Rng(4);
    let mut w = simulate(&[0.7], 0, &[], &mut rng, true);
    for value in &mut w {
        *value += 0.3;
    }

    let mut level = 0.0;
    let y: Vec<f64> = w
        .iter()
        .map(|v| {
            level += v;
            level
        })
        .collect();
    let fit = Arima::new(1, 1, 0).mean(true).fit_pmm2(&y).unwrap();
    // The drift estimate has a standard error of about 0.15 here, so compare it with the
    // sample mean of the differences rather than with 0.3.
    let sample = (y[y.len() - 1] - y[0]) / (y.len() - 1) as f64;
    assert!(fit.fitted.converged);
    assert!(
        (fit.fitted.mean - sample).abs() < 0.05,
        "{} vs {sample}",
        fit.fitted.mean
    );
    assert!((fit.fitted.mean - fit.start.mean).abs() < 0.05);
    assert!((fit.fitted.ar[0] - 0.7).abs() < 0.08, "{:?}", fit.fitted.ar);
}

#[test]
fn pmm2_fits_an_arma_with_a_mean() {
    // d = 0 with a mean and an MA term: the intercept column and the lagged innovations.
    let mut rng = Rng(5);
    let y: Vec<f64> = simulate(&[0.6], 0, &[-0.4], &mut rng, true)
        .iter()
        .map(|v| v + 2.0)
        .collect();
    let fit = Arima::new(1, 0, 1).fit_pmm2(&y).unwrap();
    assert!(fit.fitted.converged);
    assert!(
        (fit.fitted.mean - fit.start.mean).abs() < 0.05,
        "{}",
        fit.fitted.mean
    );
    assert!((fit.fitted.ar[0] - 0.6).abs() < 0.2, "{:?}", fit.fitted.ar);
    assert!((fit.fitted.ma[0] + 0.4).abs() < 0.2, "{:?}", fit.fitted.ma);
}

#[test]
fn pmm2_rejects_seasonal_models() {
    let y: Vec<f64> = (0..60)
        .map(|t| f64::from(t % 4) + 0.1 * f64::from(t))
        .collect();
    let seasonal = Arima::new(1, 0, 0).seasonal(1, 0, 0, 4);
    assert_eq!(seasonal.fit_pmm2(&y), Err(FitError::Unsupported));
}

//! Checks the changepoint detectors: CUSUM against its known average run lengths, BOCPD
//! against the exact posterior by enumerating every segmentation. Neither has an R or
//! Python reference in the pinned environments.

use backshift::{Bocpd, Cusum, NormalGamma, Shift};

/// Standard normal values from a fixed seed (Box–Muller on a 64-bit LCG).
fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
        })
        .collect()
}

/// The mean number of values until an alarm, over `runs` runs with mean `shift`, and the
/// share of alarms upwards.
fn average_run_length(shift: f64, runs: usize) -> (f64, f64) {
    let z = normals(2_000_000, 11);
    let mut values = z.iter();
    let (mut total, mut up) = (0, 0);
    for _ in 0..runs {
        let mut cusum = Cusum::new(0.5, 4.0).unwrap();
        let mut length = 0;
        let alarm = loop {
            length += 1;
            if let Some(alarm) = cusum.update(values.next().unwrap() + shift) {
                break alarm;
            }
        };
        total += length;
        up += usize::from(alarm == Shift::Up);
    }
    (total as f64 / runs as f64, up as f64 / runs as f64)
}

#[test]
fn cusum_matches_its_average_run_lengths() {
    // Montgomery, table 9.3, k = 0.5, h = 4: 168 in control, 8.38 after a 1 sigma shift.
    // Run lengths are roughly geometric, so 4000 runs have a standard error near 168 / 63.
    let (in_control, _) = average_run_length(0.0, 4000);
    assert!((in_control - 168.0).abs() < 12.0, "ARL0 {in_control}");

    let (shifted, up) = average_run_length(1.0, 4000);
    assert!((shifted - 8.38).abs() < 0.3, "ARL1 {shifted}");
    assert!(up > 0.99, "only {up} of the alarms went up");
}

/// `ln Γ(x)` for `x` a positive multiple of 1/2, by recurrence from Γ(1/2) and Γ(1).
fn ln_gamma_half(x: f64) -> f64 {
    let (mut y, mut value) = if x.fract() == 0.0 {
        (1.0, 0.0)
    } else {
        (0.5, 0.5 * core::f64::consts::PI.ln())
    };
    while y < x {
        value += y.ln();
        y += 1.0;
    }
    value
}

/// The log marginal likelihood of `x` as one segment under `prior`.
fn segment_evidence(x: &[f64], prior: NormalGamma) -> f64 {
    let m = x.len() as f64;
    let mean = x.iter().sum::<f64>() / m;
    let squares: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    let precision = prior.precision + m;
    let shape = prior.shape + m / 2.0;
    let rate = prior.rate
        + squares / 2.0
        + prior.precision * m * (mean - prior.mean) * (mean - prior.mean) / (2.0 * precision);
    ln_gamma_half(shape) - ln_gamma_half(prior.shape) + prior.shape * prior.rate.ln()
        - shape * rate.ln()
        + 0.5 * (prior.precision / precision).ln()
        - m / 2.0 * core::f64::consts::TAU.ln()
}

/// The exact posterior of the current run length after `x`, index `r - 1` for length `r`,
/// by enumerating every segmentation: a segment starts before each value after the first
/// with probability `hazard`.
fn exact_posterior(x: &[f64], hazard: f64, prior: NormalGamma) -> Vec<f64> {
    let n = x.len();
    let mut posterior = vec![0.0; n];
    for cuts in 0..1_u32 << (n - 1) {
        // Bit `i` set: a segment starts before value `i + 1`.
        let mut log_weight = 0.0;
        let mut start = 0;
        for i in 0..n - 1 {
            if cuts >> i & 1 == 1 {
                log_weight += hazard.ln() + segment_evidence(&x[start..=i], prior);
                start = i + 1;
            } else {
                log_weight += (1.0 - hazard).ln();
            }
        }
        log_weight += segment_evidence(&x[start..], prior);
        posterior[n - start - 1] += log_weight.exp();
    }

    let total: f64 = posterior.iter().sum();
    posterior.iter().map(|p| p / total).collect()
}

/// The detector's probability of each run length `1..=n`.
fn run_lengths<const R: usize>(detector: &Bocpd<R>, n: usize) -> Vec<f64> {
    (1..=n)
        .map(|r| detector.probability_within(r) - detector.probability_within(r - 1))
        .collect()
}

#[test]
fn bocpd_matches_the_enumerated_posterior() {
    let prior = NormalGamma {
        mean: 0.3,
        precision: 2.0,
        shape: 1.0,
        rate: 1.5,
    };
    let x = [0.1, -0.4, 0.3, 2.5, 3.1, 2.2, 2.9, -1.0, 0.2, 0.4];
    let mut detector = Bocpd::<16>::new(0.2, prior).unwrap();
    for n in 1..=x.len() {
        detector.update(x[n - 1]);
        let exact = exact_posterior(&x[..n], 0.2, prior);
        for (r, (ours, exact)) in run_lengths(&detector, n).iter().zip(&exact).enumerate() {
            assert!(
                (ours - exact).abs() < 1e-12,
                "n {n}, run length {}: {ours} vs {exact}",
                r + 1
            );
        }
    }
}

/// The run length after `after` shifted values, preceded by 300 standard normal ones, with
/// `shift` applied to each shifted value.
fn detect(shift: impl Fn(f64) -> f64, after: usize) -> (usize, usize, f64) {
    let z = normals(300 + after, 23);
    let mut detector = Bocpd::<16>::new(1.0 / 250.0, NormalGamma::STANDARD).unwrap();
    for &v in &z[..300] {
        detector.update(v);
    }
    let before = detector.run_length();
    for &v in &z[300..] {
        detector.update(shift(v));
    }
    (
        before,
        detector.run_length(),
        detector.probability_within(after),
    )
}

#[test]
fn bocpd_finds_mean_and_scale_shifts_with_few_hypotheses() {
    let (before, after, probability) = detect(|v| v + 1.5, 25);
    assert!(before > 50, "no change yet, but run length {before}");
    assert!(
        (15..=25).contains(&after),
        "run length {after} after a mean shift"
    );
    assert!(probability > 0.9, "{probability}");

    let (_, after, probability) = detect(|v| 3.0 * v, 25);
    assert!(
        (15..=25).contains(&after),
        "run length {after} after a scale shift"
    );
    assert!(probability > 0.9, "{probability}");
}

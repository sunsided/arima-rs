//! MSTL: STL for several seasonal periods (Bandara, Hyndman & Bergmeir 2021,
//! arXiv:2107.13462), as statsmodels' `MSTL` computes it.
//!
//! Each pass removes one seasonal component after another with [`stl_with`], shortest
//! period first, each from the series with the other components removed. The trend is the
//! one of the last STL.
//!
//! ```mermaid
//! flowchart LR
//!     Y["y"] --> D["deseasonalised"]
//!     D -->|"+ seasonal i"| S["STL(period i, window i)"]
//!     S -->|"new seasonal i"| D
//!     S -->|"last STL"| T["trend"]
//! ```

use super::stl_with;
use alloc::vec;
use alloc::vec::Vec;

/// The components of an MSTL decomposition:
/// `values = sum(seasonal) + trend + remainder`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiDecomposition {
    /// The periods kept, in ascending order: those below half the series length.
    pub periods: Vec<usize>,
    /// One seasonal component per entry of `periods`.
    pub seasonal: Vec<Vec<f64>>,
    /// The trend component.
    pub trend: Vec<f64>,
    /// What is left: `values - sum(seasonal) - trend`.
    pub remainder: Vec<f64>,
}

impl MultiDecomposition {
    /// The series with every seasonal component removed: `trend + remainder`.
    #[must_use]
    pub fn seasonally_adjusted(&self) -> Vec<f64> {
        self.trend
            .iter()
            .zip(&self.remainder)
            .map(|(t, r)| t + r)
            .collect()
    }
}

/// statsmodels' default number of MSTL passes.
pub const MSTL_ITERATIONS: usize = 2;

/// MSTL with statsmodels' defaults: periods in ascending order, the seasonal smoother of
/// the `i`-th period (from 1) `7 + 4i`, non-robust STL, and `iterate` passes (one for a
/// single period).
///
/// Matches statsmodels `MSTL(values, periods=periods, iterate=iterate).fit()`. Periods of
/// at least half the series length are dropped, as statsmodels does. Box–Cox
/// transformation is not done. Returns [`None`] if no period is left, a period is below 2,
/// `iterate` is 0, or a value is not finite.
///
/// # Examples
///
/// ```
/// use backshift::{MSTL_ITERATIONS, mstl};
///
/// let y: Vec<f64> = (0..120)
///     .map(|t| [1.0, -1.0, 0.5][t % 3] + [2.0, 0.0, -1.0, 0.5, -1.5, 0.0, 0.0][t % 7])
///     .collect();
/// let parts = mstl(&y, &[7, 3], MSTL_ITERATIONS).unwrap();
/// assert_eq!(parts.periods, [3, 7]);
/// assert_eq!(parts.seasonal.len(), 2);
/// ```
#[must_use]
pub fn mstl(values: &[f64], periods: &[usize], iterate: usize) -> Option<MultiDecomposition> {
    let n = values.len();
    let mut kept: Vec<usize> = periods.iter().copied().filter(|&p| 2 * p < n).collect();
    kept.sort_unstable();
    let finite = values.iter().all(|v| v.is_finite());
    if !finite || kept.is_empty() || kept[0] < 2 || iterate == 0 {
        return None;
    }

    let passes = if kept.len() == 1 { 1 } else { iterate };
    let mut seasonal = vec![vec![0.0; n]; kept.len()];
    let mut deseasonalised = values.to_vec();
    let mut trend = Vec::new();
    for _ in 0..passes {
        for (i, (&period, component)) in kept.iter().zip(&mut seasonal).enumerate() {
            add(&mut deseasonalised, component, 1.0);
            let parts = stl_with(&deseasonalised, period, seasonal_window(i))?;
            *component = parts.seasonal;
            add(&mut deseasonalised, component, -1.0);
            trend = parts.trend;
        }
    }

    let remainder = deseasonalised
        .iter()
        .zip(&trend)
        .map(|(d, t)| d - t)
        .collect();
    Some(MultiDecomposition {
        periods: kept,
        seasonal,
        trend,
        remainder,
    })
}

/// statsmodels' seasonal smoother for the period at (0-based) position `i`: `7 + 4 (i + 1)`.
const fn seasonal_window(i: usize) -> usize {
    7 + 4 * (i + 1)
}

/// `target += sign * component`, element by element.
fn add(target: &mut [f64], component: &[f64], sign: f64) {
    for (t, c) in target.iter_mut().zip(component) {
        *t += sign * c;
    }
}

//! Streaming changepoint detection on standardised innovations, without a heap.
//!
//! A filter's innovation divided by the square root of its variance is close to standard
//! normal while the model holds. A change in the process shows up as a shift in the mean or
//! the scale of these values. Two detectors watch for it, one value at a time:
//!
//! - [`Cusum`]: Page's two-sided cumulative sum, the classic quality-control alarm for a
//!   shift in the mean, tuned by its average run length.
//! - [`Bocpd`]: Bayesian online changepoint detection (Adams and MacKay 2007), the posterior
//!   distribution of how long the current segment has lasted, for shifts in the mean or the
//!   scale.

mod bocpd;
mod cusum;

pub use bocpd::{Bocpd, NormalGamma};
pub use cusum::{Cusum, Shift};

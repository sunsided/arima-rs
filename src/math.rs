//! Floating-point functions that `core` lacks.
//!
//! `f64::sqrt` and friends live in `std`. On `no_std` targets the `libm`
//! feature supplies pure-Rust versions. When both features are on, `std` wins.

/// Square root of `x`; `NaN` for negative input.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Square root of `x`; `NaN` for negative input.
#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub(crate) fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// Natural logarithm of `x`; `NaN` for negative input, `-inf` for zero.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn ln(x: f64) -> f64 {
    x.ln()
}

/// Natural logarithm of `x`; `NaN` for negative input, `-inf` for zero.
#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub(crate) fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// `e^x`.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn exp(x: f64) -> f64 {
    x.exp()
}

/// `e^x`.
#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub(crate) fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// The dot product of two equally long slices.
#[cfg(feature = "alloc")]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

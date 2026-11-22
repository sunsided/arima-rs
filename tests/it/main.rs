//! Integration tests for the public API.
//!
//! - [`reference`] compares results against statsmodels reference vectors.
//! - [`properties`] checks mathematical invariants on random series with proptest.

mod properties;
mod reference;
mod reference_data;

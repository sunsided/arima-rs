//! A fixed-memory streaming forecaster for MSTL plus ARIMA: a [`Filter`] on the seasonally
//! adjusted series, and the last cycle of each seasonal component, repeated.
//!
//! [`AutoArima::fit_mstl`](crate::AutoArima::fit_mstl) fits the model offline;
//! [`MstlArima::stream`](crate::MstlArima::stream) turns it into an [`MstlFilter`] that runs
//! on a device without a heap. Each new value has its seasonal part removed before the
//! filter sees it, and each forecast adds the seasonal part back. The seasonal pattern stays
//! as fitted (seasonal naive); refit offline to update it.
//!
//! ```mermaid
//! flowchart LR
//!     Y["y_t"] -->|"- seasonal(t)"| A["adjusted"]
//!     A -->|update| F["Filter"]
//!     F -->|forecast| M["adjusted forecast"]
//!     M -->|"+ seasonal(t + h)"| P["forecast"]
//! ```

use crate::filter::{ArimaSpec, Filter, FilterError};
use crate::state_space::Innovation;
use core::error::Error;
use core::fmt::{Display, Formatter};

/// The most seasonal periods an [`MstlFilter`] holds.
pub const MAX_SEASONAL_PERIODS: usize = 4;

/// Why an [`MstlFilter`] could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MstlFilterError {
    /// The model does not fit the filter (see [`Filter::new`]).
    Filter(FilterError),
    /// The cycles hold more values than the capacity `L`.
    Capacity {
        /// The sum of the periods.
        needed: usize,
        /// The capacity `L`.
        capacity: usize,
    },
    /// No period, more than [`MAX_SEASONAL_PERIODS`], a period of 0, or `cycles` does not
    /// hold one value per step of every period.
    Periods,
}

impl Display for MstlFilterError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            MstlFilterError::Filter(error) => write!(f, "{error}"),
            MstlFilterError::Capacity { needed, capacity } => {
                write!(
                    f,
                    "the cycles need {needed} values, the filter holds {capacity}"
                )
            }
            MstlFilterError::Periods => write!(
                f,
                "need 1 to {MAX_SEASONAL_PERIODS} positive periods and one cycle value per step"
            ),
        }
    }
}

impl Error for MstlFilterError {}

impl From<FilterError> for MstlFilterError {
    fn from(error: FilterError) -> Self {
        MstlFilterError::Filter(error)
    }
}

/// A streaming MSTL plus ARIMA forecaster: a [`Filter`] with room for `R` states on the
/// seasonally adjusted series, and room for `L` seasonal values, the sum of the periods.
///
/// It lives inline with no heap. A period of 8766 (hourly data, yearly cycle) takes about
/// 70 KB, so keep a large one in a `static`.
///
/// # Examples
///
/// ```
/// use backshift::{ArimaSpec, MstlFilter};
///
/// // A white-noise adjusted series and a weekly pattern, as fitted offline.
/// let weekly = [2.0, 1.0, 0.0, -0.5, -1.0, -2.0, 0.5];
/// let mut stream = MstlFilter::<1, 7>::new(&ArimaSpec::default(), &[7], &weekly).unwrap();
/// let (mut mean, mut lower, mut upper) = ([0.0; 7], [0.0; 7], [0.0; 7]);
/// stream.forecast_interval_into(0.9, &mut mean, &mut lower, &mut upper);
/// assert_eq!(mean, weekly);
///
/// stream.update(2.5); // the seasonal value 2.0 is removed before filtering
/// stream.forecast_interval_into(0.9, &mut mean, &mut lower, &mut upper);
/// assert_eq!(mean[0], 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MstlFilter<const R: usize, const L: usize> {
    /// The filter on the seasonally adjusted series.
    filter: Filter<R>,
    /// The periods, then zeros; the first `count` entries are used.
    periods: [usize; MAX_SEASONAL_PERIODS],
    /// The number of periods.
    count: usize,
    /// Each period's cycle, one after another: value `j` of a period is its seasonal value
    /// `j` steps after the start.
    cycles: [f64; L],
    /// The steps since the start, the time index of the next value.
    step: u64,
}

impl<const R: usize, const L: usize> MstlFilter<R, L> {
    /// Builds a forecaster from the model of the adjusted series, its periods, and one
    /// cycle per period: `cycles` holds `periods[0]` values, then `periods[1]`, and so on,
    /// each starting at the seasonal value of the next observation. The filter starts at
    /// its stationary start; warm it up with [`MstlFilter::update`].
    ///
    /// This is the device path: it needs no heap.
    ///
    /// # Errors
    ///
    /// - [`MstlFilterError::Periods`] for no or too many periods, a period of 0, or a
    ///   `cycles` of the wrong length.
    /// - [`MstlFilterError::Capacity`] if the periods sum to more than `L`.
    /// - [`MstlFilterError::Filter`] if [`Filter::new`] rejects `spec`; it must have no
    ///   regressors.
    pub fn new(
        spec: &ArimaSpec<'_>,
        periods: &[usize],
        cycles: &[f64],
    ) -> Result<Self, MstlFilterError> {
        let total: usize = periods.iter().sum();
        let periods_ok = !periods.is_empty()
            && periods.len() <= MAX_SEASONAL_PERIODS
            && periods.iter().all(|&p| p > 0)
            && cycles.len() == total;
        if !periods_ok {
            return Err(MstlFilterError::Periods);
        }

        if total > L {
            return Err(MstlFilterError::Capacity {
                needed: total,
                capacity: L,
            });
        }

        let mut stored = [0; MAX_SEASONAL_PERIODS];
        stored[..periods.len()].copy_from_slice(periods);
        let mut values = [0.0; L];
        values[..total].copy_from_slice(cycles);
        Ok(Self {
            filter: Filter::new(spec)?,
            periods: stored,
            count: periods.len(),
            cycles: values,
            step: 0,
        })
    }

    /// Feeds one raw observation: removes its seasonal part, filters the rest, and returns
    /// the innovation of the adjusted value ([`Filter::update`]). A `NaN` counts as missing.
    pub fn update(&mut self, y: f64) -> Option<Innovation> {
        let adjusted = y - self.seasonal(self.step);
        self.step += 1;
        self.filter.update(adjusted)
    }

    /// Forecasts the next `mean.len()` values with central intervals at probability
    /// `level`: the filter's forecast of the adjusted series plus the repeated cycles. The
    /// intervals ignore the uncertainty of the seasonal part.
    ///
    /// # Panics
    ///
    /// Panics if `mean`, `lower` and `upper` do not all have the same length.
    pub fn forecast_interval_into(
        &self,
        level: f64,
        mean: &mut [f64],
        lower: &mut [f64],
        upper: &mut [f64],
    ) {
        self.filter
            .forecast_interval_into(level, mean, lower, upper);
        for (h, ((m, lo), up)) in mean.iter_mut().zip(lower).zip(upper).enumerate() {
            let seasonal = self.seasonal(self.step + h as u64);
            *m += seasonal;
            *lo += seasonal;
            *up += seasonal;
        }
    }

    /// Feeds one value of the already adjusted history before the start, which leaves the
    /// seasonal phase where it is.
    #[cfg(feature = "alloc")]
    pub(crate) fn warm_up_adjusted(&mut self, adjusted: f64) {
        self.filter.update(adjusted);
    }

    /// The sum of the seasonal values at time index `t` (steps since the start).
    fn seasonal(&self, t: u64) -> f64 {
        let mut offset = 0;
        let mut sum = 0.0;
        for &period in &self.periods[..self.count] {
            // `period` fits in a u64, and the remainder is below `period`, so it fits back.
            let phase = (t % period as u64) as usize;
            sum += self.cycles[offset + phase];
            offset += period;
        }

        sum
    }
}

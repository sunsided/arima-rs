//! The fitting objective on caller-owned memory: unconstrained parameters in, the
//! concentrated log-likelihood out.
//!
//! The optimiser's vector `u` is laid out as `[ar..., ma..., seasonal ar..., seasonal ma...,
//! mean, beta...]`, with the mean present only when it is estimated. Each evaluation maps
//! `u` to stationary and invertible coefficients, removes the regression and the mean from
//! the differenced series, and filters the result. Nothing here allocates.

use super::{Arima, transform};
use crate::buffer::take;
use crate::filter::{ArimaSpec, Seasonal};
use crate::likelihood::{concentrated, workspace_len};

/// Constrained parameters, borrowed from an [`Objective`]'s buffer.
pub(crate) struct Parameters<'a> {
    /// The AR coefficients.
    pub(crate) ar: &'a [f64],
    /// The MA coefficients.
    pub(crate) ma: &'a [f64],
    /// The seasonal AR coefficients.
    pub(crate) seasonal_ar: &'a [f64],
    /// The seasonal MA coefficients.
    pub(crate) seasonal_ma: &'a [f64],
    /// The mean of the differenced series, or 0.
    pub(crate) mean: f64,
    /// The regression coefficients.
    pub(crate) exog: &'a [f64],
}

/// The concentrated log-likelihood of one model on one differenced series, as a function
/// of the unconstrained parameters.
pub(crate) struct Objective<'a> {
    /// The model being fitted.
    model: &'a Arima,
    /// The differenced series, `NaN` where missing.
    dy: &'a [f64],
    /// The differenced regressors, column-major: column `j` is `dx[j n..(j + 1) n]`.
    dx: &'a [f64],
    /// The number of regressors.
    columns: usize,
    /// The constrained parameters of the last evaluation, laid out like `u`.
    params: &'a mut [f64],
    /// Scratch for [`transform::constrain`].
    work: &'a mut [f64],
    /// The differenced series with the regression and the mean removed.
    centered: &'a mut [f64],
    /// Scratch for the filter.
    filter: &'a mut [f64],
}

impl<'a> Objective<'a> {
    /// The scratch an objective for `model` needs, for `n` differenced values and
    /// `columns` regressors.
    pub(crate) fn scratch_len(model: &Arima, n: usize, columns: usize) -> usize {
        let parameters = model.parameter_count(columns);
        2 * parameters + n + workspace_len(model.arma_states())
    }

    /// An objective over the differenced series `dy` and the column-major differenced
    /// regressors `dx`, working in `scratch`.
    ///
    /// # Panics
    ///
    /// Panics if `scratch` holds fewer than [`Objective::scratch_len`] values.
    pub(crate) fn new(
        model: &'a Arima,
        dy: &'a [f64],
        dx: &'a [f64],
        columns: usize,
        mut scratch: &'a mut [f64],
    ) -> Self {
        let parameters = model.parameter_count(columns);
        Self {
            model,
            dy,
            dx,
            columns,
            params: take(&mut scratch, parameters),
            work: take(&mut scratch, parameters),
            centered: take(&mut scratch, dy.len()),
            filter: scratch,
        }
    }

    /// The concentrated log-likelihood at the unconstrained `u`: `(loglike, sigma2, n)`.
    ///
    /// With `steady` it takes the steady-state shortcut, as the optimiser does. Returns
    /// [`None`] where the likelihood cannot be evaluated (see [`concentrated`]).
    pub(crate) fn evaluate(&mut self, u: &[f64], steady: bool) -> Option<(f64, f64, usize)> {
        self.constrain(u);
        let parameters = split(self.model, self.params);
        residual_into(
            self.dy,
            self.dx,
            self.columns,
            parameters.exog,
            parameters.mean,
            self.centered,
        );

        let model = self.model;
        let shape = ArimaSpec {
            ar: parameters.ar,
            d: model.d,
            ma: parameters.ma,
            seasonal: Seasonal {
                ar: parameters.seasonal_ar,
                d: model.seasonal_d,
                ma: parameters.seasonal_ma,
                period: model.period,
            },
            ..ArimaSpec::default()
        };
        concentrated(self.centered, &shape, steady, self.filter)
    }

    /// The constrained parameters of the last [`Objective::evaluate`].
    pub(crate) fn parameters(&self) -> Parameters<'_> {
        split(self.model, self.params)
    }

    /// Maps `u` onto `params`.
    ///
    /// AR parts are negated after the transform and MA parts are not, as in statsmodels
    /// SARIMAX, so that both come out stationary or invertible. The mean and `beta` are
    /// copied unchanged.
    fn constrain(&mut self, u: &[f64]) {
        let model = self.model;
        let parts = [
            (model.p, true),
            (model.q, false),
            (model.seasonal_p, true),
            (model.seasonal_q, false),
        ];
        let mut start = 0;
        for (len, negate) in parts {
            let range = start..start + len;
            transform::constrain(
                &u[range.clone()],
                &mut self.params[range.clone()],
                self.work,
            );
            if negate {
                for value in &mut self.params[range] {
                    *value = -*value;
                }
            }

            start += len;
        }

        self.params[start..].copy_from_slice(&u[start..]);
    }
}

/// Splits a parameter vector laid out like `u` into its parts for `model`.
fn split<'p>(model: &Arima, params: &'p [f64]) -> Parameters<'p> {
    let (ar, rest) = params.split_at(model.p);
    let (ma, rest) = rest.split_at(model.q);
    let (seasonal_ar, rest) = rest.split_at(model.seasonal_p);
    let (seasonal_ma, rest) = rest.split_at(model.seasonal_q);
    let mean = if model.mean { rest[0] } else { 0.0 };
    Parameters {
        ar,
        ma,
        seasonal_ar,
        seasonal_ma,
        mean,
        exog: &rest[usize::from(model.mean)..],
    }
}

/// Writes `dy_t - dx_t . beta - mean` into `out`, with `dx` column-major.
pub(crate) fn residual_into(
    dy: &[f64],
    dx: &[f64],
    columns: usize,
    beta: &[f64],
    mean: f64,
    out: &mut [f64],
) {
    let n = dy.len();
    for (t, value) in out.iter_mut().enumerate().take(n) {
        let regression = (0..columns).map(|j| dx[j * n + t] * beta[j]).sum::<f64>();
        *value = dy[t] - regression - mean;
    }
}

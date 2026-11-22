# @sunsided/backshift

Seasonal ARIMA models in the browser and in Node, compiled from the Rust crate
[`backshift`](https://crates.io/crates/backshift) to WebAssembly: exact maximum-likelihood fits, R's `auto.arima`
search, forecasts with prediction intervals, a streaming filter that scores every new value, and in-place refits.
The results match statsmodels to 1e-9. [Try it in the browser](https://sunsided.github.io/arima-rs/).

```sh
npm install @sunsided/backshift
```

## Usage

The package is an ES module. Initialise it once, then fit, forecast and stream:

```js
import init, { Model, Stream } from "@sunsided/backshift";

await init(); // in Node: initSync({ module: readFileSync(wasmPath) })

const history = Float64Array.from([/* monthly values */]);
const model = Model.autoArima(history, 12);
console.log(model.label(), model.aicc);

// The next 24 months with 95% prediction intervals.
const forecast = model.forecast(history, 24, 0.95);
console.log(forecast.mean(), forecast.lower(), forecast.upper());

// Score new values as they arrive: the innovation and its variance.
const stream = new Stream(model);
stream.warmUp(history);
const [innovation, variance] = stream.update(512.0);
const isAnomaly = Math.abs(innovation) > 3 * Math.sqrt(variance);
```

- `Model.fit(values, p, d, q, P, D, Q, period, mean?)` fits one model and `Model.autoArima(values, period)` searches for
  one; `NaN` marks a missing value.
- `Model` exposes the coefficients (`ar()`, `ma()`, `seasonalAr()`, `seasonalMa()`, `exog()`), `mean`, `sigma2`, `loglike`,
  `aic`, `aicc`, `bic` and `label()`.
- `Model.fromCoefficients(ar, d, ma, seasonalAr, D, seasonalMa, period, mean, sigma2)` takes a model fitted elsewhere
  (statsmodels, R) in their sign convention, to stream and forecast without fitting.
- `model.refit(window, mean)` re-estimates the model on recent values, warm-started from the current one.
- Regressors (SARIMAX, regression with ARIMA errors as in R's `arima(xreg=)`): pass one row-major `Float64Array`
  with a row per value, `Model.fit(values, p, d, q, P, D, Q, period, mean, regressors, columns)` or
  `Model.autoArima(values, period, regressors, columns)`. `model.exog()` returns the coefficients. Forecasts need the
  rows of the history and of the horizon, `model.forecast(history, horizon, level, regressors, future)`; a `Stream`
  takes them in `warmUp(values, regressors)`, `update(value, row)` and `forecast(horizon, level, future)`, and
  `refit(window, mean, regressors)` and `fromCoefficients(..., sigma2, exog)` take them too.
- `ljungBox(residuals, lags, modelDf)` returns `[statistic, pValue]`.

Invalid input throws an `Error` with a message; nothing panics. Streaming and forecasts hold the model in a fixed
filter of `filterCapacity()` (64) states, enough for the airline model at period 24.

## License

MIT or Apache-2.0, at your option.

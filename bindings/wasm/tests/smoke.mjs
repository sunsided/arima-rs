// Loads the built package in Node and exercises every export once.
// Run after build.sh: node bindings/wasm/tests/smoke.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const pkg = fileURLToPath(new URL("../pkg/", import.meta.url));
const { initSync, Model, Stream, ljungBox, filterCapacity } = await import(`${pkg}backshift.js`);
initSync({ module: readFileSync(`${pkg}backshift_bg.wasm`) });

// A seasonal series: trend, period-12 pattern and seeded noise.
let state = 7;
const noise = () => {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
  return state / 2 ** 32 - 0.5;
};
const y = Float64Array.from({ length: 144 }, (_, t) => 100 + 0.5 * t + 10 * Math.sin((2 * Math.PI * t) / 12) + 3 * noise());

const auto = Model.autoArima(y, 12);
assert.match(auto.label(), /^ARIMA\(/);
assert.ok(Number.isFinite(auto.aicc) && auto.sigma2 > 0);

const airline = Model.fit(y, 0, 1, 1, 0, 1, 1, 12);
assert.equal(airline.label(), "ARIMA(0,1,1)(0,1,1)[12]");

// Approximate fits report the exact likelihood at CSS estimates, so they never beat the exact fit.
const quick = Model.fit(y, 0, 1, 1, 0, 1, 1, 12, undefined, undefined, undefined, true);
assert.ok(quick.loglike <= airline.loglike + 1e-6 && quick.loglike > airline.loglike - 5);
const quickAuto = Model.autoArima(y, 12, undefined, undefined, true);
assert.match(quickAuto.label(), /^ARIMA\(/);
assert.equal(airline.states, 27);
const forecast = airline.forecast(y, 24, 0.95);
const [mean, lower, upper] = [forecast.mean(), forecast.lower(), forecast.upper()];
assert.equal(mean.length, 24);
assert.ok(mean.every((m, h) => lower[h] < m && m < upper[h]));

const stream = new Stream(airline);
stream.warmUp(y);
const [innovation, variance] = stream.update(y[143] + 0.5);
assert.ok(Number.isFinite(innovation) && variance > 0);
assert.equal(stream.update(Number.NaN)[0], Number.NaN);

const refit = airline.refit(y.slice(-96), false);
assert.ok(refit.loglike > -Infinity && refit.ma().length === 1);

const [statistic, pValue] = ljungBox(Float64Array.from({ length: 50 }, noise), 10, 0);
assert.ok(statistic > 0 && pValue > 0 && pValue <= 1);
assert.equal(filterCapacity(), 64);

// Errors arrive as exceptions, not as a dead instance.
assert.throws(() => Model.fit(y, 0, 0, 0, 1, 0, 0, 0), /period/);
assert.throws(() => Model.fit(y.slice(0, 3), 2, 0, 2, 0, 0, 0, 0), /needs/);
assert.throws(() => new Stream(Model.fit(y, 0, 1, 1, 0, 1, 1, 52)), /states/);
// Orders that are each in range can still ask for thousands of states; they are refused.
assert.throws(() => Model.fit(y, 0, 0, 0, 8, 0, 8, 366), /states/);
assert.throws(() => Model.autoArima(y, 100), /periods up to/);
assert.ok(Model.fit(y, 1, 0, 0, 0, 0, 0, 0).sigma2 > 0, "the instance survives the errors");

// A model fitted elsewhere streams without a fit.
const given = Model.fromCoefficients([0.5], 1, [0.3], [], 0, [], 0, 0.2, 1.5);
assert.equal(given.label(), "ARIMA(1,1,1) with drift");
assert.ok(Number.isNaN(given.loglike));
assert.ok(given.forecast(y, 3, 0.9).mean().every(Number.isFinite));
assert.throws(() => Model.fromCoefficients([1.2], 0, [], [], 0, [], 0, 0, 1), /stationary/);
assert.throws(() => Model.fromCoefficients([0.5], 0, [], [], 0, [], 0, 0, 0), /sigma2/);

// The largest models the filter holds fit on the wasm stack: 63 of 64 states.
const large = Model.fit(y, 1, 1, 1, 1, 1, 1, 30);
assert.equal(large.states, 63);
const largeStream = new Stream(large);
largeStream.warmUp(y);
assert.ok(largeStream.forecast(48, 0.9).upper().every(Number.isFinite));

// SARIMAX: weekly data with a holiday dummy, known for the horizon too.
const days = 280;
const holiday = Float64Array.from({ length: days }, (_, t) => (t % 37 === 5 ? 1 : 0));
const users = Float64Array.from(
  { length: days },
  (_, t) => 1000 - 300 * (t % 7 >= 5) - 250 * holiday[t] + 15 * noise(),
);
const sarimax = Model.fit(users, 1, 0, 0, 0, 1, 1, 7, undefined, holiday, 1);
assert.equal(sarimax.exog().length, 1);
assert.ok(Math.abs(sarimax.exog()[0] + 250) < 25, `holiday effect ${sarimax.exog()[0]}`);
const future = Float64Array.from({ length: 14 }, (_, h) => (h === 3 ? 1 : 0));
const ahead = sarimax.forecast(users, 14, 0.95, holiday, future).mean();
assert.equal(ahead.length, 14);
const plainAhead = Model.fit(users, 1, 0, 0, 0, 1, 1, 7).forecast(users, 14, 0.95).mean();
assert.ok(ahead[3] < plainAhead[3] - 150, "the known holiday lowers the forecast");
const autoX = Model.autoArima(users, 7, holiday, 1);
assert.equal(autoX.exog().length, 1);
const sarimaxStream = new Stream(sarimax);
sarimaxStream.warmUp(users, holiday);
assert.ok(Number.isFinite(sarimaxStream.update(1000, [0])[0]));
assert.equal(sarimaxStream.forecast(14, 0.9, future).mean().length, 14);
assert.equal(sarimax.refit(users.slice(-140), false, holiday.slice(-140)).exog().length, 1);
const givenX = Model.fromCoefficients([0.5], 0, [], [], 0, [], 0, 0, 1, [2]);
assert.deepEqual([...givenX.exog()], [2]);
assert.equal(givenX.forecast(y, 1, 0.9, new Float64Array(144), [1]).mean().length, 1);

// Missing or misshapen rows throw instead of killing the instance.
assert.throws(() => Model.fit(users, 1, 0, 0, 0, 1, 1, 7, undefined, holiday), /column count/);
assert.throws(() => Model.fit(users, 1, 0, 0, 0, 1, 1, 7, undefined, holiday.slice(1), 1), /rows/);
assert.throws(() => sarimax.forecast(users, 14, 0.95), /regressors/);
assert.throws(() => sarimax.forecast(users, 14, 0.95, holiday), /horizon/);
assert.throws(() => sarimaxStream.update(1000), /regressors/);
assert.throws(() => sarimaxStream.update(1000, [0, 1]), /regressors/);
assert.throws(() => sarimaxStream.warmUp(users), /regressors/);
assert.throws(() => sarimaxStream.forecast(14, 0.9), /regressors/);
assert.throws(() => sarimax.refit(users, false), /regressors/);
assert.throws(() => airline.forecast(y, 3, 0.9, new Float64Array(144), [1, 1, 1]), /no regressors/);
assert.throws(() => Model.fromCoefficients([0.5], 0, [], [], 0, [], 0, 0, 1, new Array(65).fill(0)), /columns/);
assert.ok(sarimaxStream.update(1000, [1])[1] > 0, "the instance survives the errors");

console.log(`smoke: ok (${auto.label()}, AICc ${auto.aicc.toFixed(2)})`);

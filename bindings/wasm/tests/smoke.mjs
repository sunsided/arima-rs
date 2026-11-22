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
assert.ok(Model.fit(y, 1, 0, 0, 0, 0, 0, 0).sigma2 > 0, "the instance survives the errors");

// The largest models the filter holds fit on the wasm stack: 63 of 64 states.
const large = Model.fit(y, 1, 1, 1, 1, 1, 1, 30);
assert.equal(large.states, 63);
const largeStream = new Stream(large);
largeStream.warmUp(y);
assert.ok(largeStream.forecast(48, 0.9).upper().every(Number.isFinite));

console.log(`smoke: ok (${auto.label()}, AICc ${auto.aicc.toFixed(2)})`);

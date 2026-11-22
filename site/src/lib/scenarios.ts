// The demonstrations, as plain functions over the package: each runs the model and
// returns the chart it implies. The site renders them interactively; the README script
// renders the same functions to static images.

import type * as Backshift from '@sunsided/backshift';
import type { ChartSpec, Role } from './chart.ts';
import type { ActiveUsers } from './data.ts';

/** The package's exports the scenarios use. */
export type Api = Pick<typeof Backshift, 'Model' | 'Stream' | 'ljungBox'>;

/** A fitted model and its forecast, ready to draw. */
export interface ForecastResult {
	label: string;
	aicc: number;
	sigma2: number;
	fitMillis: number;
	chart: ChartSpec;
}

/** The orders of a manual fit. */
export interface Orders {
	p: number;
	d: number;
	q: number;
	seasonalP: number;
	seasonalD: number;
	seasonalQ: number;
}

/**
 * Fits `values` (by `auto.arima` without `orders`) and forecasts `horizon` steps with
 * 80% and 95% intervals.
 */
export function forecastScenario(
	api: Api,
	values: number[],
	period: number,
	horizon: number,
	orders?: Orders,
	size = { width: 760, height: 320 }
): ForecastResult {
	const history = Float64Array.from(values);
	const started = performance.now();
	const model = orders
		? api.Model.fit(history, orders.p, orders.d, orders.q, orders.seasonalP, orders.seasonalD, orders.seasonalQ, period)
		: api.Model.autoArima(history, period);
	const fitMillis = performance.now() - started;

	// Free the wasm objects even when a forecast throws (a model too large to stream).
	let wide: Backshift.Forecast | undefined;
	let narrow: Backshift.Forecast | undefined;
	try {
		wide = model.forecast(history, horizon, 0.95);
		narrow = model.forecast(history, horizon, 0.8);
		return forecastResult(model, values, horizon, fitMillis, wide, narrow, size);
	} finally {
		wide?.free();
		narrow?.free();
		model.free();
	}
}

/** The chart and statistics of a fitted model and its two forecasts. */
function forecastResult(
	model: Backshift.Model,
	values: number[],
	horizon: number,
	fitMillis: number,
	wide: Backshift.Forecast,
	narrow: Backshift.Forecast,
	size: { width: number; height: number }
): ForecastResult {
	const n = values.length;
	const ahead = Array.from({ length: horizon }, (_, h) => n + h);
	// The forecast line starts at the last observation, so it reads as a continuation.
	const join = (series: Float64Array) => [values[n - 1], ...series];
	const joinX = [n - 1, ...ahead];
	const result: ForecastResult = {
		label: model.label(),
		aicc: model.aicc,
		sigma2: model.sigma2,
		fitMillis,
		chart: {
			...size,
			title: `${model.label()}: history and a ${horizon}-step forecast with 80% and 95% intervals`,
			layers: [
				{ kind: 'band', x: joinX, lower: join(wide.lower()), upper: join(wide.upper()), role: 'band-outer', label: '95% interval' },
				{ kind: 'band', x: joinX, lower: join(narrow.lower()), upper: join(narrow.upper()), role: 'band-inner', label: '80% interval' },
				{ kind: 'line', x: values.map((_, t) => t), y: values, role: 'data', label: 'Observed' },
				{ kind: 'line', x: joinX, y: join(wide.mean()), role: 'forecast', label: 'Forecast', dashed: true }
			],
			markers: [{ x: n - 1, label: 'now' }]
		}
	};
	return result;
}

/** One streamed value: its innovation and whether it crossed the alarm threshold. */
export interface StreamPoint {
	t: number;
	value: number;
	innovation: number;
	bound: number;
	/** Whether the innovation crossed the band. */
	alarm: boolean;
	/** Whether this value is an anomaly: an alarm that does not follow another one. */
	anomaly: boolean;
}

/** The innovation threshold in standard deviations. */
export const ALARM_SIGMAS = 4;

/**
 * Fits an AR(1) with mean on `training`, then streams `live` through the heap-free filter
 * and scores every value: an innovation beyond 4 standard deviations is an anomaly. The
 * model is stationary on purpose: a differenced one would flag a spike twice, on the way
 * up and on the way back.
 */
export function streamScenario(api: Api, training: number[], live: number[]): { label: string; points: StreamPoint[]; microsPerUpdate: number } {
	const model = api.Model.fit(Float64Array.from(training), 1, 0, 0, 0, 0, 0, 0, true);
	const stream = new api.Stream(model);
	stream.warmUp(Float64Array.from(training));
	const points: StreamPoint[] = [];
	const started = performance.now();
	for (const [i, value] of live.entries()) {
		const [innovation, variance] = stream.update(value) ?? [Number.NaN, Number.NaN];
		const bound = ALARM_SIGMAS * Math.sqrt(variance);
		// An outlier also throws off the next prediction, so the value after it has a large
		// innovation too; only the first of consecutive alarms is the anomaly.
		const previous = points[points.length - 1];
		const alarm = Math.abs(innovation) > bound;
		points.push({ t: training.length + i, value, innovation, bound, anomaly: alarm && !(previous?.alarm ?? false), alarm });
	}
	const microsPerUpdate = ((performance.now() - started) * 1000) / Math.max(1, live.length);
	const label = model.label();
	stream.free();
	model.free();
	return { label, points, microsPerUpdate };
}

/** The chart of a stream: the innovations, their alarm band, and the flagged values. */
export function streamChart(points: StreamPoint[], size = { width: 760, height: 260 }, xDomain?: [number, number]): ChartSpec {
	const x = points.map((p) => p.t);
	const flagged = points.filter((p) => p.anomaly);
	return {
		...size,
		title: 'Streaming innovations with a 4-sigma alarm band; anomalies marked',
		xDomain,
		layers: [
			{ kind: 'band', x, lower: points.map((p) => -p.bound), upper: points.map((p) => p.bound), role: 'band-outer', label: '±4σ' },
			{ kind: 'line', x, y: points.map((p) => p.innovation), role: 'data', label: 'Innovation' },
			{ kind: 'points', x: flagged.map((p) => p.t), y: flagged.map((p) => p.innovation), role: 'critical', label: 'Anomaly' }
		],
		rules: [{ y: 0, role: 'muted' }]
	};
}

/** What happened at the end of one window of the drift loop. */
export interface WindowResult {
	start: number;
	end: number;
	pValue: number;
	refit: boolean;
	/** The AR coefficient in use after this window. */
	ar: number;
}

/** The Ljung–Box significance level below which a window triggers a refit. */
export const DRIFT_ALPHA = 0.001;

/**
 * The device loop of `examples/device_refit.rs`: stream `values` through an AR(1) filter,
 * test each window's standardised innovations with Ljung–Box, and refit on drift.
 */
export function driftScenario(api: Api, values: number[], window: number, startAr: number): WindowResult[] {
	// The deployed model: coefficients fitted elsewhere, as on a device.
	let model = api.Model.fromCoefficients(Float64Array.of(startAr), 0, new Float64Array(), new Float64Array(), 0, new Float64Array(), 0, 0, 1);
	let stream = new api.Stream(model);
	const standardised = new Float64Array(window);
	const results: WindowResult[] = [];
	for (let start = 0; start + window <= values.length; start += window) {
		const chunk = values.slice(start, start + window);
		chunk.forEach((value, i) => {
			const [innovation, variance] = stream.update(value) ?? [Number.NaN, 1];
			standardised[i] = innovation / Math.sqrt(variance);
		});
		const [, pValue] = api.ljungBox(standardised, 10, 1) ?? [Number.NaN, 1];
		let refit = false;
		if (pValue < DRIFT_ALPHA) {
			const next = model.refit(Float64Array.from(chunk), false);
			model.free();
			model = next;
			stream.free();
			stream = new api.Stream(model);
			stream.warmUp(Float64Array.from(chunk));
			refit = true;
		}
		results.push({ start, end: start + window, pValue, refit, ar: model.ar()[0] });
	}
	stream.free();
	model.free();
	return results;
}

/** The drift chart: the series with refit windows shaded and the AR coefficient over time. */
export function driftCharts(
	values: number[],
	results: WindowResult[],
	truth: { change: number; before: number; after: number },
	size = { width: 760, height: 240 }
): { series: ChartSpec; coefficient: ChartSpec } {
	const x = values.map((_, t) => t);
	const refits = results.filter((r) => r.refit);
	const steps = results.map((r) => r.end);
	return {
		series: {
			...size,
			title: 'A stream whose AR coefficient changes; windows that triggered a refit shaded',
			layers: [{ kind: 'line', x, y: values, role: 'data', label: 'Observed' }],
			regions: refits.map((r) => ({ x0: r.start, x1: r.end, role: 'critical' })),
			markers: [{ x: truth.change, label: 'process changes' }],
			legend: false
		},
		coefficient: {
			...size,
			height: 200,
			title: 'The AR coefficient the device uses, against the true one',
			xDomain: [0, values.length],
			yDomain: Math.min(truth.before, truth.after, ...results.map((r) => r.ar)) < 0 ? [-1, 1] : [0, 1],
			layers: [
				{ kind: 'step', x: [0, truth.change, values.length], y: [truth.before, truth.after, truth.after], role: 'muted', label: 'True φ', dashed: true },
				{ kind: 'step', x: [0, ...steps], y: [results[0]?.ar ?? truth.before, ...results.map((r) => r.ar)], role: 'refit', label: 'Device φ' },
				{ kind: 'points', x: refits.map((r) => r.end), y: refits.map((r) => r.ar), role: 'refit', label: 'Refit' }
			]
		}
	};
}

/** One side of the calendar comparison: a fit and its forecast. */
export interface CalendarFit {
	aicc: number;
	sigma: number;
	mean: Float64Array;
	lower: Float64Array;
	upper: Float64Array;
}

/** The calendar scenario: the fits with and without regressors, and their chart. */
export interface CalendarResult {
	label: string;
	/** The fitted users lost on a holiday and during an outage. */
	effects: { holiday: number; outage: number };
	with: CalendarFit;
	without: CalendarFit;
	fitMillis: number;
	chart: ChartSpec;
}

/**
 * The SARIMAX demonstration of `examples/holidays_outages.rs`: fits SARIMA(1,0,0)(0,1,1,7)
 * to `data` with and without the holiday and outage regressors, and forecasts `horizon`
 * days with 95% intervals. The calendar knows the holiday `upcoming` days ahead (0-based);
 * no outage is planned. The chart shows the last `shown` days of history.
 */
export function calendarScenario(
	api: Api,
	data: ActiveUsers,
	horizon = 28,
	upcoming = 10,
	shown = 140,
	size = { width: 760, height: 320 }
): CalendarResult {
	const history = Float64Array.from(data.values);
	const regressors = Float64Array.from(data.regressors);
	const future = Float64Array.from({ length: 2 * horizon }, (_, i) => Number(i === 2 * upcoming));
	const started = performance.now();
	const withModel = api.Model.fit(history, 1, 0, 0, 0, 1, 1, 7, undefined, regressors, 2);
	let withoutModel: Backshift.Model | undefined;
	let wide: Backshift.Forecast | undefined;
	let plain: Backshift.Forecast | undefined;
	try {
		withoutModel = api.Model.fit(history, 1, 0, 0, 0, 1, 1, 7);
		const fitMillis = performance.now() - started;
		wide = withModel.forecast(history, horizon, 0.95, regressors, future);
		plain = withoutModel.forecast(history, horizon, 0.95);
		const side = (model: Backshift.Model, forecast: Backshift.Forecast): CalendarFit => ({
			aicc: model.aicc,
			sigma: Math.sqrt(model.sigma2),
			mean: forecast.mean(),
			lower: forecast.lower(),
			upper: forecast.upper()
		});
		const [holiday, outage] = withModel.exog();
		const result = { with: side(withModel, wide), without: side(withoutModel, plain) };
		return {
			label: withModel.label(),
			effects: { holiday, outage },
			...result,
			fitMillis,
			chart: calendarChart(data, result.with, result.without, upcoming, shown, size)
		};
	} finally {
		wide?.free();
		plain?.free();
		withoutModel?.free();
		withModel.free();
	}
}

/** The calendar chart: recent history with holidays and outages shaded, and both forecasts. */
function calendarChart(
	data: ActiveUsers,
	withFit: CalendarFit,
	withoutFit: CalendarFit,
	upcoming: number,
	shown: number,
	size: { width: number; height: number }
): ChartSpec {
	const n = data.values.length;
	const from = Math.max(0, n - shown);
	const x = data.values.slice(from).map((_, i) => from + i);
	const joinX = [n - 1, ...withFit.mean.map((_, h) => n + h)];
	const join = (series: Float64Array) => [data.values[n - 1], ...series];
	const day = (t: number, role: Role) => ({ x0: t - 0.5, x1: t + 0.5, role });
	return {
		...size,
		title: 'Daily active users: holidays (grey) and outages (red) shaded, forecasts with and without regressors',
		xLabel: 'day',
		layers: [
			{ kind: 'band', x: joinX, lower: join(withFit.lower), upper: join(withFit.upper), role: 'band-outer', label: '95% interval, with regressors' },
			{ kind: 'line', x, y: data.values.slice(from), role: 'data', label: 'Observed' },
			{ kind: 'line', x: joinX, y: join(withFit.mean), role: 'forecast', label: 'Forecast with regressors', dashed: true },
			// On top, so it shows where the two forecasts part: at the holiday.
			{ kind: 'line', x: joinX, y: join(withoutFit.mean), role: 'muted', label: 'Forecast without regressors', dashed: true }
		],
		regions: [
			...data.holidays.filter((t) => t >= from).map((t) => day(t, 'muted')),
			...data.outages.filter((t) => t >= from).map((t) => day(t, 'critical')),
			day(n + upcoming, 'muted')
		],
		// No 'now' marker: its label would collide with the holiday's, 10 days on.
		markers: [{ x: n + upcoming, label: 'bank holiday' }]
	};
}

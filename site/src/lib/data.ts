// Seeded example series and a strict parser for pasted data. The generators are
// deterministic, so the site and the README charts show the same numbers on every run.

/** A uniform generator on [0, 1): mulberry32, seeded. */
export function uniform(seed: number): () => number {
	let state = seed >>> 0;
	return () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let t = state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** A standard normal generator (Box–Muller), seeded. */
export function normal(seed: number): () => number {
	const u = uniform(seed);
	return () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
}

/**
 * The airline model ARIMA(0,1,1)(0,1,1)[12]: a trend with a slowly changing yearly
 * pattern, the classic shape of monthly sales or passengers.
 */
export function airline(n = 144, seed = 12): number[] {
	const e = normal(seed);
	const [theta, seasonalTheta, scale] = [-0.4, -0.6, 3];
	const noise = Array.from({ length: n }, () => scale * e());
	const pattern = [-12, -16, -2, -5, -3, 12, 26, 25, 9, -7, -21, -8];
	const y: number[] = [];
	for (let t = 0; t < n; t++) {
		if (t < 13) {
			y.push(120 + 1.5 * t + pattern[t % 12] + noise[t]);
			continue;
		}
		const w =
			noise[t] + theta * noise[t - 1] + seasonalTheta * noise[t - 12] + theta * seasonalTheta * noise[t - 13];
		y.push(w + y[t - 1] + y[t - 12] - y[t - 13]);
	}
	return y;
}

/** A random walk with drift 0.3: ARIMA(0,1,0) with drift. */
export function randomWalk(n = 200, seed = 7): number[] {
	const e = normal(seed);
	let level = 50;
	return Array.from({ length: n }, () => (level += 0.3 + e()));
}

/** A quasi-periodic AR(2), `y_t = 1.6 y_(t-1) - 0.8 y_(t-2) + e_t`, around 10. */
export function cycle(n = 200, seed = 3): number[] {
	const e = normal(seed);
	const y: number[] = [];
	let [a, b] = [0, 0];
	for (let t = 0; t < n; t++) {
		const next = 1.6 * a - 0.8 * b + e();
		[b, a] = [a, next];
		y.push(10 + next);
	}
	return y;
}

/**
 * A sensor reading: an AR(1) around 20 with a daily-like cycle, and spikes at the given
 * positions, the anomalies a streaming filter should flag.
 */
export function sensor(n: number, seed: number, spikes: number[]): number[] {
	const e = normal(seed);
	const y: number[] = [];
	let state = 0;
	for (let t = 0; t < n; t++) {
		state = 0.8 * state + 0.5 * e();
		y.push(20 + state + (spikes.includes(t) ? 4.5 : 0));
	}
	return y;
}

/** An AR(1) whose coefficient jumps from `before` to `after` at `change`. */
export function regimeChange(n: number, change: number, before: number, after: number, seed = 11): number[] {
	const e = normal(seed);
	const y: number[] = [];
	let state = 0;
	for (let t = 0; t < n; t++) {
		state = (t < change ? before : after) * state + e();
		y.push(state);
	}
	return y;
}

/** Two years of daily active users, interrupted by bank holidays and server outages. */
export interface ActiveUsers {
	values: number[];
	/** The regressor rows, row-major: `[holiday, outage]` per day. */
	regressors: number[];
	holidays: number[];
	outages: number[];
	/** The users lost on a holiday and during an outage. */
	effects: { holiday: number; outage: number };
}

/**
 * Daily active users with quiet weekends, carried by seasonal ARIMA noise
 * `(1 - 0.5 B)(1 - B^7) u_t = (1 - 0.6 B^7) e_t` with `sigma = 15`, minus 250 users on
 * a bank holiday and 400 during an outage: the scenario of `examples/holidays_outages.rs`.
 */
export function activeUsers(seed = 7, days = 730): ActiveUsers {
	const e = normal(seed);
	const holidays = [0, 88, 91, 120, 129, 139, 275, 358, 359, 364, 365, 453, 456, 485, 494, 504, 640, 723, 724, 729];
	const outages = [212, 401, 602, 603];
	const effects = { holiday: -250, outage: -400 };
	// Monday to Sunday.
	const week = [1000, 1040, 1050, 1030, 980, 700, 650];
	const [phi, theta, sigma] = [0.5, -0.6, 15];
	const shocks: number[] = [];
	const u: number[] = [];
	let z = 0;
	for (let t = 0; t < days; t++) {
		const shock = sigma * e();
		shocks.push(shock);
		if (t < 7) {
			u.push(week[t] + shock);
			continue;
		}
		// z = (1 - B^7) u follows an ARMA(1, 0)(0, 1) model.
		z = phi * z + shock + theta * shocks[t - 7];
		u.push(u[t - 7] + z);
	}

	const regressors = u.flatMap((_, t) => [Number(holidays.includes(t)), Number(outages.includes(t))]);
	const values = u.map((value, t) => value + regressors[2 * t] * effects.holiday + regressors[2 * t + 1] * effects.outage);
	return { values, regressors, holidays, outages, effects };
}

/** A field separator: a comma, semicolon or tab with any spaces after it, or spaces. */
const FIELD = /[,;\t] *| +/;

/** The most values a pasted series may have, as the package accepts. */
export const MAX_VALUES = 100_000;

/**
 * Parses pasted values: numbers separated by commas, semicolons, tabs, spaces or newlines.
 * An empty field or `NA`/`NaN`/`null` is a missing value; anything else is an error.
 * With several columns per line, the last one is taken, so `date,value` rows work.
 */
export function parseSeries(text: string): { values: number[] } | { error: string } {
	let lines = text.split(/\r?\n/).filter((line) => line.trim() !== '');
	// A single line holds the whole series, one value per field.
	if (lines.length === 1) lines = lines[0].split(FIELD);
	const values: number[] = [];
	for (const [index, line] of lines.entries()) {
		const fields = line.split(FIELD).map((field) => field.trim());
		const field = fields[fields.length - 1];
		if (/^(|na|nan|null)$/i.test(field)) {
			values.push(Number.NaN);
			continue;
		}
		const value = Number(field);
		if (!Number.isFinite(value)) {
			// A header line is fine; any other non-number is a mistake worth reporting.
			if (index === 0) continue;
			return { error: `line ${index + 1}: "${field}" is not a number` };
		}
		values.push(value);
	}

	if (values.length < 8) return { error: 'paste at least 8 values' };
	if (values.length > MAX_VALUES) return { error: `at most ${MAX_VALUES} values` };
	return { values };
}

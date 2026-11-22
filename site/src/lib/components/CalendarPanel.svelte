<script lang="ts">
	// Runs examples/holidays_outages.rs: daily users interrupted by bank holidays and
	// outages, fitted with and without regressors that mark them (SARIMAX), and forecast
	// through a bank holiday the calendar already knows about.
	import Chart from './Chart.svelte';
	import { activeUsers } from '#lib/data.ts';
	import { calendarScenario, type Api } from '#lib/scenarios.ts';

	let { api }: { api: Api } = $props();

	const HORIZON = 28;
	const UPCOMING = 10;

	let seed = $state(7);

	const data = $derived(activeUsers(seed));
	const result = $derived(calendarScenario(api, data, HORIZON, UPCOMING));
	const n = $derived(data.values.length);

	/** Tooltip rows at day `x`: the observation, or both forecasts. */
	function describeDay(x: number): string[] | null {
		const tags = [data.holidays.includes(x) && 'bank holiday', data.outages.includes(x) && 'outage'].filter(Boolean);
		if (x < n) {
			if (x < n - 140) return null;
			return [`day ${x}${tags.length ? ` (${tags.join(', ')})` : ''}`, `${Math.round(data.values[x])} users`];
		}
		const h = x - n;
		if (h >= HORIZON) return null;
		const w = result.with;
		return [
			`day ${x}${h === UPCOMING ? ' (bank holiday)' : ''}`,
			`with regressors: ${Math.round(w.mean[h])} [${Math.round(w.lower[h])}, ${Math.round(w.upper[h])}]`,
			`without: ${Math.round(result.without.mean[h])}`
		];
	}
</script>

<section aria-labelledby="calendar-title">
	<h2 id="calendar-title">Tell the model about holidays and outages</h2>
	<p class="lede">
		Daily active users dip on bank holidays and during server outages. Two regressors mark those days, and the model
		becomes regression with ARIMA errors (SARIMAX) on {result.label}. It learns how many users each interruption
		costs, and its weekly pattern stays clean. For the forecast, the calendar already knows that day {UPCOMING + 1}
		ahead is a bank holiday, and no outage is planned. Without the regressors, the dips count as noise: the intervals
		widen, and the holiday comes as a surprise.
	</p>

	<div class="controls">
		<button type="button" onclick={() => (seed += 1)}>New noise</button>
	</div>

	<dl class="stats">
		<div>
			<dt>Holiday effect</dt>
			<dd>{result.effects.holiday.toFixed(0)} (true {data.effects.holiday})</dd>
		</div>
		<div>
			<dt>Outage effect</dt>
			<dd>{result.effects.outage.toFixed(0)} (true {data.effects.outage})</dd>
		</div>
		<div>
			<dt>AICc with / without</dt>
			<dd>{result.with.aicc.toFixed(0)} / {result.without.aicc.toFixed(0)}</dd>
		</div>
		<div>
			<dt>Noise SD with / without</dt>
			<dd>{result.with.sigma.toFixed(1)} / {result.without.sigma.toFixed(1)}</dd>
		</div>
		<div>
			<dt>Both fits</dt>
			<dd>{result.fitMillis.toFixed(0)} ms</dd>
		</div>
	</dl>

	<Chart spec={result.chart} describe={describeDay} />
</section>

<script lang="ts">
	// Fit a model (by auto.arima or by hand) to an example or pasted series, and forecast
	// it with 80% and 95% prediction intervals.
	import Chart from './Chart.svelte';
	import { airline, cycle, parseSeries, randomWalk } from '#lib/data.ts';
	import { forecastScenario, type Api, type ForecastResult, type Orders } from '#lib/scenarios.ts';

	let { api }: { api: Api } = $props();

	/** The example series, each with the season length that fits it. */
	const examples = {
		airline: { name: 'Monthly, seasonal (airline model)', values: airline(), period: 12 },
		walk: { name: 'Random walk with drift', values: randomWalk(), period: 0 },
		cycle: { name: 'Quasi-periodic AR(2)', values: cycle(), period: 0 }
	} as const;

	let source = $state<keyof typeof examples | 'paste'>('airline');
	let pasted = $state('');
	let period = $state(12);
	let horizon = $state(36);
	let automatic = $state(true);
	let orders = $state<Orders>({ p: 0, d: 1, q: 1, seasonalP: 0, seasonalD: 1, seasonalQ: 1 });
	let result = $state.raw<ForecastResult | null>(null);
	let error = $state('');

	/** The series the form currently describes, or an error for bad pasted input. */
	function series(): { values: number[] } | { error: string } {
		return source === 'paste' ? parseSeries(pasted) : { values: [...examples[source].values] };
	}

	/** Fits and forecasts the current series. */
	function run() {
		const input = series();
		if ('error' in input) {
			error = input.error;
			return;
		}
		try {
			result = forecastScenario(api, input.values, period, horizon, automatic ? undefined : $state.snapshot(orders));
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	/** Switches the example and its season length, then refits. */
	function choose(key: typeof source) {
		source = key;
		if (key !== 'paste') {
			period = examples[key].period;
			run();
		}
	}

	run();

	/** Tooltip rows at time `x`. */
	function describe(x: number): string[] | null {
		if (!result) return null;
		const layers = result.chart.layers;
		const observed = layers.find((l) => l.label === 'Observed');
		const forecast = layers.find((l) => l.label === 'Forecast');
		const wide = layers.find((l) => l.label === '95% interval');
		if (observed?.kind === 'line' && x < observed.x.length) {
			return [`t = ${x}`, `observed ${fmt(observed.y[x])}`];
		}
		if (forecast?.kind === 'line' && wide?.kind === 'band') {
			const i = forecast.x.indexOf(x);
			if (i > 0) return [`t = ${x} (h = ${i})`, `forecast ${fmt(forecast.y[i])}`, `95%: ${fmt(wide.lower[i])} … ${fmt(wide.upper[i])}`];
		}
		return null;
	}

	/** A value with sensible precision. */
	function fmt(value: number): string {
		return Number.isFinite(value) ? Number(value.toPrecision(5)).toString() : '—';
	}

	const orderFields = [
		['p', 'p'],
		['d', 'd'],
		['q', 'q'],
		['seasonalP', 'P'],
		['seasonalD', 'D'],
		['seasonalQ', 'Q']
	] as const;
</script>

<section aria-labelledby="forecast-title">
	<h2 id="forecast-title">Fit and forecast</h2>
	<p class="lede">
		Exact maximum likelihood with R's <code>auto.arima</code> search, then forecasts with prediction intervals. The
		fit runs in your browser, in the same Rust code that runs on a microcontroller.
	</p>

	<form
		class="controls"
		onsubmit={(event) => {
			event.preventDefault();
			run();
		}}
	>
		<label>
			Series
			<select value={source} onchange={(event) => choose(event.currentTarget.value as typeof source)}>
				{#each Object.entries(examples) as [key, example] (key)}
					<option value={key}>{example.name}</option>
				{/each}
				<option value="paste">Paste your own…</option>
			</select>
		</label>
		<label>
			Season length
			<input type="number" min="0" max="366" bind:value={period} />
		</label>
		<label>
			Horizon
			<input type="number" min="1" max="500" bind:value={horizon} />
		</label>
		<fieldset>
			<legend>Model</legend>
			<label class="inline"><input type="radio" bind:group={automatic} value={true} /> auto.arima</label>
			<label class="inline"><input type="radio" bind:group={automatic} value={false} /> by hand</label>
		</fieldset>
		{#if !automatic}
			<fieldset class="orders">
				<legend>Orders (p, d, q)(P, D, Q)</legend>
				{#each orderFields as [key, name] (key)}
					<label class="inline">
						{name}
						<input type="number" min="0" max={key.endsWith('d') || key.endsWith('D') ? 2 : 8} bind:value={orders[key]} />
					</label>
				{/each}
			</fieldset>
		{/if}
		{#if source === 'paste'}
			<label class="wide">
				Values: one per line, or <code>date,value</code> rows; blank or <code>NA</code> is missing
				<textarea rows="5" bind:value={pasted} placeholder="112&#10;118&#10;132&#10;…"></textarea>
			</label>
		{/if}
		<button type="submit">Fit and forecast</button>
	</form>

	{#if error}
		<p class="error" role="alert">{error}</p>
	{/if}

	{#if result}
		<dl class="stats">
			<div><dt>Model</dt><dd>{result.label}</dd></div>
			<div><dt>AICc</dt><dd>{fmt(result.aicc)}</dd></div>
			<div><dt>σ²</dt><dd>{fmt(result.sigma2)}</dd></div>
			<div><dt>Fit time</dt><dd>{result.fitMillis.toFixed(0)} ms</dd></div>
		</dl>
		<Chart spec={result.chart} {describe} />
	{/if}
</section>

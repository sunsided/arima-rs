<script lang="ts">
	// Fit a model (by auto.arima or by hand) to an example or pasted series, and forecast
	// it with 80% and 95% prediction intervals. Fits run in a worker: some searches take
	// seconds, and the page must stay responsive (and cancellable) meanwhile.
	import { onDestroy } from 'svelte';
	import Chart from './Chart.svelte';
	import { airline, cycle, parseSeries, randomWalk } from '#lib/data.ts';
	import type { FitRequest, FitResponse } from '#lib/fit.worker.ts';
	import type { ForecastResult, Orders } from '#lib/scenarios.ts';

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
	/** When the fit in flight started, or null when idle. */
	let busySince = $state<number | null>(null);
	/** Seconds the fit in flight has taken so far, ticking while busy. */
	let elapsed = $state(0);

	let worker: Worker | null = null;
	let latest = 0;
	let ticker = 0;

	/** A fresh worker that reports to this panel; a worker that fails ends the fit with an error. */
	function spawn(): Worker {
		const next = new Worker(new URL('../fit.worker.ts', import.meta.url), { type: 'module' });
		next.onerror = (event) => {
			event.preventDefault();
			error = `the fitting worker failed${event.message ? `: ${event.message}` : ''}`;
			worker = null;
			idle();
		};
		next.onmessage = ({ data }: MessageEvent<FitResponse>) => {
			// A cancelled request can still answer; only the latest one counts.
			if (data.id !== latest) return;
			if ('error' in data) error = data.error;
			else {
				result = data.result;
				error = '';
			}
			idle();
		};
		return next;
	}

	/** Marks the panel idle. */
	function idle() {
		busySince = null;
		clearInterval(ticker);
	}

	/** The series the form currently describes, or an error for bad pasted input. */
	function series(): { values: number[] } | { error: string } {
		return source === 'paste' ? parseSeries(pasted) : { values: [...examples[source].values] };
	}

	/** Fits and forecasts the current series in the worker, cancelling any fit in flight. */
	function run() {
		const input = series();
		if ('error' in input) {
			error = input.error;
			return;
		}

		// Wasm cannot be interrupted, so cancelling a running fit means a new worker.
		if (busySince !== null || !worker) {
			worker?.terminate();
			try {
				worker = spawn();
			} catch (e) {
				// The old worker is gone, so a fit in flight ends here too.
				worker = null;
				latest++;
				idle();
				error = `cannot start a worker: ${e instanceof Error ? e.message : String(e)}`;
				return;
			}
		}

		const request: FitRequest = {
			id: ++latest,
			values: input.values,
			period,
			horizon,
			orders: automatic ? undefined : $state.snapshot(orders)
		};
		worker.postMessage(request);
		busySince = performance.now();
		elapsed = 0;
		clearInterval(ticker);
		ticker = window.setInterval(() => {
			if (busySince !== null) elapsed = (performance.now() - busySince) / 1000;
		}, 100);
	}

	/** Stops the fit in flight, keeping the last result. */
	function cancel() {
		worker?.terminate();
		worker = null;
		latest++;
		idle();
	}

	onDestroy(() => {
		worker?.terminate();
		clearInterval(ticker);
	});

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
		<button type="submit">{busySince === null ? 'Fit and forecast' : 'Restart'}</button>
		{#if busySince !== null}
			<button type="button" class="secondary" onclick={cancel}>Cancel</button>
			<span class="busy" role="status">Fitting… {elapsed.toFixed(1)} s</span>
		{/if}
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
		<div class={['result', { stale: busySince !== null }]}>
			<Chart spec={result.chart} {describe} />
		</div>
	{/if}
</section>

<style>
	.busy {
		align-self: center;
		color: var(--text-muted);
		font-variant-numeric: tabular-nums;
	}

	.secondary {
		background: transparent !important;
		color: var(--accent) !important;
	}

	.stale {
		opacity: 0.45;
		transition: opacity 0.2s;
	}
</style>

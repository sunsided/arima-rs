<script lang="ts">
	// Runs the device loop of examples/device_refit.rs: a deployed AR(1) streams a process
	// whose coefficient changes, tests every window for leftover autocorrelation and
	// refits itself when the test fails.
	import Chart from './Chart.svelte';
	import { regimeChange } from '#lib/data.ts';
	import { DRIFT_ALPHA, driftCharts, driftScenario, type Api } from '#lib/scenarios.ts';

	let { api }: { api: Api } = $props();

	const LENGTH = 1000;
	const CHANGE = 500;
	const WINDOW = 100;
	const BEFORE = 0.2;

	let after = $state(0.85);
	let seed = $state(11);

	const values = $derived(regimeChange(LENGTH, CHANGE, BEFORE, after, seed));
	const results = $derived(driftScenario(api, values, WINDOW, BEFORE));
	const charts = $derived(driftCharts(values, results, { change: CHANGE, before: BEFORE, after }));

	/** Tooltip rows at time `x` for the coefficient chart. */
	function describeWindow(x: number): string[] | null {
		const window = results.find((r) => x >= r.start && x < r.end);
		if (!window) return null;
		return [
			`window ${window.start}–${window.end}`,
			`Ljung–Box p = ${window.pValue.toExponential(1)}`,
			window.refit ? `refit: φ = ${window.ar.toFixed(3)}` : `φ = ${window.ar.toFixed(3)}`
		];
	}
</script>

<section aria-labelledby="drift-title">
	<h2 id="drift-title">Notice drift and refit in place</h2>
	<p class="lede">
		A device runs an AR(1) with φ = {BEFORE}. Halfway, the process changes. After every {WINDOW} values the device
		runs Ljung–Box on its standardised innovations; below p = {DRIFT_ALPHA} it refits on that window, warm-started
		from the current model, and swaps the new filter in. The refit is heap-free: a microcontroller runs it in
		a fixed buffer.
	</p>

	<div class="controls">
		<label>
			φ after the change: {after.toFixed(2)}
			<input type="range" min="-0.9" max="0.95" step="0.05" bind:value={after} />
		</label>
		<button type="button" onclick={() => (seed += 1)}>New noise</button>
	</div>

	<Chart spec={charts.series} describe={describeWindow} />
	<Chart spec={charts.coefficient} describe={describeWindow} />
</section>

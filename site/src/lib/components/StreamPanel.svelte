<script lang="ts">
	// Streams a live sensor through the heap-free filter one value at a time and flags
	// values whose innovation leaves the 4-sigma band. A button injects a spike.
	import type { Model, Stream } from '@sunsided/backshift';
	import { onDestroy, untrack } from 'svelte';
	import Chart from './Chart.svelte';
	import { normal, sensor } from '#lib/data.ts';
	import { ALARM_SIGMAS, streamChart, type Api, type StreamPoint } from '#lib/scenarios.ts';

	let { api }: { api: Api } = $props();

	/** How many recent values the chart shows. */
	const VISIBLE = 300;
	/** Values streamed per animation frame. */
	const PER_FRAME = 2;

	// The model is fitted once on quiet history; the filter then runs on its own. The
	// panel is created once the package has loaded, so `api` never changes.
	const backshift = untrack(() => api);
	const training = sensor(400, 21, []);
	const model: Model = backshift.Model.fit(Float64Array.from(training), 1, 0, 0, 0, 0, 0, 0, true);
	const stream: Stream = new backshift.Stream(model);
	stream.warmUp(Float64Array.from(training));
	const noise = normal(99);
	let level = 0;
	let t = training.length;
	let pendingSpike = 0;
	let frame = 0;

	let points = $state.raw<StreamPoint[]>([]);
	let playing = $state(false);
	let anomalies = $state(0);
	let streamed = $state(0);

	/** The next sensor value: AR(1) noise around 20, plus any injected spike. */
	function nextValue(): number {
		level = 0.8 * level + 0.5 * noise();
		const value = 20 + level + pendingSpike;
		pendingSpike = 0;
		return value;
	}

	/** Streams `count` values and appends their scores. */
	function step(count: number) {
		const next = points.slice(-(VISIBLE - count));
		for (let i = 0; i < count; i++) {
			const value = nextValue();
			const [innovation, variance] = stream.update(value) ?? [Number.NaN, Number.NaN];
			const bound = ALARM_SIGMAS * Math.sqrt(variance);
			const alarm = Math.abs(innovation) > bound;
			const anomaly = alarm && !(next[next.length - 1]?.alarm ?? false);
			if (anomaly) anomalies += 1;
			next.push({ t: t++, value, innovation, bound, alarm, anomaly });
		}
		streamed += count;
		points = next;
	}

	/** The animation loop while playing. */
	function tick() {
		step(PER_FRAME);
		frame = requestAnimationFrame(tick);
	}

	function toggle() {
		playing = !playing;
		if (playing) frame = requestAnimationFrame(tick);
		else cancelAnimationFrame(frame);
	}

	step(VISIBLE);

	/**
	 * The cost of one update, timed over a batch on a separate filter: browsers coarsen
	 * `performance.now()` far below the length of a single update.
	 */
	const microsPerUpdate = (() => {
		const BATCH = 20_000;
		const bench = new backshift.Stream(model);
		const inputs = Array.from({ length: BATCH }, () => 20 + noise());
		const started = performance.now();
		for (const value of inputs) bench.update(value);
		const micros = ((performance.now() - started) * 1000) / BATCH;
		bench.free();
		return micros;
	})();

	onDestroy(() => {
		cancelAnimationFrame(frame);
		stream.free();
		model.free();
	});

	const lastT = $derived(points.length ? points[points.length - 1].t : 0);
	const spec = $derived(streamChart(points, { width: 760, height: 260 }, [lastT - VISIBLE + 1, lastT]));

	/** Tooltip rows at time `x`. */
	function describe(x: number): string[] | null {
		const point = points.find((p: StreamPoint) => p.t === x);
		if (!point) return null;
		return [
			`t = ${x}${point.anomaly ? ' · anomaly' : ''}`,
			`value ${point.value.toFixed(2)}`,
			`innovation ${point.innovation.toFixed(2)} (±${point.bound.toFixed(2)})`
		];
	}
</script>

<section aria-labelledby="stream-title">
	<h2 id="stream-title">Stream and score</h2>
	<p class="lede">
		The filter takes one value at a time in fixed memory, exactly as on a device, and returns its one-step
		prediction error with that error's variance: a ready-made anomaly score. A value outside ±{ALARM_SIGMAS}σ is
		flagged; the value after a spike also errs, because the spike threw off its prediction, so only the first of
		consecutive alarms counts.
	</p>

	<div class="controls">
		<button type="button" onclick={toggle}>{playing ? 'Pause' : 'Play'}</button>
		<button type="button" onclick={() => (pendingSpike += 4.5)}>Inject a spike</button>
		<button type="button" onclick={() => step(1)} disabled={playing}>Step</button>
	</div>

	<dl class="stats">
		<div><dt>Model</dt><dd>{model.label()}</dd></div>
		<div><dt>Values streamed</dt><dd>{streamed}</dd></div>
		<div><dt>Anomalies</dt><dd>{anomalies}</dd></div>
		<div><dt>Per update</dt><dd>{microsPerUpdate.toFixed(2)} µs</dd></div>
	</dl>
	<Chart {spec} {describe} />
</section>

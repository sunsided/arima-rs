<script lang="ts">
	// Shows a chart from `renderChart` and adds a hover layer: a crosshair at the nearest
	// x and a tooltip whose rows come from `describe`.
	import { renderChart, type ChartSpec } from '#lib/chart.ts';

	let {
		spec,
		describe
	}: {
		/** The chart to draw. */
		spec: ChartSpec;
		/** The tooltip rows at an x value, or null for no tooltip there. */
		describe?: (x: number) => string[] | null;
	} = $props();

	/** The width the chart is shown at, so text keeps its size on narrow screens. */
	let shownWidth = $state(0);
	/** The spec at the shown width (at least 320 wide), drawn once the width is known. */
	const sized = $derived({ ...spec, width: shownWidth ? Math.max(320, Math.round(shownWidth)) : spec.width });
	const rendered = $derived(renderChart(sized));
	/** The hovered x in data units and its position in percent of the chart width. */
	let hover = $state<{ x: number; left: number; rows: string[] } | null>(null);

	/** Maps the pointer to the nearest whole x and asks `describe` for its rows. */
	function move(event: PointerEvent) {
		if (!describe) return;
		const box = (event.currentTarget as HTMLElement).getBoundingClientRect();
		const px = ((event.clientX - box.left) / box.width) * sized.width;
		const { plot } = rendered.scales;
		if (px < plot.left || px > plot.right) {
			hover = null;
			return;
		}
		const x = Math.round(rendered.scales.invertX(px));
		const rows = describe(x);
		hover = rows ? { x, left: (rendered.scales.x(x) / sized.width) * 100, rows } : null;
	}
</script>

<figure class="chart" bind:clientWidth={shownWidth} onpointermove={move} onpointerleave={() => (hover = null)}>
	<!-- renderChart escapes every text node and writes only numbers into attributes. -->
	{@html rendered.svg}
	{#if hover}
		<div
			class="crosshair"
			style:left="{hover.left}%"
			style:top="{(rendered.scales.plot.top / sized.height) * 100}%"
			style:bottom="{100 - (rendered.scales.plot.bottom / sized.height) * 100}%"
		></div>
		<div class={['tooltip', { flip: hover.left > 60 }]} style:left="{hover.left}%">
			{#each hover.rows as row (row)}
				<div>{row}</div>
			{/each}
		</div>
	{/if}
</figure>

<style>
	.chart {
		position: relative;
		margin: 0;
		touch-action: pan-y;
	}

	.chart :global(svg) {
		display: block;
		width: 100%;
		height: auto;
	}

	.crosshair {
		position: absolute;
		width: 1px;
		background: var(--text-muted);
		pointer-events: none;
	}

	.tooltip {
		position: absolute;
		top: 2.4rem;
		transform: translateX(12px);
		padding: 0.4rem 0.6rem;
		border-radius: 6px;
		background: var(--surface-raised);
		border: 1px solid var(--border);
		box-shadow: 0 2px 8px rgb(0 0 0 / 0.12);
		color: var(--text-primary);
		font-size: 0.8rem;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
		pointer-events: none;
	}

	.tooltip.flip {
		transform: translateX(calc(-100% - 12px));
	}
</style>

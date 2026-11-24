// Renders time-series charts as standalone SVG strings, without touching the DOM, so the
// site and the README image script (Node) draw the same charts. Colours are roles that an
// embedded <style> resolves for light and dark mode, following the dataviz reference
// palette: blue for observations, orange for forecasts, a blue ramp for intervals and
// the critical red, with a marker and a label, for anomalies.

/** A colour role; the embedded stylesheet maps each to a light and a dark value. */
export type Role = 'data' | 'forecast' | 'band-outer' | 'band-inner' | 'critical' | 'muted' | 'refit';

/** One layer of a chart, drawn in the order given. */
export type Layer =
	| { kind: 'line'; x: number[]; y: number[]; role: Role; label: string; dashed?: boolean }
	| { kind: 'band'; x: number[]; lower: number[]; upper: number[]; role: Role; label: string }
	| { kind: 'points'; x: number[]; y: number[]; role: Role; label: string }
	| { kind: 'step'; x: number[]; y: number[]; role: Role; label: string; dashed?: boolean };

/** A shaded span of the x axis, such as a window that triggered a refit. */
export interface Region {
	x0: number;
	x1: number;
	role: Role;
}

/** Everything a chart shows. */
export interface ChartSpec {
	width: number;
	height: number;
	layers: Layer[];
	/** Shaded spans of x, drawn under the layers. */
	regions?: Region[];
	/** Horizontal reference lines, such as zero. */
	rules?: { y: number; role: Role; label?: string }[];
	/** Vertical reference lines, such as the forecast origin. */
	markers?: { x: number; label: string }[];
	/** The x domain; defaults to the extent of the layers. */
	xDomain?: [number, number];
	/** The y domain; defaults to the padded extent of the layers. */
	yDomain?: [number, number];
	/** Labels the axes. */
	xLabel?: string;
	yLabel?: string;
	/** An accessible title, rendered as <title>. */
	title: string;
	/** Draws the legend above the plot; on by default for two or more labelled layers. */
	legend?: boolean;
}

/** The scales of a rendered chart, for hover layers drawn on top of it. */
export interface Scales {
	x: (value: number) => number;
	y: (value: number) => number;
	invertX: (pixel: number) => number;
	plot: { left: number; top: number; right: number; bottom: number };
}

const MARGIN = { top: 14, right: 16, bottom: 34, left: 52 };

/** The height of one legend row. */
const LEGEND_ROW = 20;

/** Light values per role, from the reference palette. */
const LIGHT = `
.bs-surface { fill: #fcfcfb; }
.bs-text { fill: #52514e; font: 12px system-ui, -apple-system, 'Segoe UI', sans-serif; }
.bs-title { fill: #0b0b0b; }
.bs-grid { stroke: #e5e4e0; stroke-width: 1; }
.bs-axis { stroke: #a9a8a2; stroke-width: 1; }
.bs-data { stroke: #2a78d6; fill: #2a78d6; }
.bs-forecast { stroke: #eb6834; fill: #eb6834; }
.bs-band-outer { fill: #cde2fb; stroke: none; }
.bs-band-inner { fill: #9ec5f4; stroke: none; }
.bs-critical { stroke: #d03b3b; fill: #d03b3b; }
.bs-muted { stroke: #a9a8a2; fill: #e5e4e0; }
.bs-refit { stroke: #6250d6; fill: #6250d6; }
.bs-region-critical { fill: #d03b3b; fill-opacity: 0.08; stroke: none; }
.bs-region-muted { fill: #a9a8a2; fill-opacity: 0.1; stroke: none; }
.bs-ring { stroke: #fcfcfb; }`;

/** Dark values per role: the same hues, stepped for the dark surface. */
const DARK = `
.bs-surface { fill: #1a1a19; }
.bs-text { fill: #c3c2b7; font: 12px system-ui, -apple-system, 'Segoe UI', sans-serif; }
.bs-title { fill: #f0efec; }
.bs-grid { stroke: #2e2e2c; stroke-width: 1; }
.bs-axis { stroke: #5c5b55; stroke-width: 1; }
.bs-data { stroke: #3987e5; fill: #3987e5; }
.bs-forecast { stroke: #d95926; fill: #d95926; }
.bs-band-outer { fill: #104281; stroke: none; }
.bs-band-inner { fill: #1c5cab; stroke: none; }
.bs-critical { stroke: #e66767; fill: #e66767; }
.bs-muted { stroke: #5c5b55; fill: #2e2e2c; }
.bs-refit { stroke: #9085e9; fill: #9085e9; }
.bs-region-critical { fill: #e66767; fill-opacity: 0.12; stroke: none; }
.bs-region-muted { fill: #5c5b55; fill-opacity: 0.2; stroke: none; }
.bs-ring { stroke: #1a1a19; }`;

/** Lines are never filled; this comes last so no role's `fill` overrides it. */
const LAST = `.bs-line { fill: none; }`;

/**
 * The stylesheet for a theme. `auto` follows `prefers-color-scheme`, which suits the site;
 * the README needs fixed files, because GitHub picks the image by its own theme.
 */
function style(theme: 'auto' | 'light' | 'dark'): string {
	if (theme === 'light') return LIGHT + LAST;
	if (theme === 'dark') return DARK + LAST;
	return `${LIGHT}\n@media (prefers-color-scheme: dark) {${DARK}\n}\n${LAST}`;
}

/** Renders `spec` as an SVG string and returns it with the scales it used. */
export function renderChart(
	spec: ChartSpec,
	theme: 'auto' | 'light' | 'dark' = 'auto'
): { svg: string; scales: Scales } {
	const labelled = spec.layers.filter((layer) => layer.label);
	const showLegend = spec.legend ?? labelled.length >= 2;
	const legendItems = showLegend ? layoutLegend(labelled, MARGIN.left, spec.width - MARGIN.right) : [];
	const legendRows = legendItems.reduce((rows, item) => Math.max(rows, item.row + 1), 0);
	// Room for the legend rows, or for the markers' labels without a legend.
	const top = MARGIN.top + Math.max(legendRows * LEGEND_ROW, spec.markers?.length ? 12 : 0) + 8;
	const plot = {
		left: MARGIN.left,
		top,
		right: spec.width - MARGIN.right,
		bottom: spec.height - MARGIN.bottom
	};
	const [x0, x1] = spec.xDomain ?? extent(spec.layers.flatMap(xValues));
	const [y0, y1] = spec.yDomain ?? padded(extent(spec.layers.flatMap(yValues)));
	const yTicks = niceTicks(y0, y1, 5);
	const [yLow, yHigh] = [Math.min(y0, yTicks[0]), Math.max(y1, yTicks[yTicks.length - 1])];
	const x = linear(x0, x1, plot.left, plot.right);
	const y = linear(yLow, yHigh, plot.bottom, plot.top);
	const scales: Scales = { x, y, invertX: linear(plot.left, plot.right, x0, x1), plot };

	const parts: string[] = [];
	parts.push(`<rect class="bs-surface" width="${spec.width}" height="${spec.height}" rx="8"/>`);
	for (const region of spec.regions ?? []) {
		const left = x(Math.max(region.x0, x0));
		const right = x(Math.min(region.x1, x1));
		parts.push(
			`<rect class="bs-region-${region.role}" x="${f(left)}" y="${plot.top}" width="${f(right - left)}" height="${plot.bottom - plot.top}"/>`
		);
	}

	for (const tick of yTicks) {
		const py = f(y(tick));
		parts.push(`<line class="bs-grid" x1="${plot.left}" x2="${plot.right}" y1="${py}" y2="${py}"/>`);
		parts.push(`<text class="bs-text" x="${plot.left - 8}" y="${py}" dy="0.32em" text-anchor="end">${formatTick(tick)}</text>`);
	}

	for (const tick of niceTicks(x0, x1, Math.max(2, Math.round((plot.right - plot.left) / 90)))) {
		if (tick < x0 || tick > x1) continue;
		parts.push(`<text class="bs-text" x="${f(x(tick))}" y="${plot.bottom + 18}" text-anchor="middle">${formatTick(tick)}</text>`);
	}

	parts.push(`<line class="bs-axis" x1="${plot.left}" x2="${plot.right}" y1="${plot.bottom}" y2="${plot.bottom}"/>`);
	for (const rule of spec.rules ?? []) {
		const py = f(y(rule.y));
		parts.push(`<line class="bs-${rule.role} bs-line" x1="${plot.left}" x2="${plot.right}" y1="${py}" y2="${py}" stroke-width="1" stroke-dasharray="4 3"/>`);
		if (rule.label) {
			parts.push(`<text class="bs-text" x="${plot.right - 4}" y="${py}" dy="-0.4em" text-anchor="end">${escape(rule.label)}</text>`);
		}
	}

	for (const layer of spec.layers) parts.push(drawLayer(layer, x, y));
	for (const marker of spec.markers ?? []) {
		const px = f(x(marker.x));
		parts.push(`<line class="bs-axis" x1="${px}" x2="${px}" y1="${plot.top}" y2="${plot.bottom}" stroke-dasharray="2 3"/>`);
		parts.push(`<text class="bs-text" x="${px}" y="${plot.top - 4}" text-anchor="middle">${escape(marker.label)}</text>`);
	}

	parts.push(...legendItems.map((item) => drawLegendItem(item)));
	if (spec.yLabel) {
		parts.push(`<text class="bs-text" x="12" y="${(plot.top + plot.bottom) / 2}" transform="rotate(-90 12 ${(plot.top + plot.bottom) / 2})" text-anchor="middle">${escape(spec.yLabel)}</text>`);
	}

	if (spec.xLabel) {
		parts.push(`<text class="bs-text" x="${plot.right}" y="${spec.height - 4}" text-anchor="end">${escape(spec.xLabel)}</text>`);
	}

	const svg =
		`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${spec.width} ${spec.height}" width="${spec.width}" height="${spec.height}" role="img">` +
		`<title>${escape(spec.title)}</title><style>${style(theme)}</style>${parts.join('')}</svg>`;
	return { svg, scales };
}

/** Draws one layer; gaps (`NaN`) split lines and bands into pieces. */
function drawLayer(layer: Layer, x: (v: number) => number, y: (v: number) => number): string {
	switch (layer.kind) {
		case 'line':
		case 'step': {
			const dash = layer.dashed ? ' stroke-dasharray="5 4"' : '';
			return segments(layer.x, layer.y)
				.map((run) => {
					const points = layer.kind === 'step' ? stepped(run) : run;
					const d = points.map(([px, py], i) => `${i ? 'L' : 'M'}${f(x(px))},${f(y(py))}`).join('');
					return `<path class="bs-${layer.role} bs-line" d="${d}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"${dash}/>`;
				})
				.join('');
		}
		case 'band': {
			const out: string[] = [];
			let run: number[] = [];
			const flush = () => {
				if (run.length > 1) {
					const top = run.map((i, k) => `${k ? 'L' : 'M'}${f(x(layer.x[i]))},${f(y(layer.upper[i]))}`).join('');
					const bottom = [...run].reverse().map((i) => `L${f(x(layer.x[i]))},${f(y(layer.lower[i]))}`).join('');
					out.push(`<path class="bs-${layer.role}" d="${top}${bottom}Z"/>`);
				}
				run = [];
			};
			layer.x.forEach((_, i) => {
				if (Number.isFinite(layer.lower[i]) && Number.isFinite(layer.upper[i])) run.push(i);
				else flush();
			});
			flush();
			return out.join('');
		}
		case 'points':
			return layer.x
				.map((px, i) =>
					Number.isFinite(layer.y[i])
						? // The surface-coloured ring (.bs-ring overrides the role's stroke) separates
							// a marker from the line beneath it.
							`<circle class="bs-${layer.role} bs-ring" cx="${f(x(px))}" cy="${f(y(layer.y[i]))}" r="4.5" stroke-width="2"/>`
						: ''
				)
				.join('');
	}
}

/** A legend entry placed at `x` in row `row`. */
interface LegendItem {
	layer: Layer;
	x: number;
	row: number;
}

/** Places a swatch and a name per labelled layer in rows that wrap at `right`. */
function layoutLegend(layers: Layer[], left: number, right: number): LegendItem[] {
	let [cursor, row] = [left, 0];
	return layers.map((layer) => {
		const width = 20 + layer.label.length * 6.6;
		if (cursor > left && cursor + width > right) {
			[cursor, row] = [left, row + 1];
		}
		const item = { layer, x: cursor, row };
		cursor += width + 18;
		return item;
	});
}

/** Draws one legend entry. */
function drawLegendItem({ layer, x, row }: LegendItem): string {
	const cy = MARGIN.top + 3 + row * LEGEND_ROW;
	const swatch =
		layer.kind === 'band'
			? `<rect class="bs-${layer.role}" x="${x}" y="${cy - 5}" width="14" height="10" rx="2"/>`
			: layer.kind === 'points'
				? `<circle class="bs-${layer.role}" cx="${x + 7}" cy="${cy}" r="4.5"/>`
				: `<line class="bs-${layer.role} bs-line" x1="${x}" x2="${x + 14}" y1="${cy}" y2="${cy}" stroke-width="2"${'dashed' in layer && layer.dashed ? ' stroke-dasharray="4 3"' : ''}/>`;
	return `${swatch}<text class="bs-text bs-title" x="${x + 20}" y="${cy}" dy="0.32em">${escape(layer.label)}</text>`;
}

/** Runs of consecutive finite points as `[x, y]` pairs. */
function segments(xs: number[], ys: number[]): [number, number][][] {
	const runs: [number, number][][] = [];
	let run: [number, number][] = [];
	xs.forEach((px, i) => {
		if (Number.isFinite(ys[i]) && Number.isFinite(px)) run.push([px, ys[i]]);
		else if (run.length) {
			runs.push(run);
			run = [];
		}
	});
	if (run.length) runs.push(run);
	return runs;
}

/** A step function through `points`: each value holds until the next x. */
function stepped(points: [number, number][]): [number, number][] {
	return points.flatMap(([px, py], i) => (i ? [[px, points[i - 1][1]], [px, py]] : [[px, py]])) as [number, number][];
}

/** The x values a layer spans. */
function xValues(layer: Layer): number[] {
	return layer.x;
}

/** The y values a layer spans. */
function yValues(layer: Layer): number[] {
	return layer.kind === 'band' ? [...layer.lower, ...layer.upper] : layer.y;
}

/** The finite minimum and maximum, or `[0, 1]` without finite values. */
function extent(values: number[]): [number, number] {
	let [low, high] = [Infinity, -Infinity];
	for (const v of values) {
		if (Number.isFinite(v)) {
			low = Math.min(low, v);
			high = Math.max(high, v);
		}
	}
	if (low > high) return [0, 1];
	if (low !== high) return [low, high];
	// Widen relative to the value too: at 1e17, `low - 1` is `low` again.
	const pad = Math.max(1, Math.abs(low) * 1e-3);
	return [low - pad, high + pad];
}

/** The extent widened by 5% on each side. */
function padded([low, high]: [number, number]): [number, number] {
	const pad = (high - low) * 0.05;
	return [low - pad, high + pad];
}

/** A linear map from `[d0, d1]` to `[r0, r1]`. */
export function linear(d0: number, d1: number, r0: number, r1: number): (v: number) => number {
	const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
	return (v) => r0 + (v - d0) * k;
}

/** About `count` round tick values covering `[low, high]`. */
export function niceTicks(low: number, high: number, count: number): number[] {
	const span = high - low || 1;
	const raw = span / count;
	const power = 10 ** Math.floor(Math.log10(raw));
	const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? 10 * power;
	const ticks: number[] = [];
	// The bound stops the loop where `t += step` no longer moves `t` (large values, tiny span).
	for (let t = Math.floor(low / step) * step; t <= high + step * 1e-9 && ticks.length <= 4 * count + 2; t += step) {
		ticks.push(Math.round(t / step) * step);
	}
	return ticks;
}

/** A tick label: compact, without float noise. */
function formatTick(value: number): string {
	const abs = Math.abs(value);
	if (abs >= 10000) return `${(value / 1000).toFixed(0)}k`;
	return Number(value.toPrecision(4)).toString();
}

/** A coordinate with one decimal. */
function f(value: number): string {
	return (Math.round(value * 10) / 10).toString();
}

/** Escapes text for SVG content. */
function escape(text: string): string {
	return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

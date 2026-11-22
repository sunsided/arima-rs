// Renders the README's charts from the same scenarios and seeds as the demo site, in a
// light and a dark variant (GitHub picks one by the viewer's theme through <picture>).
// The data is seeded and nothing time-dependent is drawn, so a rerun writes identical
// files. Run after building the package:
//     yarn readme-charts
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import * as backshift from '@sunsided/backshift';
import { renderChart, type ChartSpec } from '../src/lib/chart.ts';
import { airline, regimeChange, sensor } from '../src/lib/data.ts';
import { driftCharts, driftScenario, forecastScenario, streamChart, streamScenario } from '../src/lib/scenarios.ts';

const wasm = createRequire(import.meta.url).resolve('@sunsided/backshift/backshift_bg.wasm');
backshift.initSync({ module: readFileSync(wasm) });
const out = fileURLToPath(new URL('../../docs/img/', import.meta.url));

/** Writes `spec` as `<name>-light.svg` and `<name>-dark.svg`. */
function write(name: string, spec: ChartSpec) {
	for (const theme of ['light', 'dark'] as const) {
		writeFileSync(`${out}${name}-${theme}.svg`, `${renderChart(spec, theme).svg}\n`);
	}
}

const forecast = forecastScenario(backshift, airline(), 12, 36);
write('forecast', forecast.chart);

const readings = sensor(600, 5, [380, 455, 520]);
const stream = streamScenario(backshift, readings.slice(0, 300), readings.slice(300));
write('stream', streamChart(stream.points));

const truth = { change: 500, before: 0.2, after: 0.85 };
const values = regimeChange(1000, truth.change, truth.before, truth.after);
const drift = driftCharts(values, driftScenario(backshift, values, 100, truth.before), truth);
write('drift', drift.series);
write('drift-coefficient', drift.coefficient);

console.log(`wrote charts for ${forecast.label} and ${stream.label} to ${out}`);

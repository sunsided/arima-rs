// Fits and forecasts off the main thread, so a slow auto.arima search (seconds for some
// season lengths) never freezes the page. One request at a time; the page terminates the
// worker to cancel a fit it no longer needs.
import init, * as backshift from '@sunsided/backshift';
import { forecastScenario, type ForecastResult, type Orders } from './scenarios.ts';

/** A fit request from the page. */
export interface FitRequest {
	id: number;
	values: number[];
	period: number;
	horizon: number;
	orders?: Orders;
}

/** The answer to a request: its result, or the error message. */
export type FitResponse = { id: number; result: ForecastResult } | { id: number; error: string };

const ready = init();

self.onmessage = async ({ data }: MessageEvent<FitRequest>) => {
	let response: FitResponse;
	try {
		await ready;
		const result = forecastScenario(backshift, data.values, data.period, data.horizon, data.orders);
		response = { id: data.id, result };
	} catch (error) {
		response = { id: data.id, error: error instanceof Error ? error.message : String(error) };
	}
	self.postMessage(response);
};

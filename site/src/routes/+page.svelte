<script lang="ts">
	// The demo page: loads the WebAssembly package in the browser, then shows the
	// panels. The page is prerendered; the wasm only runs on the client.
	import { onMount } from 'svelte';
	import CalendarPanel from '#lib/components/CalendarPanel.svelte';
	import DriftPanel from '#lib/components/DriftPanel.svelte';
	import ForecastPanel from '#lib/components/ForecastPanel.svelte';
	import StreamPanel from '#lib/components/StreamPanel.svelte';
	import type { Api } from '#lib/scenarios.ts';

	let api = $state.raw<Api | null>(null);
	let failure = $state('');

	onMount(async () => {
		try {
			const backshift = await import('@sunsided/backshift');
			await backshift.default();
			api = backshift;
		} catch (error) {
			failure = error instanceof Error ? error.message : String(error);
		}
	});
</script>

{#snippet panelError(error: unknown, reset: () => void)}
	<p class="error" role="alert">
		This demo failed: {error instanceof Error ? error.message : String(error)}
		<button type="button" onclick={reset}>Try again</button>
	</p>
{/snippet}

<header>
	<p class="eyebrow">backshift</p>
	<h1>Fit anywhere, forecast on anything</h1>
	<p class="intro">
		Seasonal ARIMA, fitted by exact maximum likelihood and checked against statsmodels to 1e-9, with a streaming
		filter that needs no heap. Everything below runs in your browser through WebAssembly, from the same Rust code that
		runs on a microcontroller.
	</p>
	<nav>
		<a href="https://github.com/sunsided/arima-rs">GitHub</a>
		<a href="https://crates.io/crates/backshift">crates.io</a>
		<a href="https://www.npmjs.com/package/@sunsided/backshift">npm</a>
		<a href="https://docs.rs/backshift">docs.rs</a>
	</nav>
</header>

<main>
	{#if failure}
		<p class="error" role="alert">Could not load the WebAssembly module: {failure}</p>
	{:else if api}
		<!-- A panel whose wasm call throws shows its error instead of freezing. -->
		<svelte:boundary>
			<ForecastPanel />
			{#snippet failed(error, reset)}{@render panelError(error, reset)}{/snippet}
		</svelte:boundary>
		<svelte:boundary>
			<CalendarPanel {api} />
			{#snippet failed(error, reset)}{@render panelError(error, reset)}{/snippet}
		</svelte:boundary>
		<svelte:boundary>
			<StreamPanel {api} />
			{#snippet failed(error, reset)}{@render panelError(error, reset)}{/snippet}
		</svelte:boundary>
		<svelte:boundary>
			<DriftPanel {api} />
			{#snippet failed(error, reset)}{@render panelError(error, reset)}{/snippet}
		</svelte:boundary>
	{:else}
		<p class="loading">Loading the WebAssembly module…</p>
	{/if}
</main>

<footer>
	<p>
		MIT or Apache-2.0. Use it in Rust (<code>cargo add backshift</code>), in JavaScript
		(<code>npm install @sunsided/backshift</code>), or on bare metal with <code>no_std</code>.
	</p>
</footer>

<style>
	header,
	main,
	footer {
		max-width: 52rem;
		margin: 0 auto;
		padding: 0 1rem;
	}

	header {
		padding-top: 3rem;
	}

	.eyebrow {
		margin: 0;
		color: var(--accent);
		font-weight: 600;
		letter-spacing: 0.04em;
	}

	h1 {
		margin: 0.2rem 0 0.8rem;
		font-size: clamp(1.8rem, 5vw, 2.6rem);
		line-height: 1.15;
	}

	.intro {
		color: var(--text-secondary);
		max-width: 44rem;
	}

	nav {
		display: flex;
		flex-wrap: wrap;
		gap: 1.2rem;
		margin: 1rem 0 2rem;
	}

	main :global(section) {
		margin: 0 0 2.5rem;
		padding: 1.5rem;
		border: 1px solid var(--border);
		border-radius: 12px;
		background: var(--surface-raised);
	}

	main :global(h2) {
		margin: 0 0 0.4rem;
		font-size: 1.3rem;
	}

	main :global(.lede) {
		margin: 0 0 1rem;
		color: var(--text-secondary);
	}

	main :global(.controls) {
		display: flex;
		flex-wrap: wrap;
		align-items: end;
		gap: 0.8rem 1rem;
		margin-bottom: 1rem;
	}

	main :global(label) {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		font-size: 0.85rem;
		color: var(--text-secondary);
	}

	main :global(label.inline) {
		flex-direction: row;
		align-items: center;
		gap: 0.35rem;
	}

	main :global(label.wide) {
		flex-basis: 100%;
	}

	main :global(fieldset) {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem 0.8rem;
		margin: 0;
		padding: 0.3rem 0.6rem 0.5rem;
		border: 1px solid var(--border);
		border-radius: 8px;
		font-size: 0.85rem;
		color: var(--text-secondary);
	}

	main :global(input[type='number']) {
		width: 4.5rem;
	}

	main :global(input),
	main :global(select),
	main :global(textarea),
	main :global(button) {
		font: inherit;
		font-size: 0.9rem;
		color: var(--text-primary);
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.35rem 0.5rem;
	}

	main :global(textarea) {
		width: 100%;
		box-sizing: border-box;
		font-family: ui-monospace, Menlo, Consolas, monospace;
	}

	main :global(button) {
		cursor: pointer;
		background: var(--accent);
		border-color: var(--accent);
		color: #fff;
		font-weight: 600;
	}

	main :global(button:disabled) {
		opacity: 0.5;
		cursor: default;
	}

	main :global(.stats) {
		display: flex;
		flex-wrap: wrap;
		gap: 0.6rem 1.8rem;
		margin: 0 0 0.8rem;
	}

	main :global(.stats div) {
		display: flex;
		flex-direction: column;
	}

	main :global(dt) {
		font-size: 0.75rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	main :global(dd) {
		margin: 0;
		font-variant-numeric: tabular-nums;
		font-weight: 600;
	}

	main :global(.error) {
		color: var(--critical);
	}

	.loading {
		color: var(--text-muted);
	}

	footer {
		padding-bottom: 3rem;
		color: var(--text-muted);
		font-size: 0.9rem;
	}
</style>

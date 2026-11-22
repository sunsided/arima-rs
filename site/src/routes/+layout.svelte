<script lang="ts">
	// Page chrome and the colour tokens every component uses, for light and dark mode.
	let { children } = $props();
</script>

<svelte:head>
	<title>backshift: ARIMA in your browser</title>
</svelte:head>

{@render children()}

<style>
	:global(:root) {
		color-scheme: light dark;
		--surface: #f6f5f3;
		--surface-raised: #fcfcfb;
		--border: #e5e4e0;
		--text-primary: #0b0b0b;
		--text-secondary: #52514e;
		--text-muted: #8a8984;
		--accent: #2a78d6;
		--critical: #d03b3b;
	}

	@media (prefers-color-scheme: dark) {
		:global(:root) {
			--surface: #111110;
			--surface-raised: #1a1a19;
			--border: #2e2e2c;
			--text-primary: #f0efec;
			--text-secondary: #c3c2b7;
			--text-muted: #8a8984;
			--accent: #3987e5;
			--critical: #e66767;
		}
	}

	:global(body) {
		margin: 0;
		background: var(--surface);
		color: var(--text-primary);
		font: 16px/1.55 system-ui, -apple-system, 'Segoe UI', sans-serif;
	}

	:global(code) {
		font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
		font-size: 0.9em;
	}

	:global(a) {
		color: var(--accent);
	}
</style>

<!--
  Side-by-side comparison page (/compare).
  Lets users summarize the same article with different modes or personas
  and see the results next to each other. Useful for evaluating how
  pipeline settings affect output quality.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { fetchPersonas, summarize } from '$lib/api';
	import { articles, loadArticles } from '$lib/stores/articles';
	import { userId } from '$lib/stores/user';
	import ConfidenceBadge from '$lib/components/ConfidenceBadge.svelte';
	import FlaggedEntityChips from '$lib/components/FlaggedEntityChips.svelte';
	import { MODES, LENGTHS } from '$lib/types';
	import type { Article, SummaryResult } from '$lib/types';

	let personas: string[] = ['default'];

	// left panel settings
	let modeA = 'extractive';
	let personaA = 'default';
	let lengthA = 'standard';

	// right panel settings
	let modeB = 'hybrid';
	let personaB = 'executive';
	let lengthB = 'standard';

	let selectedArticle: Article | null = null;
	let k = 5;

	let resultA: SummaryResult | null = null;
	let resultB: SummaryResult | null = null;
	let loadingA = false;
	let loadingB = false;
	let latencyA: number | null = null;
	let latencyB: number | null = null;

	onMount(async () => {
		try { personas = await fetchPersonas(); } catch {}
	});

	async function runComparison() {
		if (!selectedArticle) return;

		resultA = null;
		resultB = null;

		// run both summarizations concurrently
		const basePayload = { url: selectedArticle.link, k, user_id: $userId || undefined };

		loadingA = true;
		loadingB = true;

		const t0A = performance.now();
		const t0B = performance.now();

		const [resA, resB] = await Promise.allSettled([
			summarize({ ...basePayload, mode: modeA, persona: personaA, length: lengthA }),
			summarize({ ...basePayload, mode: modeB, persona: personaB, length: lengthB }),
		]);

		if (resA.status === 'fulfilled') {
			resultA = resA.value;
			latencyA = Math.round(performance.now() - t0A);
		}
		loadingA = false;

		if (resB.status === 'fulfilled') {
			resultB = resB.value;
			latencyB = Math.round(performance.now() - t0B);
		}
		loadingB = false;
	}
</script>

<svelte:head>
	<title>Compare - Adaptive Summarizer</title>
</svelte:head>

<h1 class="page-title">Side-by-Side Comparison</h1>

<!-- article selector -->
<div class="card article-selector">
	<label>
		<span class="label-text">Article</span>
		<select bind:value={selectedArticle}>
			<option value={null}>Select an article...</option>
			{#each $articles as art}
				<option value={art}>{art.title}</option>
			{/each}
		</select>
	</label>

	<label class="k-input">
		<span class="label-text">k</span>
		<input type="number" bind:value={k} min={1} max={10} />
	</label>

	<button class="btn" on:click={runComparison} disabled={!selectedArticle || loadingA || loadingB}>
		{loadingA || loadingB ? 'Running...' : 'Compare'}
	</button>
</div>

<!-- two-column comparison -->
<div class="compare-grid">
	<!-- Panel A -->
	<div class="card compare-panel">
		<div class="panel-controls">
			<h3>Configuration A</h3>
			<select bind:value={modeA}>
				{#each MODES as m}<option value={m}>{m}</option>{/each}
			</select>
			<select bind:value={personaA}>
				{#each personas as p}<option value={p}>{p}</option>{/each}
			</select>
			<select bind:value={lengthA}>
				{#each LENGTHS as l}<option value={l}>{l}</option>{/each}
			</select>
		</div>

		<div class="result-area">
			{#if loadingA}
				<div class="skeleton skel-text"></div>
				<div class="skeleton skel-text short"></div>
			{:else if resultA}
				<div class="result-meta">
					<span class="chip">{resultA.mode}</span>
					<span class="chip">{resultA.persona}</span>
					<ConfidenceBadge confidence={resultA.confidence ?? null} />
					{#if latencyA}<span class="latency">{latencyA}ms</span>{/if}
				</div>
				<p class="result-text">{resultA.summary}</p>
				<FlaggedEntityChips entities={resultA.flagged_entities ?? null} />
			{:else}
				<p class="placeholder">Results will appear here.</p>
			{/if}
		</div>
	</div>

	<!-- Panel B -->
	<div class="card compare-panel">
		<div class="panel-controls">
			<h3>Configuration B</h3>
			<select bind:value={modeB}>
				{#each MODES as m}<option value={m}>{m}</option>{/each}
			</select>
			<select bind:value={personaB}>
				{#each personas as p}<option value={p}>{p}</option>{/each}
			</select>
			<select bind:value={lengthB}>
				{#each LENGTHS as l}<option value={l}>{l}</option>{/each}
			</select>
		</div>

		<div class="result-area">
			{#if loadingB}
				<div class="skeleton skel-text"></div>
				<div class="skeleton skel-text short"></div>
			{:else if resultB}
				<div class="result-meta">
					<span class="chip">{resultB.mode}</span>
					<span class="chip">{resultB.persona}</span>
					<ConfidenceBadge confidence={resultB.confidence ?? null} />
					{#if latencyB}<span class="latency">{latencyB}ms</span>{/if}
				</div>
				<p class="result-text">{resultB.summary}</p>
				<FlaggedEntityChips entities={resultB.flagged_entities ?? null} />
			{:else}
				<p class="placeholder">Results will appear here.</p>
			{/if}
		</div>
	</div>
</div>

<style>
	.page-title { font-size: 20px; margin: 0 0 16px; }

	.article-selector {
		display: flex;
		align-items: flex-end;
		gap: 12px;
		margin-bottom: 16px;
		flex-wrap: wrap;
	}
	.article-selector label { flex: 1; min-width: 200px; display: flex; flex-direction: column; gap: 4px; }
	.article-selector select { width: 100%; }
	.k-input { flex: 0; min-width: 70px !important; }
	.label-text { font-size: 12px; color: var(--muted); font-weight: 500; }

	.compare-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}

	.compare-panel { display: flex; flex-direction: column; gap: 12px; }

	.panel-controls {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}
	.panel-controls h3 { margin: 0; font-size: 14px; margin-right: auto; }
	.panel-controls select { font-size: 12px; padding: 4px 8px; }

	.result-area { min-height: 120px; }
	.result-meta { display: flex; gap: 6px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
	.result-text { margin: 0; font-size: 14px; line-height: 1.7; white-space: pre-wrap; }
	.placeholder { color: var(--muted); font-size: 13px; text-align: center; padding: 30px; }

	.skel-text { height: 16px; margin-bottom: 10px; width: 100%; }
	.skel-text.short { width: 65%; }

	.latency { font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }

	@media (max-width: 768px) {
		.compare-grid { grid-template-columns: 1fr; }
	}
</style>

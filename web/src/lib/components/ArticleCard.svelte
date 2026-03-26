<!--
  Single article card.
  Displays the title, link, and (if ranked) relevance score + match reasons.
  Emits a "select" event when clicked so parent components can track selection.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { Article, RankedArticle } from '$lib/types';

	export let article: Article | RankedArticle;
	export let selected = false;

	const dispatch = createEventDispatcher();

	// type guard to check if this is a ranked article with scoring info
	$: isRanked = 'score' in article && (article as RankedArticle).score > 0;
	$: ranked = article as RankedArticle;
</script>

<button
	class="article-card"
	class:selected
	on:click={() => dispatch('select', article)}
>
	<div class="card-header">
		<h3 class="card-title">{article.title}</h3>
		{#if isRanked}
			<span class="score-badge">{(ranked.score * 100).toFixed(0)}%</span>
		{/if}
	</div>

	{#if isRanked && ranked.match_reasons.length > 0}
		<div class="reasons">
			{#each ranked.match_reasons as reason}
				<span class="chip">{reason}</span>
			{/each}
		</div>
	{/if}

	<a
		href={article.link}
		target="_blank"
		rel="noopener"
		class="article-link"
		on:click|stopPropagation
	>
		Open article ↗
	</a>
</button>

<style>
	.article-card {
		display: block;
		width: 100%;
		text-align: left;
		background: rgba(255, 255, 255, 0.04);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 14px 16px;
		cursor: pointer;
		transition: all 0.15s ease;
		color: var(--text);
		font-family: inherit;
	}
	.article-card:hover {
		background: rgba(255, 255, 255, 0.07);
		border-color: rgba(124, 58, 237, 0.3);
	}
	.article-card.selected {
		border-color: var(--purple);
		background: rgba(124, 58, 237, 0.1);
		box-shadow: 0 0 0 1px var(--purple);
	}

	.card-header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 10px;
	}

	.card-title {
		margin: 0;
		font-size: 14px;
		font-weight: 600;
		line-height: 1.4;
	}

	.score-badge {
		flex-shrink: 0;
		padding: 2px 8px;
		border-radius: 999px;
		font-size: 11px;
		font-weight: 700;
		background: linear-gradient(135deg, var(--purple), var(--burgundy));
		color: #fff;
	}

	.reasons {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-top: 8px;
	}

	.article-link {
		display: inline-block;
		margin-top: 8px;
		font-size: 12px;
		color: var(--muted);
	}
	.article-link:hover { color: var(--purple); }
</style>

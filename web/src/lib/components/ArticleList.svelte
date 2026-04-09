<!--
  Scrollable article feed.
  Renders ArticleCard components for each article in the store.
  Shows loading skeletons while fetching and an error state if it fails.
-->
<script lang="ts">
	import { articles, articlesLoading, articlesError, loadArticles } from '$lib/stores/articles';
	import { isLoggedIn, profile } from '$lib/stores/user';
	import ArticleCard from './ArticleCard.svelte';
	import type { Article } from '$lib/types';

	/** Which article is currently selected (bound by parent) */
	export let selectedArticle: Article | null = null;

	function handleSelect(e: CustomEvent<Article>) {
		selectedArticle = e.detail;
	}
</script>

<div class="article-list card">
	<div class="list-header">
		<h2>
			{$isLoggedIn && $profile ? 'Your Feed' : 'Latest Articles'}
		</h2>
		<button class="btn btn-sm" on:click={loadArticles} disabled={$articlesLoading}>
			{$articlesLoading ? 'Loading...' : 'Refresh'}
		</button>
	</div>

	{#if $isLoggedIn && !$profile && !$articlesLoading && !$articlesError}
		<p class="info-msg">
			No saved profile found for this user yet. Showing the latest articles until preferences are saved.
		</p>
	{/if}

	{#if $articlesError}
		<div class="error-msg">{$articlesError}</div>
	{:else if $articlesLoading}
		<!-- skeleton placeholders while loading -->
		{#each Array(5) as _}
			<div class="skeleton skel-card"></div>
		{/each}
	{:else if $articles.length === 0}
		<p class="empty">No articles found. Click Refresh to fetch the RSS feed.</p>
	{:else}
		<div class="list-scroll">
			{#each $articles as article (article.link)}
				<ArticleCard
					{article}
					selected={selectedArticle?.link === article.link}
					on:select={handleSelect}
				/>
			{/each}
		</div>
	{/if}
</div>

<style>
	.article-list { display: flex; flex-direction: column; gap: 10px; }

	.list-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.list-header h2 { margin: 0; font-size: 16px; }

	.list-scroll {
		display: flex;
		flex-direction: column;
		gap: 8px;
		max-height: 520px;
		overflow-y: auto;
		padding-right: 4px;
	}

	.skel-card { height: 72px; margin-bottom: 8px; }

	.error-msg {
		padding: 12px;
		border-radius: var(--radius-sm);
		background: rgba(239, 68, 68, 0.1);
		border: 1px solid rgba(239, 68, 68, 0.3);
		color: var(--red);
		font-size: 13px;
	}

	.info-msg {
		padding: 10px 12px;
		border-radius: var(--radius-sm);
		background: rgba(124, 58, 237, 0.08);
		border: 1px solid rgba(124, 58, 237, 0.18);
		color: var(--muted);
		font-size: 12px;
		line-height: 1.5;
	}

	.empty { color: var(--muted); font-size: 13px; text-align: center; padding: 20px; }

	.btn-sm { padding: 6px 12px; font-size: 12px; }
</style>

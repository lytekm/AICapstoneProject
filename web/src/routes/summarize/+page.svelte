<!--
  Dedicated summarization workspace (/summarize).
  Fuller layout than the dashboard -- article picker on top,
  controls and results below. Good for focused summarization work.
-->
<script lang="ts">
	import ArticleList from '$lib/components/ArticleList.svelte';
	import SummarizeForm from '$lib/components/SummarizeForm.svelte';
	import SummaryView from '$lib/components/SummaryView.svelte';
	import { resetSummary } from '$lib/stores/summary';
	import type { Article } from '$lib/types';

	let selectedArticle: Article | null = null;

	// clear old results when switching articles
	$: if (selectedArticle) resetSummary();
</script>

<svelte:head>
	<title>Summarize - Adaptive Summarizer</title>
</svelte:head>

<h1 class="page-title">Summarization Workspace</h1>

<div class="workspace">
	<div class="article-col">
		<ArticleList bind:selectedArticle />
	</div>

	<div class="work-col">
		<SummarizeForm bind:selectedArticle />
		<SummaryView articleTitle={selectedArticle?.title ?? ''} />
	</div>
</div>

<style>
	.page-title {
		font-size: 20px;
		margin: 0 0 16px;
	}

	.workspace {
		display: grid;
		grid-template-columns: 1fr 1.4fr;
		gap: 16px;
		align-items: start;
	}

	.work-col {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	@media (max-width: 900px) {
		.workspace { grid-template-columns: 1fr; }
	}
</style>

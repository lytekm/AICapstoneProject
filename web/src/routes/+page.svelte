<!--
  Dashboard page (/).
  Shows the article feed on the left and a quick summary panel on
  the right. Logged-in users see personalized (ranked) articles;
  anonymous users see the raw RSS feed.
-->
<script lang="ts">
	import ArticleList from '$lib/components/ArticleList.svelte';
	import SummarizeForm from '$lib/components/SummarizeForm.svelte';
	import SummaryView from '$lib/components/SummaryView.svelte';
	import { isLoggedIn } from '$lib/stores/user';
	import type { Article } from '$lib/types';

	let selectedArticle: Article | null = null;
</script>

<svelte:head>
	<title>Dashboard - Adaptive Summarizer</title>
</svelte:head>

<div class="dashboard">
	<div class="col-left">
		<ArticleList bind:selectedArticle />

		{#if $isLoggedIn}
			<p class="personalized-note">
				Articles ranked by your profile preferences. Adjust them on the
				<a href="/profile">Profile</a> page.
			</p>
		{/if}
	</div>

	<div class="col-right">
		<SummarizeForm bind:selectedArticle />
		<SummaryView articleTitle={selectedArticle?.title ?? ''} />
	</div>
</div>

<style>
	.dashboard {
		display: grid;
		grid-template-columns: 1fr 1.2fr;
		gap: 16px;
		align-items: start;
	}

	.col-right {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.personalized-note {
		font-size: 12px;
		color: var(--muted);
		text-align: center;
		margin-top: 8px;
	}

	@media (max-width: 900px) {
		.dashboard { grid-template-columns: 1fr; }
	}
</style>

<!--
  User profile management page (/profile).
  Wraps the PreferencesForm component with some context about
  how the profile affects the rest of the application.
-->
<script lang="ts">
	import PreferencesForm from '$lib/components/PreferencesForm.svelte';
	import { isLoggedIn, profile } from '$lib/stores/user';
</script>

<svelte:head>
	<title>Profile - Adaptive Summarizer</title>
</svelte:head>

<h1 class="page-title">User Profile</h1>

<div class="profile-layout">
	<div class="form-col">
		<PreferencesForm />
	</div>

	<div class="info-col">
		<div class="card info-card">
			<h3>How Profiles Work</h3>
			<ul>
				<li>
					<strong>Topics</strong> and <strong>keywords</strong> influence which articles
					appear first in your feed. Articles matching your interests get higher relevance scores.
				</li>
				<li>
					<strong>Default persona</strong> and <strong>length</strong> pre-fill the summarization
					controls so you don't have to set them every time.
				</li>
				<li>
					<strong>Feedback weights</strong> are adjusted automatically when you like or dislike
					summaries. Liked topics get boosted in future rankings; disliked topics get slightly
					penalized.
				</li>
			</ul>
		</div>

		{#if $isLoggedIn && $profile}
			<div class="card stats-card">
				<h3>Profile Statistics</h3>
				<div class="stat-row">
					<span>Topics configured</span>
					<strong>{$profile.preferred_topics.length}</strong>
				</div>
				<div class="stat-row">
					<span>Keywords configured</span>
					<strong>{$profile.keywords.length}</strong>
				</div>
				<div class="stat-row">
					<span>Feedback signals</span>
					<strong>{Object.keys($profile.feedback_weights).length}</strong>
				</div>
			</div>
		{/if}
	</div>
</div>

<style>
	.page-title { font-size: 20px; margin: 0 0 16px; }

	.profile-layout {
		display: grid;
		grid-template-columns: 1.2fr 1fr;
		gap: 16px;
		align-items: start;
	}

	.info-col {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.info-card h3, .stats-card h3 { margin: 0 0 10px; font-size: 14px; }

	.info-card ul {
		margin: 0;
		padding-left: 18px;
		font-size: 13px;
		color: var(--muted);
		line-height: 1.7;
	}
	.info-card li { margin-bottom: 8px; }
	.info-card strong { color: var(--text); }

	.stat-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 8px 0;
		border-bottom: 1px solid var(--border);
		font-size: 13px;
		color: var(--muted);
	}
	.stat-row:last-child { border-bottom: none; }
	.stat-row strong { color: var(--purple); font-variant-numeric: tabular-nums; }

	@media (max-width: 768px) {
		.profile-layout { grid-template-columns: 1fr; }
	}
</style>

<!--
  Thumbs up / thumbs down feedback buttons.
  Posts feedback to the backend which adjusts the user's profile
  weights for future article ranking. Only visible when a user
  is logged in (anonymous users can't give feedback).
-->
<script lang="ts">
	import { submitFeedback } from '$lib/api';
	import { userId, loadProfile } from '$lib/stores/user';
	import { loadArticles } from '$lib/stores/articles';

	export let articleTitle: string;
	export let persona: string;
	export let mode: string;

	let feedbackGiven: 'liked' | 'disliked' | null = null;
	let submitting = false;

	async function giveFeedback(liked: boolean) {
		if (!$userId || submitting) return;

		submitting = true;
		try {
			await submitFeedback({
				user_id: $userId,
				article_title: articleTitle,
				persona,
				mode,
				liked,
			});
			feedbackGiven = liked ? 'liked' : 'disliked';
			// reload profile and articles so updated weights take effect
			await loadProfile();
			await loadArticles();
		} catch {
			// silently fail -- feedback is non-critical
		} finally {
			submitting = false;
		}
	}
</script>

{#if $userId}
	<div class="feedback-row">
		{#if feedbackGiven}
			<span class="feedback-thanks">
				{feedbackGiven === 'liked' ? '👍' : '👎'} Feedback recorded
			</span>
		{:else}
			<span class="feedback-prompt">Rate this summary:</span>
			<button
				class="fb-btn fb-like"
				on:click={() => giveFeedback(true)}
				disabled={submitting}
				title="Like this summary"
			>👍</button>
			<button
				class="fb-btn fb-dislike"
				on:click={() => giveFeedback(false)}
				disabled={submitting}
				title="Dislike this summary"
			>👎</button>
		{/if}
	</div>
{/if}

<style>
	.feedback-row {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-top: 12px;
	}

	.feedback-prompt {
		font-size: 12px;
		color: var(--muted);
	}

	.fb-btn {
		background: rgba(255, 255, 255, 0.06);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 6px 12px;
		cursor: pointer;
		font-size: 16px;
		transition: all 0.15s;
	}
	.fb-btn:hover:not(:disabled) {
		background: rgba(255, 255, 255, 0.12);
		transform: scale(1.1);
	}
	.fb-btn:disabled { opacity: 0.4; cursor: not-allowed; }

	.feedback-thanks {
		font-size: 13px;
		color: var(--green);
	}
</style>

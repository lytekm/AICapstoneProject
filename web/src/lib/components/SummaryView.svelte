<!--
  Complete summary display with metadata, confidence, flagged entities,
  and feedback controls. Handles all three states: loading, streaming,
  and completed. Shows a skeleton loader before results arrive.
-->
<script lang="ts">
	import {
		summaryText,
		summaryResult,
		summaryLoading,
		summaryError,
		summaryLatency,
		isStreaming,
	} from '$lib/stores/summary';
	import StreamingText from './StreamingText.svelte';
	import ConfidenceBadge from './ConfidenceBadge.svelte';
	import FlaggedEntityChips from './FlaggedEntityChips.svelte';
	import FeedbackButtons from './FeedbackButtons.svelte';

	export let articleTitle = '';
</script>

<div class="summary-panel card">
	<div class="panel-header">
		<h2>Summary</h2>
		{#if $summaryResult}
			<div class="meta-chips">
				<span class="chip">{$summaryResult.mode}</span>
				<span class="chip">{$summaryResult.persona}</span>
				<ConfidenceBadge confidence={$summaryResult.confidence ?? null} />
			</div>
		{/if}
	</div>

	{#if $summaryError}
		<div class="error-msg">{$summaryError}</div>
	{:else if $summaryLoading && !$isStreaming}
		<!-- non-streaming: show skeleton while waiting -->
		<div class="skeleton skel-text"></div>
		<div class="skeleton skel-text short"></div>
	{:else if $isStreaming || $summaryText}
		<!-- streaming or completed: show the text -->
		<StreamingText />

		{#if $summaryResult}
			<FlaggedEntityChips entities={$summaryResult.flagged_entities ?? null} />

			<div class="summary-footer">
				{#if $summaryLatency}
					<span class="latency">{$summaryLatency}ms</span>
				{/if}
			</div>

			<FeedbackButtons
				{articleTitle}
				persona={$summaryResult.persona}
				mode={$summaryResult.mode}
			/>
		{/if}
	{:else}
		<p class="empty-state">
			Select an article and click Summarize to see results here.
		</p>
	{/if}
</div>

<style>
	.summary-panel { min-height: 200px; }

	.panel-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 8px;
		margin-bottom: 14px;
	}
	.panel-header h2 { margin: 0; font-size: 16px; }

	.meta-chips { display: flex; gap: 6px; align-items: center; }

	.error-msg {
		padding: 12px;
		border-radius: var(--radius-sm);
		background: rgba(239, 68, 68, 0.1);
		border: 1px solid rgba(239, 68, 68, 0.3);
		color: var(--red);
		font-size: 13px;
	}

	.skel-text { height: 16px; margin-bottom: 10px; width: 100%; }
	.skel-text.short { width: 65%; }

	.summary-footer {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-top: 12px;
		padding-top: 10px;
		border-top: 1px solid var(--border);
	}

	.latency {
		font-size: 12px;
		color: var(--muted);
		font-variant-numeric: tabular-nums;
	}

	.empty-state {
		color: var(--muted);
		font-size: 13px;
		text-align: center;
		padding: 40px 20px;
	}
</style>

<!--
  Progressive text renderer for SSE streaming.
  Binds to the summary store and shows a blinking cursor while
  tokens are arriving. When streaming finishes, the cursor disappears
  and the full text is shown.
-->
<script lang="ts">
	import { summaryText, isStreaming } from '$lib/stores/summary';
</script>

<div class="streaming-container">
	<p class="streaming-text">
		{$summaryText}{#if $isStreaming}<span class="cursor">|</span>{/if}
	</p>
</div>

<style>
	.streaming-container {
		min-height: 60px;
	}

	.streaming-text {
		margin: 0;
		font-size: 14px;
		line-height: 1.7;
		white-space: pre-wrap;
		word-break: break-word;
	}

	.cursor {
		animation: blink 0.7s step-end infinite;
		color: var(--purple);
		font-weight: 300;
	}

	@keyframes blink {
		50% { opacity: 0; }
	}
</style>

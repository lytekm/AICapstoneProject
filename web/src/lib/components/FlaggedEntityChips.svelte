<!--
  Displays entities that the NER verifier flagged as potentially
  hallucinated (present in the summary but not the source article).
  Only shows when there are flagged entities to display.
-->
<script lang="ts">
	export let entities: string[] | null;

	$: visible = entities && entities.length > 0;
</script>

{#if visible}
	<div class="flagged-section">
		<span class="flagged-label">Flagged entities:</span>
		<div class="flagged-chips">
			{#each entities as entity}
				<span class="chip chip-warning" title="Entity not found in source article">
					{entity}
				</span>
			{/each}
		</div>
	</div>
{/if}

<style>
	.flagged-section {
		display: flex;
		align-items: flex-start;
		gap: 8px;
		margin-top: 8px;
		flex-wrap: wrap;
	}

	.flagged-label {
		font-size: 12px;
		color: var(--muted);
		padding-top: 3px;
		white-space: nowrap;
	}

	.flagged-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}

	.chip-warning {
		background: rgba(239, 68, 68, 0.1);
		border-color: rgba(239, 68, 68, 0.25);
		color: var(--red);
	}
</style>

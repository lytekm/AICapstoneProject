<!--
  Color-coded confidence indicator.
  Green (>= 0.8), yellow (>= 0.5), red (below 0.5).
  Null confidence means the metric is not applicable (extractive mode).
-->
<script lang="ts">
	export let confidence: number | null;

	$: numericConfidence =
		typeof confidence === 'number' && Number.isFinite(confidence) ? confidence : null;
	$: label = numericConfidence !== null ? `${(numericConfidence * 100).toFixed(0)}%` : 'N/A';
	$: level =
		numericConfidence === null ? 'neutral'
		: numericConfidence >= 0.8 ? 'high'
		: numericConfidence >= 0.5 ? 'medium'
		: 'low';
</script>

<span class="badge badge-{level}" title="NER verification confidence">
	{label}
</span>

<style>
	.badge {
		display: inline-flex;
		align-items: center;
		padding: 3px 10px;
		border-radius: 999px;
		font-size: 12px;
		font-weight: 600;
	}
	.badge-high    { background: rgba(34, 197, 94, 0.15); color: var(--green); border: 1px solid rgba(34, 197, 94, 0.3); }
	.badge-medium  { background: rgba(234, 179, 8, 0.15); color: var(--yellow); border: 1px solid rgba(234, 179, 8, 0.3); }
	.badge-low     { background: rgba(239, 68, 68, 0.15); color: var(--red); border: 1px solid rgba(239, 68, 68, 0.3); }
	.badge-neutral { background: rgba(255, 255, 255, 0.06); color: var(--muted); border: 1px solid var(--border); }
</style>

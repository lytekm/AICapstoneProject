<!--
  Summarization control panel.
  Dropdowns for mode, persona, and length.
  The UI keeps a fixed extraction budget internally so users are not
  asked to manage both "length" and "k" in the same screen.
  Handles both streaming and non-streaming requests based on mode.
  The "Summarize" button is disabled until an article is selected.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { fetchPersonas } from '$lib/api';
	import { userId, profile } from '$lib/stores/user';
	import { runSummarize, runSummarizeStream, summaryLoading } from '$lib/stores/summary';
	import { MODES, LENGTHS } from '$lib/types';
	import type { Article } from '$lib/types';

	export let selectedArticle: Article | null = null;

	/** Whether to use SSE streaming (auto-enabled for abstractive/hybrid) */
	export let useStreaming = true;
	const DEFAULT_K = 5;

	let personas: string[] = ['default'];
	let mode = 'hybrid';
	let persona = 'default';
	let length = 'standard';
	let personaDirty = false;
	let lengthDirty = false;
	let appliedProfileUserId = '';

	onMount(async () => {
		try {
			personas = await fetchPersonas();
		} catch {
			// fallback to the known set if the API is down
			personas = ['default', 'technical', 'casual', 'executive', 'academic'];
		}
	});

	// apply profile defaults once per loaded profile unless the user has
	// already made a manual choice in this session
	$: if ($profile) {
		if ($profile.user_id !== appliedProfileUserId) {
			appliedProfileUserId = $profile.user_id;
			personaDirty = false;
			lengthDirty = false;
		}

		if (!personaDirty) {
			persona = $profile.default_persona || persona;
		}

		if (!lengthDirty) {
			length = $profile.default_length || length;
		}
	} else if (appliedProfileUserId) {
		appliedProfileUserId = '';
		personaDirty = false;
		lengthDirty = false;
	}

	function handleSummarize() {
		if (!selectedArticle) return;

		const payload = {
			url: selectedArticle.link,
			k: DEFAULT_K,
			mode,
			persona,
			length,
			user_id: $userId || undefined,
		};

		// extractive has nothing to stream, always use POST
		// for abstractive/hybrid, use SSE if streaming is enabled
		if (mode === 'extractive' || !useStreaming) {
			runSummarize(payload);
		} else {
			runSummarizeStream({
				url: selectedArticle.link,
				k: DEFAULT_K,
				mode,
				persona,
				length,
			});
		}
	}
</script>

<div class="form-panel card">
	<h2>Pipeline Controls</h2>

	<div class="controls-grid">
		<label>
			<span class="label-text">Mode</span>
			<select bind:value={mode}>
				{#each MODES as m}
					<option value={m}>{m}</option>
				{/each}
			</select>
		</label>

		<label>
			<span class="label-text">Persona</span>
			<select bind:value={persona} on:change={() => { personaDirty = true; }}>
				{#each personas as p}
					<option value={p}>{p}</option>
				{/each}
			</select>
		</label>

		<label>
			<span class="label-text">Length</span>
			<select bind:value={length} on:change={() => { lengthDirty = true; }}>
				{#each LENGTHS as l}
					<option value={l}>{l}</option>
				{/each}
			</select>
		</label>
	</div>

	<div class="form-footer">
		<label class="stream-toggle">
			<input type="checkbox" bind:checked={useStreaming} />
			<span>Stream tokens</span>
		</label>

		<button
			class="btn"
			on:click={handleSummarize}
			disabled={!selectedArticle || $summaryLoading}
		>
			{#if $summaryLoading}
				Summarizing...
			{:else}
				Summarize
			{/if}
		</button>
	</div>

	{#if !selectedArticle}
		<p class="hint">Select an article from the feed to begin.</p>
	{/if}

	<div class="mode-hint">
		{#if mode === 'extractive'}
			Selects top evidence sentences using TextRank + MMR. Length now scales the effective sentence budget. No LLM required.
		{:else if mode === 'abstractive'}
			Extracts a fixed evidence set, then rewrites it with the LLM in the selected persona style.
		{:else}
			Full pipeline: extract, abstract with LLM, then verify with NER for grounding issues.
		{/if}
	</div>
</div>

<style>
	.form-panel h2 { margin: 0 0 14px; font-size: 16px; }

	.controls-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 12px;
	}

	label { display: flex; flex-direction: column; gap: 4px; }
	.label-text { font-size: 12px; color: var(--muted); font-weight: 500; }

	select { width: 100%; }

	.form-footer {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-top: 16px;
	}

	.stream-toggle {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 13px;
		color: var(--muted);
		cursor: pointer;
		flex-direction: row;
	}
	.stream-toggle input { accent-color: var(--purple); }

	.hint {
		margin: 10px 0 0;
		font-size: 12px;
		color: var(--muted);
		text-align: center;
	}

	.mode-hint {
		margin-top: 12px;
		padding: 10px 12px;
		border-radius: var(--radius-sm);
		background: rgba(124, 58, 237, 0.08);
		border: 1px solid rgba(124, 58, 237, 0.15);
		font-size: 12px;
		color: var(--muted);
		line-height: 1.5;
	}

	@media (max-width: 480px) {
		.controls-grid { grid-template-columns: 1fr; }
	}
</style>

<!--
  User preferences editor.
  Lets users configure their profile: preferred topics, keywords,
  default persona, and default summary length. Saves to the backend
  and reloads the profile store on success.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { fetchPersonas, saveProfile } from '$lib/api';
	import { userId, profile, loadProfile } from '$lib/stores/user';
	import { loadArticles } from '$lib/stores/articles';
	import { LENGTHS } from '$lib/types';

	let personas: string[] = ['default'];
	let topics = '';
	let keywords = '';
	let defaultPersona = 'default';
	let defaultLength = 'standard';
	let saving = false;
	let message = '';

	onMount(async () => {
		try { personas = await fetchPersonas(); } catch {}
	});

	// populate form fields when the profile loads
	$: if ($profile) {
		topics = $profile.preferred_topics.join(', ');
		keywords = $profile.keywords.join(', ');
		defaultPersona = $profile.default_persona;
		defaultLength = $profile.default_length;
	}

	async function handleSave() {
		if (!$userId) {
			message = 'Sign in first using the navbar.';
			return;
		}
		saving = true;
		message = '';
		try {
			// split comma-separated inputs into arrays, trim whitespace
			const topicList = topics.split(',').map((t) => t.trim()).filter(Boolean);
			const kwList = keywords.split(',').map((k) => k.trim()).filter(Boolean);

			await saveProfile({
				user_id: $userId,
				preferred_topics: topicList,
				keywords: kwList,
				default_persona: defaultPersona,
				default_length: defaultLength,
			});
			await loadProfile();
			await loadArticles();
			message = 'Preferences saved.';
		} catch (err) {
			message = err instanceof Error ? err.message : 'Save failed';
		} finally {
			saving = false;
		}
	}
</script>

<div class="prefs-form card">
	<h2>User Preferences</h2>

	{#if !$userId}
		<p class="hint">Sign in using the navbar to manage your preferences.</p>
	{:else}
		<p class="user-label">Editing profile: <strong>{$userId}</strong></p>

		<div class="field-stack">
			<label>
				<span class="label-text">Preferred Topics</span>
				<input type="text" bind:value={topics} placeholder="AI, finance, health..." />
				<span class="field-hint">Comma-separated. Used to rank articles in your feed.</span>
			</label>

			<label>
				<span class="label-text">Keywords</span>
				<input type="text" bind:value={keywords} placeholder="startup, GPU, climate..." />
				<span class="field-hint">Specific terms to boost in article ranking.</span>
			</label>

			<div class="inline-fields">
				<label>
					<span class="label-text">Default Persona</span>
					<select bind:value={defaultPersona}>
						{#each personas as p}
							<option value={p}>{p}</option>
						{/each}
					</select>
				</label>

				<label>
					<span class="label-text">Default Length</span>
					<select bind:value={defaultLength}>
						{#each LENGTHS as l}
							<option value={l}>{l}</option>
						{/each}
					</select>
				</label>
			</div>
		</div>

		<div class="form-actions">
			<button class="btn" on:click={handleSave} disabled={saving}>
				{saving ? 'Saving...' : 'Save Preferences'}
			</button>
			{#if message}
				<span class="save-msg">{message}</span>
			{/if}
		</div>

		{#if $profile && Object.keys($profile.feedback_weights).length > 0}
			<div class="weights-section">
				<h3>Feedback Weights</h3>
				<p class="field-hint">
					These are automatically adjusted as you like/dislike summaries.
					Higher weights boost related articles in your feed.
				</p>
				<div class="weight-grid">
					{#each Object.entries($profile.feedback_weights) as [topic, weight]}
						<div class="weight-item">
							<span class="weight-topic">{topic}</span>
							<span class="weight-value">{weight.toFixed(2)}</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	{/if}
</div>

<style>
	.prefs-form h2 { margin: 0 0 14px; font-size: 16px; }

	.user-label { font-size: 13px; color: var(--muted); margin: 0 0 16px; }
	.user-label strong { color: var(--purple); }

	.field-stack { display: flex; flex-direction: column; gap: 14px; }

	label { display: flex; flex-direction: column; gap: 4px; }
	.label-text { font-size: 12px; color: var(--muted); font-weight: 500; }
	.field-hint { font-size: 11px; color: rgba(203, 191, 230, 0.6); }

	input[type="text"] { width: 100%; }

	.inline-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

	.form-actions {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-top: 16px;
	}

	.save-msg { font-size: 13px; color: var(--green); }

	.hint { color: var(--muted); font-size: 13px; text-align: center; padding: 20px; }

	.weights-section {
		margin-top: 20px;
		padding-top: 16px;
		border-top: 1px solid var(--border);
	}
	.weights-section h3 { margin: 0 0 6px; font-size: 14px; }

	.weight-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-top: 8px;
	}
	.weight-item {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		background: rgba(255, 255, 255, 0.04);
		border: 1px solid var(--border);
		border-radius: 6px;
		font-size: 12px;
	}
	.weight-topic { color: var(--muted); }
	.weight-value { color: var(--purple); font-weight: 600; font-variant-numeric: tabular-nums; }
</style>

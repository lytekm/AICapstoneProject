<!--
  Top navigation bar.
  Shows the app branding, navigation links, and user selector.
  The user selector persists across pages via the user store.
-->
<script lang="ts">
	import { userId, loadProfile } from '$lib/stores/user';
	import { loadArticles } from '$lib/stores/articles';
	import { page } from '$app/stores';

	let userInput = $userId;

	/** When the user changes their ID, reload profile + articles */
	function switchUser() {
		const trimmed = userInput.trim();
		$userId = trimmed;
		loadProfile();
		loadArticles();
	}

	function clearUser() {
		userInput = '';
		$userId = '';
		loadProfile();
		loadArticles();
	}

	// highlight the active nav link
	$: currentPath = $page.url.pathname;
</script>

<nav class="navbar">
	<div class="nav-inner container">
		<a href="/" class="brand">
			<span class="brand-icon">📰</span>
			<div>
				<div class="brand-title">Adaptive Summarizer</div>
				<div class="brand-sub">COMP385-402 Group 4</div>
			</div>
		</a>

		<div class="nav-links">
			<a href="/" class:active={currentPath === '/'}>Dashboard</a>
			<a href="/summarize" class:active={currentPath === '/summarize'}>Summarize</a>
			<a href="/compare" class:active={currentPath === '/compare'}>Compare</a>
			<a href="/profile" class:active={currentPath === '/profile'}>Profile</a>
		</div>

		<div class="user-control">
			<input
				type="text"
				placeholder="User ID..."
				bind:value={userInput}
				on:keydown={(e) => e.key === 'Enter' && switchUser()}
			/>
			{#if $userId}
				<button class="btn-outline btn-sm" on:click={clearUser}>Sign out</button>
			{:else}
				<button class="btn btn-sm" on:click={switchUser} disabled={!userInput.trim()}>
					Sign in
				</button>
			{/if}
		</div>
	</div>
</nav>

<style>
	.navbar {
		position: sticky;
		top: 0;
		z-index: 100;
		padding: 10px 0;
		background: rgba(11, 6, 18, 0.85);
		backdrop-filter: blur(16px);
		-webkit-backdrop-filter: blur(16px);
		border-bottom: 1px solid var(--border);
	}

	.nav-inner {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
		flex-wrap: wrap;
	}

	.brand {
		display: flex;
		align-items: center;
		gap: 10px;
		text-decoration: none;
		color: var(--text);
	}
	.brand:hover { text-decoration: none; }

	.brand-icon { font-size: 28px; }
	.brand-title { font-size: 16px; font-weight: 700; }
	.brand-sub { font-size: 11px; color: var(--muted); }

	.nav-links {
		display: flex;
		gap: 6px;
	}
	.nav-links a {
		padding: 6px 14px;
		border-radius: 8px;
		font-size: 13px;
		font-weight: 500;
		color: var(--muted);
		text-decoration: none;
		transition: all 0.15s;
	}
	.nav-links a:hover { background: rgba(255,255,255,0.06); color: var(--text); }
	.nav-links a.active {
		background: rgba(124, 58, 237, 0.2);
		color: var(--purple);
	}

	.user-control {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.user-control input {
		width: 140px;
		padding: 6px 10px;
		font-size: 13px;
	}

	.btn-sm { padding: 6px 12px; font-size: 12px; }

	@media (max-width: 768px) {
		.nav-inner { justify-content: center; }
		.nav-links { order: 3; width: 100%; justify-content: center; }
	}
</style>

<!--
  Simple toast notification component.
  Displays a temporary message at the bottom of the screen,
  auto-dismisses after a timeout. Supports success/error variants.
-->
<script lang="ts">
	import { writable } from 'svelte/store';

	interface ToastMessage {
		id: number;
		text: string;
		type: 'success' | 'error' | 'info';
	}

	// global toast queue -- components import and call show()
	export const toasts = writable<ToastMessage[]>([]);

	let counter = 0;

	export function showToast(text: string, type: 'success' | 'error' | 'info' = 'info') {
		const id = ++counter;
		toasts.update((t) => [...t, { id, text, type }]);
		// auto-dismiss after 4 seconds
		setTimeout(() => {
			toasts.update((t) => t.filter((m) => m.id !== id));
		}, 4000);
	}
</script>

<div class="toast-container">
	{#each $toasts as toast (toast.id)}
		<div class="toast toast-{toast.type}">
			{toast.text}
		</div>
	{/each}
</div>

<style>
	.toast-container {
		position: fixed;
		bottom: 20px;
		right: 20px;
		z-index: 9999;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.toast {
		padding: 12px 18px;
		border-radius: 10px;
		font-size: 13px;
		font-weight: 500;
		color: #fff;
		backdrop-filter: blur(12px);
		animation: slideIn 0.2s ease;
		max-width: 360px;
	}

	.toast-success { background: rgba(34, 197, 94, 0.85); }
	.toast-error   { background: rgba(239, 68, 68, 0.85); }
	.toast-info    { background: rgba(124, 58, 237, 0.85); }

	@keyframes slideIn {
		from { opacity: 0; transform: translateX(20px); }
		to   { opacity: 1; transform: translateX(0); }
	}
</style>

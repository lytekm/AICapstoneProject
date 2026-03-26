/**
 * Summary store -- manages the current summarization state.
 *
 * Handles both non-streaming (single JSON) and streaming (SSE)
 * responses. Components bind to these stores to render results
 * progressively as tokens arrive.
 */

import { writable, get } from 'svelte/store';
import { summarize, summarizeStream } from '$lib/api';
import type { SummarizePayload, SummaryResult } from '$lib/types';

/** The final (or in-progress) summary text */
export const summaryText = writable('');

/** Full result metadata once the request completes */
export const summaryResult = writable<SummaryResult | null>(null);

/** Whether a summarization request is currently running */
export const summaryLoading = writable(false);

/** Error message if the request failed */
export const summaryError = writable<string | null>(null);

/** Latency in milliseconds for the most recent request */
export const summaryLatency = writable<number | null>(null);

/** Whether the current request is streaming tokens */
export const isStreaming = writable(false);

/** Active EventSource reference so we can close it if the user navigates away */
let activeSource: EventSource | null = null;

/** Clean up any active SSE connection */
export function cancelStream(): void {
	if (activeSource) {
		activeSource.close();
		activeSource = null;
	}
	isStreaming.set(false);
}

/** Reset all summary state (e.g., when switching articles) */
export function resetSummary(): void {
	cancelStream();
	summaryText.set('');
	summaryResult.set(null);
	summaryLoading.set(false);
	summaryError.set(null);
	summaryLatency.set(null);
}

/**
 * Run a non-streaming summarization.
 * Used when the user doesn't need progressive rendering
 * or when the mode is extractive (nothing to stream).
 */
export async function runSummarize(payload: SummarizePayload): Promise<void> {
	resetSummary();
	summaryLoading.set(true);
	const t0 = performance.now();

	try {
		const result = await summarize(payload);
		summaryText.set(result.summary);
		summaryResult.set(result);
		summaryLatency.set(Math.round(performance.now() - t0));
	} catch (err) {
		summaryError.set(err instanceof Error ? err.message : 'Summarization failed');
	} finally {
		summaryLoading.set(false);
	}
}

/**
 * Run a streaming summarization via SSE.
 * Tokens arrive one at a time and get appended to summaryText,
 * giving the user real-time feedback as the LLM generates.
 */
export function runSummarizeStream(params: {
	url: string;
	k: number;
	mode: string;
	persona: string;
	length: string;
}): void {
	resetSummary();
	summaryLoading.set(true);
	isStreaming.set(true);
	const t0 = performance.now();

	const source = summarizeStream(params);
	activeSource = source;

	// each token chunk gets appended to the running text
	source.addEventListener('token', (e: MessageEvent) => {
		try {
			const data = JSON.parse(e.data);
			summaryText.update((prev) => prev + (data.text ?? ''));
		} catch { /* malformed event, skip */ }
	});

	// the final event carries confidence, flagged entities, full summary
	source.addEventListener('done', (e: MessageEvent) => {
		try {
			const data = JSON.parse(e.data);
			summaryResult.set({
				summary: data.summary ?? get(summaryText),
				mode: data.mode ?? params.mode,
				persona: data.persona ?? params.persona,
				confidence: data.confidence ?? null,
				flagged_entities: data.flagged_entities ?? null,
			});
			// use the server's full text in case we missed a token
			summaryText.set(data.summary ?? get(summaryText));
			summaryLatency.set(Math.round(performance.now() - t0));
		} catch { /* best effort */ }
		cancelStream();
		summaryLoading.set(false);
	});

	// SSE error -- either network failure or the server sent an error event
	source.addEventListener('error', (e: MessageEvent) => {
		let detail = 'Stream connection failed';
		try {
			if (e.data) {
				const data = JSON.parse(e.data);
				detail = data.detail ?? detail;
			}
		} catch { /* not a JSON error event, probably a connection drop */ }
		summaryError.set(detail);
		cancelStream();
		summaryLoading.set(false);
	});
}

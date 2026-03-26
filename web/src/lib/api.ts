/**
 * API client -- thin wrapper around fetch() for every backend endpoint.
 *
 * All functions throw on non-2xx responses so callers can catch in
 * a single try/catch rather than checking status codes everywhere.
 * The base URL is empty because Vite proxies /api to FastAPI in dev,
 * and in production the same server serves both static files and the API.
 */

import type {
	Article,
	FeedbackPayload,
	PreferencesPayload,
	RankedArticle,
	SummarizePayload,
	SummaryResult,
	UserProfile,
} from './types';

const BASE = '';

/** Generic fetch helper with error handling */
async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(`${BASE}${path}`, init);
	if (!res.ok) {
		const body = await res.text().catch(() => 'Unknown error');
		throw new Error(`API ${res.status}: ${body}`);
	}
	return res.json();
}

// ---------------------------------------------------------------
// Health
// ---------------------------------------------------------------

export async function checkHealth(): Promise<{ status: string }> {
	return apiFetch('/api/health');
}

// ---------------------------------------------------------------
// Articles
// ---------------------------------------------------------------

/** Fetch raw (unranked) RSS articles */
export async function fetchArticles(): Promise<Article[]> {
	return apiFetch('/api/articles');
}

/** Fetch articles ranked by a user's profile preferences */
export async function fetchPersonalizedArticles(userId: string): Promise<RankedArticle[]> {
	return apiFetch(`/api/articles/personalized?user_id=${encodeURIComponent(userId)}`);
}

// ---------------------------------------------------------------
// Personas
// ---------------------------------------------------------------

export async function fetchPersonas(): Promise<string[]> {
	const data = await apiFetch<{ personas: string[] }>('/api/personas');
	return data.personas;
}

// ---------------------------------------------------------------
// Summarization
// ---------------------------------------------------------------

/** Non-streaming summarize -- single JSON response */
export async function summarize(payload: SummarizePayload): Promise<SummaryResult> {
	return apiFetch('/api/summarize', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(payload),
	});
}

/**
 * Streaming summarize via Server-Sent Events.
 *
 * Returns an EventSource. The caller should attach listeners for:
 *   - "meta"  (mode + persona confirmation)
 *   - "token" (incremental text chunks)
 *   - "done"  (final result with confidence)
 *   - "error" (something went wrong)
 *
 * EventSource auto-reconnects on failure, but we close it on "done"
 * so we don't accidentally re-trigger the pipeline.
 */
export function summarizeStream(params: {
	url: string;
	k: number;
	mode: string;
	persona: string;
	length: string;
}): EventSource {
	const qs = new URLSearchParams({
		url: params.url,
		k: String(params.k),
		mode: params.mode,
		persona: params.persona,
		length: params.length,
	});
	return new EventSource(`/api/summarize/stream?${qs}`);
}

// ---------------------------------------------------------------
// User Profiles
// ---------------------------------------------------------------

/** Get a saved user profile by ID */
export async function getProfile(userId: string): Promise<UserProfile> {
	return apiFetch(`/api/user/preferences/${encodeURIComponent(userId)}`);
}

/** Create or update a user profile */
export async function saveProfile(payload: PreferencesPayload): Promise<{ status: string }> {
	return apiFetch('/api/user/preferences', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(payload),
	});
}

// ---------------------------------------------------------------
// Feedback
// ---------------------------------------------------------------

/** Record a like or dislike on a summary */
export async function submitFeedback(payload: FeedbackPayload): Promise<{ status: string }> {
	return apiFetch('/api/user/feedback', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(payload),
	});
}

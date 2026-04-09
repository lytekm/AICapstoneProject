/**
 * Shared TypeScript interfaces for the frontend.
 * These mirror the JSON shapes returned by the FastAPI backend
 * so components get type safety on every API response.
 */

/** Raw article from the RSS feed */
export interface Article {
	title: string;
	link: string;
}

/** Ranked article with relevance scoring from the ranker */
export interface RankedArticle extends Article {
	score: number;
	match_reasons: string[];
}

/** Full summarization response from POST /api/summarize */
export interface SummaryResult {
	summary: string;
	mode: string;
	persona: string;
	confidence: number | null;
	flagged_entities: string[] | null;
}

/** SSE event payloads -- each event type has its own shape */
export interface SSEMetaEvent {
	mode: string;
	persona: string;
}

export interface SSETokenEvent {
	text: string;
}

export interface SSEDoneEvent extends SummaryResult {
	fallback?: string;
	error?: string;
}

export interface SSEErrorEvent {
	detail: string;
}

/** User profile as stored in the backend */
export interface UserProfile {
	user_id: string;
	preferred_topics: string[];
	keywords: string[];
	default_persona: string;
	default_length: string;
	feedback_weights: Record<string, number>;
}

/** Request body for saving preferences */
export interface PreferencesPayload {
	user_id: string;
	preferred_topics: string[];
	keywords: string[];
	default_persona: string;
	default_length: string;
}

/** Request body for submitting feedback */
export interface FeedbackPayload {
	user_id: string;
	article_title: string;
	persona: string;
	mode: string;
	liked: boolean;
}

/** Request body for the summarize endpoint */
export interface SummarizePayload {
	url: string;
	k: number;
	mode: string;
	persona: string;
	length: string;
	user_id?: string;
}

/** Available pipeline modes */
export const MODES = ['extractive', 'abstractive', 'hybrid'] as const;
export type Mode = (typeof MODES)[number];

/** Available output lengths */
export const LENGTHS = ['brief', 'standard', 'detailed'] as const;
export type Length = (typeof LENGTHS)[number];

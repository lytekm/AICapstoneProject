/**
 * Articles store -- holds the current article feed.
 *
 * Automatically switches between the generic RSS feed and the
 * personalized (ranked) feed based on whether a user is logged in.
 * Components just read $articles and get the right data.
 */

import { writable, get } from 'svelte/store';
import { fetchArticles, fetchPersonalizedArticles } from '$lib/api';
import { userId } from './user';
import type { Article, RankedArticle } from '$lib/types';

/** The article list (may be plain or ranked depending on login state) */
export const articles = writable<(Article | RankedArticle)[]>([]);

/** Loading flag for the fetch */
export const articlesLoading = writable(false);

/** Error message if the fetch failed */
export const articlesError = writable<string | null>(null);

/**
 * Fetch articles from the backend.
 * If a user is logged in, fetches personalized (ranked) articles.
 * Otherwise, fetches the raw RSS feed.
 */
export async function loadArticles(): Promise<void> {
	articlesLoading.set(true);
	articlesError.set(null);
	try {
		const uid = get(userId);
		if (uid) {
			const ranked = await fetchPersonalizedArticles(uid);
			articles.set(ranked);
		} else {
			const raw = await fetchArticles();
			articles.set(raw);
		}
	} catch (err) {
		articlesError.set(err instanceof Error ? err.message : 'Failed to load articles');
		articles.set([]);
	} finally {
		articlesLoading.set(false);
	}
}

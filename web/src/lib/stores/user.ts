/**
 * User store -- tracks the currently active user ID and their profile.
 *
 * Persists the user_id to localStorage so refreshing the page
 * doesn't lose the selection. The full profile is fetched from
 * the backend whenever the user_id changes.
 */

import { writable, derived, get } from 'svelte/store';
import { getProfile } from '$lib/api';
import type { UserProfile } from '$lib/types';

/** The active user ID (empty string = anonymous / no profile) */
export const userId = writable<string>(
	typeof localStorage !== 'undefined'
		? localStorage.getItem('uas_user_id') ?? ''
		: ''
);

// sync to localStorage whenever it changes
userId.subscribe((val) => {
	if (typeof localStorage !== 'undefined') {
		if (val) localStorage.setItem('uas_user_id', val);
		else localStorage.removeItem('uas_user_id');
	}
});

/** Full profile fetched from backend (null if anonymous or not yet loaded) */
export const profile = writable<UserProfile | null>(null);

/** Whether a profile fetch is in flight */
export const profileLoading = writable(false);

/**
 * Load the profile for the current user ID.
 * Called on app mount and whenever the user switches accounts.
 */
export async function loadProfile(): Promise<void> {
	const uid = get(userId);
	if (!uid) {
		profile.set(null);
		return;
	}
	profileLoading.set(true);
	try {
		const p = await getProfile(uid);
		profile.set(p);
	} catch {
		// user doesn't exist yet -- that's fine, they'll create via prefs page
		profile.set(null);
	} finally {
		profileLoading.set(false);
	}
}

/** Convenience: whether a user is currently logged in */
export const isLoggedIn = derived(userId, ($uid) => $uid.length > 0);

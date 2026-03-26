<!--
  Root layout -- wraps every page with the navbar and toast system.
  Loads the user profile and article feed on first mount so child
  pages always have data ready.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import '../app.css';
	import Navbar from '$lib/components/Navbar.svelte';
	import Toast from '$lib/components/Toast.svelte';
	import { loadProfile } from '$lib/stores/user';
	import { loadArticles } from '$lib/stores/articles';

	onMount(() => {
		// hydrate stores on first load
		loadProfile();
		loadArticles();
	});
</script>

<Navbar />

<main class="container page-enter">
	<slot />
</main>

<Toast />

<style>
	main {
		padding-top: 20px;
		padding-bottom: 40px;
	}
</style>

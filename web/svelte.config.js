import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		// builds to static HTML/JS/CSS so FastAPI can serve it
		adapter: adapter({
			pages: '../frontend',
			assets: '../frontend',
			fallback: 'index.html',
			precompress: false,
		}),
		// all API calls go through /api/ which FastAPI handles
		paths: {
			base: '',
		},
	},
};

export default config;

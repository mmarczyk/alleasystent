const CACHE = 'alleasystent-v25';

// Everything needed to render the UI shell without a network request
const SHELL = [
  './',
  './manifest.json',
  './css/app.css',
  './css/vendor/github-dark.min.css',
  './js/app.js',
  './js/vendor/marked.min.js',
  './js/vendor/highlight.min.js',
  './icons/icon-192.svg',
  './icons/icon-512.svg',
];

// Allow the page to trigger an immediate SW swap via postMessage
self.addEventListener('message', e => {
  if (e.data?.type === 'SKIP_WAITING') self.skipWaiting();
});

// Pre-cache the entire shell on install so the app is instantly available
self.addEventListener('install', e =>
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)))
);

// Drop old caches and take control of all clients immediately
self.addEventListener('activate', e =>
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  )
);

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);

  // API & auth — always network, never cache
  if (
    url.pathname.startsWith('/api/') ||
    url.pathname.startsWith('/allegro/') ||
    url.pathname.startsWith('/auth/') ||
    url.pathname.startsWith('/chat')
  ) return;

  // App shell HTML — network-first so auth state stays fresh;
  // fall back to cached shell so the UI opens even when offline
  if (url.pathname === '/' || url.pathname.endsWith('.html')) {
    e.respondWith(
      fetch(e.request)
        .then(r => {
          caches.open(CACHE).then(c => c.put(e.request, r.clone()));
          return r;
        })
        .catch(() => caches.match('./'))
    );
    return;
  }

  // Static assets (CSS, JS, icons, vendor) — cache-first, update in background
  e.respondWith(
    caches.open(CACHE).then(cache =>
      cache.match(e.request).then(cached => {
        const fromNetwork = fetch(e.request).then(r => {
          if (r.ok) cache.put(e.request, r.clone());
          return r;
        }).catch(() => cached);
        // Return cached immediately; network response updates cache in background
        return cached ?? fromNetwork;
      })
    )
  );
});

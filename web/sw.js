const CACHE = 'alleasystent-v5';
const STATIC = [
  './manifest.json',
  './css/app.css', './js/app.js'
];

self.addEventListener('install', e =>
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(STATIC)).then(() => self.skipWaiting()))
);

self.addEventListener('activate', e =>
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  )
);

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  // Always fetch HTML from network so auth state is always fresh
  if (url.pathname === '/' || url.pathname.endsWith('.html')) {
    e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
    return;
  }
  if (url.hostname.includes('cdn.') || url.hostname !== self.location.hostname) return;
  e.respondWith(caches.match(e.request).then(c => c ?? fetch(e.request)));
});

const CACHE = 'alleasystent-v2';
const STATIC = [
  './', './index.html', './manifest.json',
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
  if (e.request.url.includes('googleapis.com') || e.request.url.includes('cdn.')) return;
  e.respondWith(caches.match(e.request).then(c => c ?? fetch(e.request)));
});

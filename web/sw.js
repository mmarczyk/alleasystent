const CACHE = 'alleasystent-v39';

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

// Pre-cache the entire shell on install so the app is instantly available.
// skipWaiting() activates a new SW as soon as it's installed instead of
// sitting in "waiting" until every open tab closes — Safari in particular
// can go a very long time (sometimes indefinitely across tabs/windows)
// without ever surfacing the "installed" update-banner event otherwise.
self.addEventListener('install', e =>
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  )
);

// Drop old caches and take control of all clients immediately
self.addEventListener('activate', e =>
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  )
);

// ── Web Push ──────────────────────────────────────────────────────────────────

self.addEventListener('push', e => {
  const data = e.data?.json() ?? {};
  e.waitUntil(
    self.registration.showNotification(data.title ?? 'AllEasystent', {
      body: data.body ?? '',
      icon: './icons/icon-192.svg',
      badge: './icons/icon-192.svg',
      data: { url: data.url ?? '/' },
      vibrate: [200, 100, 200],
      tag: 'alleasystent-monitor',  // replaces any direct Notification on same device silently
      renotify: false,
    })
  );
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  const url = e.notification.data?.url ?? '/';
  e.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(cs => {
      const origin = self.location.origin;
      const target = new URL(url, origin).href;
      const existing = cs.find(c => c.url === target || c.url.startsWith(origin));
      return existing ? existing.focus() : clients.openWindow(url);
    })
  );
});

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
  // fall back to cached shell so the UI opens even when offline.
  // Paths ending in '/' cover GitHub Pages subdir URLs (/alleasystent/).
  if (url.pathname === '/' || url.pathname.endsWith('/') || url.pathname.endsWith('.html')) {
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

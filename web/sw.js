// Injected with the deploy's commit SHA by deploy-chat.yml, so this file's
// bytes — and therefore the cache name — change on every deploy. That's what
// makes the browser's own sw.js byte-diff detect an update; no one has to
// remember to bump a version number by hand.
const CACHE = 'alleasystent-__GIT_SHA__';

// Everything needed to render the UI shell without a network request
const SHELL = [
  './',
  './manifest.json',
  './config.js',
  './css/app.css',
  './css/vendor/github-dark.min.css',
  './css/vendor/github.min.css',
  './js/app.js',
  './js/vendor/marked.min.js',
  './js/vendor/highlight.min.js',
  './js/vendor/chart.min.js',
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
      // Carry the full payload through to notificationclick (not just the url) so
      // the app can paint this notification into the inbox instantly on launch,
      // instead of waiting on a round-trip to the server to find out it exists.
      data: {
        url: data.url ?? '/',
        id: data.id ?? null,
        title: data.title ?? null,
        body: data.body ?? null,
        prompt: data.prompt ?? null,
        created_at: data.created_at ?? null,
      },
      vibrate: [200, 100, 200],
      tag: 'alleasystent-monitor',  // replaces any direct Notification on same device silently
      renotify: false,
    })
  );
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  // Tapping the notification opens straight into the Notifications panel (app.js
  // reads ?open=notifications on load) — it does NOT re-fire the chat question or
  // re-poll for new orders/invoices; the notification itself IS the detection.
  // Notification URLs from the backend are root-relative (e.g. "/?open=notifications"),
  // which is correct for an all-in-one deployment but wrong on GitHub Pages, where the
  // app lives under a subpath (e.g. /alleasystent/) — resolving "/x" against the origin
  // drops that subpath entirely and 404s. Strip the leading slash and resolve against
  // this SW's own registration scope instead, so both deployment modes land correctly.
  const d = e.notification.data || {};
  const url = d.url ?? './';
  const target = new URL(url.replace(/^\/+/, ''), self.registration.scope);
  // Tack the notification's own title/body/prompt onto the launch URL so app.js
  // can render it immediately, before its background refresh() reaches the server
  // — the notification data IS already known, no need to wait to display it.
  if (d.id) target.searchParams.set('nid', d.id);
  if (d.title) target.searchParams.set('ntitle', d.title);
  if (d.body) target.searchParams.set('nbody', d.body);
  if (d.prompt) target.searchParams.set('nprompt', d.prompt);
  if (d.created_at) target.searchParams.set('ncreated', d.created_at);
  const href = target.href;
  e.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(cs => {
      const origin = self.location.origin;
      const existing = cs.find(c => c.url.startsWith(origin));
      if (existing) {
        return (existing.navigate ? existing.navigate(href) : Promise.resolve(existing)).then(c => c.focus());
      }
      return clients.openWindow(href);
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
    url.pathname.startsWith('/chat') ||
    url.pathname.startsWith('/notifications') ||
    url.pathname.startsWith('/push/')
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

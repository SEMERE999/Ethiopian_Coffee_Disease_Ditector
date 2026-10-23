// Coffee Doctor — service worker
// Caches the app shell (HTML/CSS/JS/locales) so the interface still loads
// offline or on a weak connection. API calls (/api/*) always hit the
// network — predictions require the live model, so they are never cached.

const CACHE_NAME = "coffee-doctor-shell-v1";
const SHELL_ASSETS = [
  "./",
  "index.html",
  "manifest.json",
  "static/css/style.css",
  "static/js/app.js",
  "static/locales/en.json",
  "static/locales/am.json",
  "static/locales/ti.json",
  "static/locales/om.json",
  "static/icons/icon-192.png",
  "static/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls — always go to the network.
  if (url.pathname.startsWith("/api/")) return;

  if (event.request.method !== "GET") return;

  event.respondWith(
    caches.match(event.request).then((cached) => {
      const networkFetch = fetch(event.request)
        .then((response) => {
          if (response && response.status === 200) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
          }
          return response;
        })
        .catch(() => cached);
      return cached || networkFetch;
    })
  );
});

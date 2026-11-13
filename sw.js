// vqwen site service worker: controls the page; passes all requests through.
self.addEventListener('install', function(e){ self.skipWaiting(); });
self.addEventListener('activate', function(e){ e.waitUntil(self.clients.claim()); });
self.addEventListener('fetch', function(e){
  // Pass through: the page reaches the machine directly over CORS.
});

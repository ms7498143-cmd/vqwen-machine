// vqwen proxy service worker: serve same-origin /api/* from the real machine.
// The page's base URL is therefore its own github.io link - no Cloudflare,
// no Serveo. Repoint by editing endpoint.json (no HTML republish needed).
var ENDPOINT = null;
self.addEventListener('install', function(e){ self.skipWaiting(); });
self.addEventListener('activate', function(e){ e.waitUntil(self.clients.claim()); });
function loadEndpoint(){
  if(ENDPOINT) return Promise.resolve(ENDPOINT);
  return fetch('endpoint.json',{cache:'no-store'}).then(function(r){return r.json();})
    .then(function(j){ ENDPOINT=(j.base||'').replace(/\/$/,''); return ENDPOINT; })
    .catch(function(){ return ''; });
}
self.addEventListener('fetch', function(e){
  var u = new URL(e.request.url);
  if(u.origin !== self.location.origin) return;      // only our own origin
  if(u.pathname.indexOf('/api/') !== 0) return;      // only API calls
  var site = (self.registration.scope || (self.location.origin + '/')).replace(/\/$/,'');
  e.respondWith(loadEndpoint().then(function(base){
    if(!base) return fetch(e.request);
    var h = new Headers(e.request.headers);
    h.set('X-VQ-Site', site);
    return fetch(base + u.pathname + u.search, {
      method: e.request.method,
      headers: h,
      body: (e.request.method==='GET'||e.request.method==='HEAD')?undefined:e.request.body,
      credentials: 'include',
      redirect: 'follow'
    });
  }));
});

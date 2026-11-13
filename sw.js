// vqwen service worker: the site's own link answers API calls.
// GitHub Pages serves only static files, so a POST there would 405. This
// worker forwards same-origin /api/* and /v1/* to the real machine (read from
// endpoint.json) and tags it with this site's origin, so the github link is a
// genuine API base URL for browser clients. Repoint the machine by editing
// endpoint.json - no HTML republish needed.
var BASE = null;
function endpoint(){
  if(BASE !== null) return Promise.resolve(BASE);
  return fetch('endpoint.json',{cache:'no-store'})
    .then(function(r){ return r.json(); })
    .then(function(j){ BASE=(j.base||'').replace(/\/$/,''); return BASE; })
    .catch(function(){ BASE=''; return BASE; });
}
function isApi(p){ return p.indexOf('/api/')===0 || p.indexOf('/v1/')===0; }
self.addEventListener('install', function(e){ self.skipWaiting(); });
self.addEventListener('activate', function(e){ e.waitUntil(self.clients.claim()); });
self.addEventListener('fetch', function(e){
  var u;
  try{ u = new URL(e.request.url); }catch(err){ return; }
  if(u.origin !== self.location.origin) return;   // only our own origin
  if(!isApi(u.pathname)) return;                  // only API calls
  e.respondWith(proxy(e.request, u));
});
async function proxy(req, u){
  var base = await endpoint();
  if(!base) return fetch(req);                    // no endpoint: pass through
  var scope = self.registration.scope || (self.location.origin + '/');
  var site = scope.replace(/\/$/,'');
  // Strip the site's own path prefix (/vqwen-machine) so both the page's
  // /api/* calls and a guide's <site>/v1/* calls map onto the machine.
  var prefix = new URL(scope).pathname.replace(/\/$/,'');
  var path = u.pathname;
  if(prefix && path.indexOf(prefix) === 0) path = path.slice(prefix.length) || '/';
  var url = base + path + u.search +
            (u.search ? '&' : '?') + 'vq_site=' + encodeURIComponent(site);
  var init = {method: req.method, headers: req.headers, mode: 'cors',
              credentials: 'include', redirect: 'follow'};
  if(req.method !== 'GET' && req.method !== 'HEAD'){
    try{ init.body = await req.blob(); }catch(err){}
  }
  return fetch(url, init);                        // response streams through
}

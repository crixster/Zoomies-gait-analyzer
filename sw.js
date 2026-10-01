const C='stridelab-v1';
self.addEventListener('install',e=>{e.waitUntil(caches.open(C).then(c=>c.addAll(['./','index.html','manifest.webmanifest','icon.svg'])));self.skipWaiting()});
self.addEventListener('activate',e=>e.waitUntil(clients.claim()));
self.addEventListener('fetch',e=>{
 if(e.request.method!=='GET'||e.request.headers.has('range'))return;
 e.respondWith(caches.match(e.request).then(h=>h||fetch(e.request).then(r=>{
  if(r.ok||r.type==='opaque'){const x=r.clone();caches.open(C).then(c=>c.put(e.request,x))}return r
 }).catch(()=>caches.match('index.html'))))});

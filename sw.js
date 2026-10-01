const C='zoomies-v3';
self.addEventListener('install',e=>{e.waitUntil(caches.open(C).then(c=>c.addAll(['./','index.html','manifest.webmanifest','favicon.png','apple-touch-icon.png','icon-192.png','icon-512.png','icon-maskable-512.png','splash-screen.gif','styles.css','app.js','splash.js'])));self.skipWaiting()});
self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(n=>n!==C).map(n=>caches.delete(n)))).then(()=>clients.claim())));
self.addEventListener('fetch',e=>{
 if(e.request.method!=='GET'||e.request.headers.has('range'))return;
 e.respondWith(caches.match(e.request).then(h=>h||fetch(e.request).then(r=>{
  if(r.ok||r.type==='opaque'){const x=r.clone();caches.open(C).then(c=>c.put(e.request,x))}return r
 }).catch(()=>caches.match('index.html'))))});

/*changes proposed by gemini

Curved Surface Ground Contact: The process() function replaced its flat positional baseline with vertical foot velocity (vy) and sub-frame interpolation. 
This accurately pinpoints ground contact on the sloped curve of the belt rather than assuming a flat road.

Shoulder Sway Tracking: A new S.rot array was initialized to track frame-by-frame 2D shoulder separation. 
This measures transverse rotation to catch side-to-side energy leaks during heavy efforts

High-Intensity Scoring Rubrics: The R scoring object was completely overhauled to evaluate faster, hybrid conditioning performance. 
The app now strictly punishes low cadences (dropping below 170 spm), excessive lateral shoulder sway, and heavy vertical bounce.

Metric Aggregation: The agg() function was updated to calculate and scale the new shoulder rotation metric (m.rot) alongside the existing gait and cadence data.

*/

// app.js
let PoseLandmarker, FilesetResolver;
const $ = s => document.querySelector(s);
const cv = $('#cv'), ctx = cv.getContext('2d'), vid = $('#vid');
let lm, mode = null, stream, facing = 'environment', running = false, rec, chunks = [], S, lastT = 0, fc = 0, live = { rows: [] };

const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN;
const next = f => mode !== 'demo' && vid.requestVideoFrameCallback ? vid.requestVideoFrameCallback(f) : requestAnimationFrame(f);
const say = t => $('#status').textContent = t || '';

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

/* ---------- Storage (IndexedDB) ---------- */
const db = new Promise(r => {
  const q = indexedDB.open('zoomiesdatabase', 1);
  q.onupgradeneeded = () => q.result.createObjectStore('s', { keyPath: 'id' });
  q.onsuccess = () => r(q.result);
});
const tx = async (m, f) => {
  const d = await db;
  return new Promise((ok, no) => {
    const t = d.transaction('s', m), q = f(t.objectStore('s'));
    t.oncomplete = () => ok(q.result);
    t.onerror = () => no(t.error);
  });
};

/* ---------- Pose Landmarker Loader ---------- */
async function loadModel() {
  try { await loadModel0(); }
  catch (e) {
    say('');
    throw new Error('Pose model could not load (offline or blocked network). Try demo runner or HTTPS.');
  }
}
async function loadModel0() {
  if (lm) return;
  say('Loading MediaPipe Pose Model…');
  if (!PoseLandmarker) ({ PoseLandmarker, FilesetResolver } = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs"));
  const fs = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const mk = d => PoseLandmarker.createFromOptions(fs, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", delegate: d },
    runningMode: "VIDEO", numPoses: 1
  });
  try { lm = await mk("GPU"); } catch (e) { lm = await mk("CPU"); }
  say('');
}

/* ---------- Gait Kinematics ---------- */
const newIv = () => ({ hmin: 1e9, hmax: -1e9, sm: 0, ovs: 0, k: 999, ks: 0 });
const newSession = () => { S = { steps: [], lean: [], elbow: [], rot: [], iv: newIv(), sg: 0, lt: -1, dir: 0, c: [{}, {}], gct: [] }; };

function ang(a, b, c) {
  const u = [a.x - b.x, a.y - b.y], v = [c.x - b.x, c.y - b.y];
  return Math.acos(Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1]) / ((Math.hypot(...u) * Math.hypot(...v)) || 1)))) * 180 / Math.PI;
}

function process(L, w, h, t) {
  const P = i => ({ x: L[i].x * w, y: L[i].y * h });
  const sd = $('#side').value, vis = o => [11, 23, 25, 27].reduce((s, i) => s + L[i + o].visibility, 0);
  const o = sd === 'L' ? 0 : sd === 'R' ? 1 : (vis(0) >= vis(1) ? 0 : 1);
  const sh = P(11 + o), hp = P(23 + o), kn = P(25 + o), an = P(27 + o);

  S.dir = S.dir * 0.95 + (L[31].x - L[29].x) + (L[32].x - L[30].x);
  const dir = S.dir < 0 ? -1 : 1;
  const tl = Math.hypot(sh.x - hp.x, sh.y - hp.y) || 1;
  const ll = (Math.hypot(hp.x - kn.x, hp.y - kn.y) + Math.hypot(kn.x - an.x, kn.y - an.y)) || 1;

  S.lean.push(Math.atan2((sh.x - hp.x) * dir, hp.y - sh.y) * 180 / Math.PI);
  S.elbow.push(ang(sh, P(13 + o), P(15 + o)));
  S.rot.push(Math.abs(L[11].x - L[12].x) * w / tl);

  const iv = S.iv, hy = (L[23].y + L[24].y) / 2 * h, hx = (L[23].x + L[24].x) / 2 * w;
  iv.hmin = Math.min(iv.hmin, hy); iv.hmax = Math.max(iv.hmax, hy);

  [0, 1].forEach(s => {
    const a = ang(P(23 + s), P(25 + s), P(27 + s));
    if (a < iv.k) { iv.k = a; iv.ks = s; }
  });

  const aL = P(27), aR = P(28), sep = (aL.x - aR.x) * dir, sg = sep > 0 ? 1 : -1;

  // Curved Belt Sub-frame Contact Detection
  const fy = [0, 1].map(s => L[29 + s].y * h);
  fy.forEach((y, s) => {
    const c = S.c[s];
    if (!c.py) { c.py = y; c.pt = t; return; }
    const vy = (y - c.py) / (t - c.pt);
    const isLow = y > hy + ll * 0.35;
    const threshold = ll * 1.5;
    const on = isLow && Math.abs(vy) < threshold;

    if (on && !c.on) {
      c.on = 1;
      const prevVy = c.pvy || 0;
      const ratio = Math.abs(vy - prevVy) > 0 ? (threshold - Math.abs(prevVy)) / Math.abs(vy - prevVy) : 0.5;
      c.t0 = c.pt + (t - c.pt) * Math.max(0, Math.min(1, ratio));
    } else if (!on && c.on) {
      c.on = 0;
      const d = (t - c.t0) * 1000;
      if (d > 80 && d < 450) S.gct.push(d);
    }
    c.pvy = vy; c.py = y; c.pt = t;
  });

  if (Math.abs(sep) > iv.sm) {
    iv.sm = Math.abs(sep);
    iv.ovs = ((sep > 0 ? aL.x : aR.x) - hx) * dir / ll;
  }

  if (!S.sg) S.sg = sg;
  else if (sg !== S.sg && iv.sm > 0.12 * ll && t - S.lt > 0.2) {
    S.steps.push({ t, ovs: iv.ovs, vo: (iv.hmax - iv.hmin) / tl * 0.29 * (+$('#ht').value || 175), k: iv.k, ks: iv.ks });
    S.iv = newIv(); S.lt = t; S.sg = sg;
  }
}

function agg(n) {
  const st = S.steps.slice(-n), f = -(n === Infinity ? 1e9 : 120);
  const m = {}, ks = [0, 1].map(s => avg(st.filter(x => x.ks === s).map(x => x.k)));
  m.cad = st.length > 2 ? 60 * (st.length - 1) / (st[st.length - 1].t - st[0].t) : NaN;
  m.lean = avg(S.lean.slice(f));
  m.ovs = avg(st.map(x => x.ovs)) * 100;
  m.vo = avg(st.map(x => x.vo));
  m.knee = avg(st.map(x => x.k));
  m.gct = avg(S.gct.slice(n === Infinity ? 0 : -16));
  m.elbow = avg(S.elbow.slice(f));
  m.rot = avg(S.rot.slice(f)) * 100;
  m.asym = Math.abs(ks[0] - ks[1]);
  return m;
}

/* ---------- Dynamic Scoring Criteria ---------- */
const G = 0, W = 1, B = 2, cl = ['good', 'warn', 'bad'];

const R = {
cad: ['Cadence', ' spm', 0, v => {
    const isEasy = $('#runMode')?.value === 'easy';
    if (isEasy) {
      return v < 152 ? [B, 'Very low cadence for easy running – focus on light turnover.'] :
             v < 160 ? [W, 'Slightly low turnover, but common for easy recovery paces.'] :
             v <= 172 ? [G, 'Optimal cadence for Zone 2 / easy recovery runs.'] :
             v <= 182 ? [W, 'High turnover for an easy run – verify your stride feels natural.'] :
             [B, 'Unnaturally rapid turnover for easy pacing.'];
    }
    return v < 170 ? [B, 'Low cadence for interval pacing – aim closer to 175-185 spm to reduce braking forces.'] :
           v < 175 ? [W, 'Slightly low cadence for threshold speeds.'] :
           v <= 195 ? [G, 'Cadence is dialed in for this pace.'] :
           [W, 'Extremely high cadence – ensure you are driving power, not just spinning wheels.'];
  }],
  
  lean: ['Trunk lean', '°', 1, v => {
    const isEasy = $('#runMode')?.value === 'easy';
    if (isEasy) {
      return v < 0 ? [B, 'Leaning backwards – maintain a tall, neutral posture.'] :
             v <= 8 ? [G, 'Relaxed, tall trunk posture ideal for easy efforts.'] :
             v <= 12 ? [W, 'Moderate forward lean – acceptable, but stay upright on easy days.'] :
             [B, 'Excessive lean for Zone 2 pacing – stand tall and relax the shoulders.'];
    }
    return v < 2 ? [B, 'Too upright for speed – lean forward from the ankles to drive the curved belt.'] :
           v < 5 ? [W, 'Slightly upright for speed – let gravity assist forward momentum.'] :
           v <= 14 ? [G, 'Excellent forward lean for high-intensity pacing.'] :
           v <= 18 ? [W, 'Heavy lean – check that you are not bending at the waist.'] :
           [B, 'Excessive lean – stay tall through the core.'];
  }],
  ovs: ['Foot reach', '% leg', 0, v => 
    v < 35 ? [G, 'Landing cleanly under center of mass.'] :
    v < 45 ? [W, 'Reaching slightly ahead – pull foot back under hips faster.'] :
    [B, 'Severe overstriding – highly inefficient on curved treadmills.']
  ],
  vo: ['Bounce', 'cm', 1, v => {
    const isEasy = $('#runMode')?.value === 'easy';
    const limit = isEasy ? 10 : 8;
    return v <= limit ? [G, 'Smooth, horizontal power distribution.'] :
           v <= limit + 3 ? [W, 'Moderate bounce – direct power backward into the belt, not upward.'] :
           [B, 'Excessive vertical oscillation – wasted energy.'];
  }],
  knee: ['Heel recovery', '° knee', 0, v => 
    v <= 100 ? [G, 'Excellent hamstring fold and heel recovery.'] :
    v <= 120 ? [W, 'Moderate heel lift – snap heel up faster after push-off.'] :
    [B, 'Swing leg is too straight – dragging through the recovery phase.']
  ],
  elbow: ['Arm angle', '°', 0, v => 
    v >= 70 && v <= 110 ? [G, 'Efficient arm carriage.'] :
    v > 110 && v <= 130 ? [W, 'Arms opening up – lock elbows closer to 90° for drive.'] :
    v < 70 && v >= 55 ? [W, 'Arms too tight – relax shoulders.'] :
    [B, 'Inefficient arm mechanics – adjust elbow angle.']
  ],
  rot: ['Shoulder Sway', '% ', 0, v => 
    v <= 18 ? [G, 'Quiet upper body, strong core stability.'] :
    v <= 30 ? [W, 'Noticeable shoulder rotation – brace core to prevent crossover.'] :
    [B, 'Heavy transverse rotation – bleeding energy laterally.']
  ],
  asym: ['L/R knee diff', '°', 0, v => 
    v <= 8 ? [G, 'Symmetrical leg mechanics.'] :
    v <= 15 ? [W, 'Mild asymmetry detected.'] :
    [B, 'Noticeable asymmetry – check for fatigue or compensation.']
  ],
  gct: ['Ground contact', ' ms', 0, v => {
    const isEasy = $('#runMode')?.value === 'easy';
    const tG = isEasy ? 260 : 220;
    const tW = isEasy ? 300 : 260;
    return v <= tG ? [G, 'Crisp, responsive ground contact.'] :
           v <= tW ? [W, 'Slightly long ground contact – focus on light foot pull.'] :
           [B, 'Heavy ground contact time – increase turnover rate.'];
  }]
};

function rows(n) {
  const m = agg(n);
  return Object.keys(R).map(k => {
    const [l, u, d, f] = R[k], v = m[k];
    if (!isFinite(v)) return { label: l, val: '–', c: '', msg: '' };
    const [c, msg] = f(v);
    return { label: l, val: v.toFixed(d) + u, c, msg };
  });
}

const score = r => {
  const x = r.filter(i => i.c !== '');
  return x.length ? Math.round(100 * avg(x.map(i => 1 - i.c / 2))) : 0;
};

/* ---------- Canvas Overlay & HUD ---------- */
const CN = [[11,12],[23,24],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[27,31],[28,30],[30,32],[28,32]];

function draw(L, w, h) {
  const k = w / 640, V = i => L[i].visibility > 0.4, P = i => [L[i].x * w, L[i].y * h];
  ctx.lineCap = 'round'; ctx.lineWidth = 4 * k; ctx.shadowColor = '#000'; ctx.shadowBlur = 6 * k;
  
  CN.forEach(([a, b]) => {
    if (!V(a) || !V(b)) return;
    ctx.strokeStyle = (a <= 12 && b <= 12) || (a === 23 && b === 24) ? '#eef3f7' : a % 2 ? '#ff8a5c' : '#5cc8ff';
    ctx.beginPath(); ctx.moveTo(...P(a)); ctx.lineTo(...P(b)); ctx.stroke();
  });
  
  ctx.shadowBlur = 0;
  for (let i = 11; i < 33; i++) if (V(i)) {
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(...P(i), 4 * k, 0, 7); ctx.fill();
  }
  if (V(0)) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(...P(0), 6 * k, 0, 7); ctx.fill(); }

  ctx.font = `bold ${13 * k}px sans-serif`;
  [0, 1].forEach(s => {
    if (V(23 + s) && V(25 + s) && V(27 + s)) {
      const a = ang({ x: P(23 + s)[0], y: P(23 + s)[1] }, { x: P(25 + s)[0], y: P(25 + s)[1] }, { x: P(27 + s)[0], y: P(27 + s)[1] });
      ctx.fillStyle = s ? '#5cc8ff' : '#ff8a5c';
      ctx.fillText(Math.round(a) + '°', P(25 + s)[0] + 8 * k, P(25 + s)[1]);
    }
  });
}

function hud(w, h) {
  if (!running) return;
  const k = w / 640, r = live.rows;
  ctx.fillStyle = '#000a'; ctx.fillRect(6 * k, 6 * k, 170 * k, r.length ? 74 * k : 24 * k);
  ctx.fillStyle = '#f87171'; ctx.beginPath(); ctx.arc(18 * k, 18 * k, 5 * k, 0, 7); ctx.fill();
  ctx.fillStyle = '#fff'; ctx.font = `bold ${13 * k}px sans-serif`;
  ctx.fillText('ANALYZING  Score ' + (r.length ? score(r) : '–'), 30 * k, 23 * k);
  
  r.slice(0, 4).forEach((x, i) => {
    ctx.fillStyle = ['#4ade80', '#fbbf24', '#f87171'][x.c] || '#fff';
    ctx.fillText(x.label + ': ' + x.val, 14 * k, (42 + i * 14) * k);
  });
}

/* ---------- Main Frame Execution ---------- */
function frame() {
  let L, w, h, ts = Math.max(performance.now(), lastT + 1); lastT = ts;
  if (mode === 'demo') {
    w = 640; h = 360;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    bg(ts / 1000); L = synth(ts / 1000);
  } else {
    if (vid.readyState < 2 || !lm) return;
    w = vid.videoWidth; h = vid.videoHeight;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    ctx.drawImage(vid, 0, 0);
    L = lm.detectForVideo(vid, ts).landmarks[0];
  }
  
  if (L) {
    draw(L, w, h);
    if (running) {
      process(L, w, h, mode === 'file' ? vid.currentTime : ts / 1000);
      if (++fc % 12 === 0) { live.rows = rows(8); paint(live.rows); }
    }
  }
  hud(w, h);
}

function bg(t) {
  ctx.fillStyle = '#233648'; ctx.fillRect(0, 0, 640, 360);
  ctx.fillStyle = '#7a4636'; ctx.fillRect(0, 342, 640, 18);
  ctx.fillStyle = '#d9c3b8';
  for (let i = 0; i < 9; i++) ctx.fillRect(((i * 80 - t * 220) % 640 + 640) % 640, 348, 40, 3);
}

function synth(t) {
  const asp = 0.5625, ph = 2 * Math.PI * t * (172 / 2 / 60), a = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));
  const set = (i, x, y) => { a[i].x = 0.5 + x * asp; a[i].y = y; };
  const hy = 0.52 + 0.012 * Math.cos(2 * ph), lean = 0.12, tl = 0.2, sy = hy - tl * Math.cos(lean), sx = tl * Math.sin(lean);
  set(23, 0, hy); set(24, 0, hy); set(11, sx, sy); set(12, sx, sy); set(0, sx + 0.05, sy - 0.07);
  [0, 1].forEach(s => {
    const p = ph + s * Math.PI, th = 0.32 * Math.sin(p), fl = 0.12 + 1.3 * Math.max(0, Math.cos(p)) ** 0.6;
    const kx = 0.2 * Math.sin(th), ky = hy + 0.2 * Math.cos(th), ax = kx + 0.2 * Math.sin(th - fl), ay = ky + 0.2 * Math.cos(th - fl);
    set(25 + s, kx, ky); set(27 + s, ax, ay); set(29 + s, ax - 0.02, ay + 0.02); set(31 + s, ax + 0.05, ay + 0.025);
    const b = -0.5 * Math.sin(p), ex = sx + 0.1 * Math.sin(b), ey = sy + 0.1 * Math.cos(b), wx = ex + 0.1 * Math.sin(b + 1.5), wy = ey + 0.1 * Math.cos(b + 1.5);
    set(13 + s, ex, ey); [15, 17, 19, 21].forEach(i => set(i + s, wx, wy));
  });
  return a;
}

let loopId = 0;
function tick(id) {
  if (id !== loopId || !mode) return;
  if (!(mode === 'file' && vid.paused)) frame();
  next(() => tick(id));
}

function paint(r) {
  $('#chips').innerHTML = r.map(x => `<div class="chip ${cl[x.c] || ''}"><b>${x.val}</b><small>${x.label}</small></div>`).join('');
  $('#cm').innerHTML = r.filter(x => x.msg).sort((a, b) => b.c - a.c).slice(0, 3).map(x => `<li class="${cl[x.c]}">${x.msg}</li>`).join('');
}

/* ---------- Sources ---------- */
async function openLive() {
  try {
    await loadModel(); stopSrc();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    vid.srcObject = stream; await vid.play(); mode = 'live'; ready();
  } catch (e) {
    say(/NotAllowed|NotFound|Permission/.test(e.name + e.message) ? 'Camera unavailable – allow camera access (HTTPS required).' : e.message);
  }
}

async function openFile(f) {
  await loadModel(); stopSrc(); vid.srcObject = null; vid.src = URL.createObjectURL(f); vid.loop = false;
  await new Promise(r => vid.onloadeddata = r);
  vid.currentTime = 0.05; await new Promise(r => vid.onseeked = r);
  mode = 'file'; ready(); frame();
}

function ready() { $('#bGo').disabled = false; $('#bFlip').hidden = mode !== 'live'; $('#scrub').hidden = mode !== 'file'; say(''); const id = ++loopId; next(() => tick(id)); }
function stopSrc() { running = false; mode = null; $('#scrub').hidden = true; stream && stream.getTracks().forEach(t => t.stop()); stream = null; vid.pause(); }

/* ---------- Recording & Recording Session ---------- */
function startRec() {
  const mt = ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm'].find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m));
  if (!mt) { say('Video saving not supported in this browser'); rec = null; return; }
  chunks = []; rec = new MediaRecorder(cv.captureStream(30), { mimeType: mt, videoBitsPerSecond: 4e6 }); rec.mt = mt.split(';')[0];
  rec.ondataavailable = e => e.data.size && chunks.push(e.data); rec.start(1000);
}
const stopRec = () => new Promise(r => { rec.onstop = () => r(new Blob(chunks, { type: rec.mt })); rec.stop(); });

async function start() {
  newSession(); live = { rows: [] }; fc = 0; running = true;
  $('#sum').innerHTML = ''; $('#bGo').textContent = 'Stop and save'; $('#bGo').classList.add('stop');
  if ($('#rec').value === '1') startRec(); else rec = null;
  if (mode === 'file') { vid.playbackRate = +$('#spd').value; if (vid.ended) vid.currentTime = 0; await vid.play(); }
}

async function finish() {
  if (!running) return;
  running = false; vid.pause();
  $('#bGo').textContent = 'Start analysis'; $('#bGo').classList.remove('stop');
  const r = rows(Infinity), ok = S.steps.length >= 4;
  const sum = { score: ok ? score(r) : 0, rows: r, steps: S.steps.length, ok, modeTarget: $('#runMode').value };
  let blob = null; if (rec && rec.state !== 'inactive') blob = await stopRec();
  $('#sum').innerHTML = sumHTML(sum); paint(r);
  if (blob && ok || blob) {
    await tx('readwrite', s => s.put({ id: Date.now(), date: new Date().toLocaleString(), mode, sum, blob }));
    say('Saved to History'); renderHist();
  }
  if (mode === 'file') { vid.currentTime = 0.05; }
}

function sumHTML(s) {
  if (!s.ok) return `<div class="card"><b>Not enough steps detected.</b><br>Keep full body in frame, film from side, and run at least 10 steps.</div>`;
  const tips = s.rows.filter(x => x.c > 0).sort((a, b) => b.c - a.c);
  return `<div class="card"><div class="score">${s.score}<small style="font-size:16px;color:var(--mu)"> / 100 form score (${s.modeTarget || 'tempo'})</small></div>
  <div class="note">${s.steps} steps analysed</div>
  <div class="grid">${s.rows.map(x => `<div class="chip ${cl[x.c] || ''}"><b>${x.val}</b><small>${x.label}</small></div>`).join('')}</div>
  <b>Coaching notes</b><ul>${(tips.length ? tips : [{ c: 0, msg: 'Form metrics look solid for this target effort!' }]).map(x => `<li class="${cl[x.c]}">${x.msg}</li>`).join('')}</ul></div>`;
}

/* ---------- History & Comparisons ---------- */
let hist = [];
async function renderHist() {
  hist = (await tx('readonly', s => s.getAll())).sort((a, b) => b.id - a.id);
  const h = $('#h');
  if (!hist.length) { h.innerHTML = '<div class="card">No saved analyses yet.</div>'; return; }
  h.innerHTML = '<div id="cmp"></div>' + (hist.length > 1 ? '<div class="row"><button id="bCmp" disabled>Compare 2 selected</button></div>' : '') +
  hist.map(x => {
    const u = x.blob ? URL.createObjectURL(x.blob) : '', ext = x.blob && x.blob.type.includes('mp4') ? 'mp4' : 'webm';
    return `<div class="card"><b>${x.date}</b> <span class="note">${x.sum.modeTarget || 'tempo'} pace · score ${x.sum.score}</span>
    ${u ? `<video class="rv" controls playsinline src="${u}"></video>` : ''}
    <div class="row" style="margin-top:6px;">${u ? `<a class="btn" href="${u}" download="zoomies-${x.id}.${ext}">Download video</a>` : ''}
    <button data-del="${x.id}">Delete</button>
    <label class="note" style="align-self:center"><input type="checkbox" data-cmp="${x.id}"> Compare</label></div>
    <details><summary>Readings and coaching</summary>${sumHTML(x.sum)}</details></div>`;
  }).join('');
}

function compare(a, b) {
  const v = x => x.blob ? `<video class="rv" muted playsinline src="${URL.createObjectURL(x.blob)}"></video>` : '',
  td = r => `<td class="${cl[r.c] || ''}">${r.val}</td>`;
  $('#cmp').innerHTML = `<div class="card"><b>Side by side</b><div class="two"><div><div class="note">Earlier · score ${a.sum.score}</div>${v(a)}</div><div><div class="note">Later · score ${b.sum.score}</div>${v(b)}</div></div>
  <div class="row"><button id="cp">Play both</button><button id="cx">Pause</button><button id="cc">Close</button></div>
  <table><tr><th></th><th>Earlier</th><th>Later</th></tr>${a.sum.rows.map(r => `<tr><td>${r.label}</td>${td(r)}${td(b.sum.rows.find(q => q.label === r.label) || { val: '–' })}</tr>`).join('')}</table></div>`;
}

$('#h').onclick = async e => {
  const t = e.target, id = t.dataset.del;
  if (id) { if (t.dataset.arm) { await tx('readwrite', s => s.delete(+id)); renderHist(); } else { t.dataset.arm = 1; t.textContent = 'Tap again to delete'; } }
  if (t.id === 'bCmp') { const c = [...document.querySelectorAll('[data-cmp]:checked')].map(i => hist.find(x => x.id == i.dataset.cmp)).sort((p, q) => p.id - q.id); compare(c[0], c[1]); $('#cmp').scrollIntoView(); }
  if (t.id === 'cp') document.querySelectorAll('#cmp video').forEach(v => { v.currentTime = 0; v.play(); });
  if (t.id === 'cx') document.querySelectorAll('#cmp video').forEach(v => v.pause());
  if (t.id === 'cc') $('#cmp').innerHTML = '';
};
$('#h').onchange = () => { const b = $('#bCmp'); if (b) b.disabled = document.querySelectorAll('[data-cmp]:checked').length !== 2; };

/* ---------- Event Wiring ---------- */
$('#bLive').onclick = openLive;
$('#bFlip').onclick = () => { facing = facing === 'environment' ? 'user' : 'environment'; openLive(); };
$('#file').onchange = e => e.target.files[0] && openFile(e.target.files[0]).catch(x => say(x.message));
$('#bGo').onclick = () => running ? finish() : start();
$('#spd').onchange = () => vid.playbackRate = +$('#spd').value;
vid.onended = () => mode === 'file' && finish();

document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.toggle('on', x === b));
  $('#a').hidden = b.dataset.t !== 'a'; $('#h').hidden = b.dataset.t !== 'h'; if (b.dataset.t === 'h') renderHist();
});

$('#bHelp').onclick = () => {
  const on = $('#help')?.hidden ?? true;
  if (!running) { stopSrc(); vid.srcObject = null; mode = 'demo'; ready(); }
};

renderHist();
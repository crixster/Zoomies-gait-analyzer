/* Zoomies gait analyzer – app.js

   Changes in this version:
   1. Run types: easy / tempo / threshold / intervals, each with its own reference ranges and its own personal baseline.
      The dropdown is built from this file (no index.html change needed) and remembers your choice.
   2. Timing: frames are stamped with the video's own clock (mediaTime / captureTime) instead of "whenever the phone got
      round to processing them". Uploaded videos slow themselves down so no frames are skipped; the live view drops to the
      lite pose model if the phone cannot keep up. The HUD shows the real analysis fps.
   3. Foot reach: furthest-forward ankle in the 0.25 s before contact. Touchdown detection lags the real contact by a few
      frames and the foot moves back ~10% of a leg per frame at running speed, which pushed readings negative.
      Ground-contact time now starts from a refined touchdown time.
   4. Far-leg handling: strides >30% off the median duration are treated as detection errors (they were inflating stride CV
      and L/R symmetry); a leg tracked below 60% visibility is left out of heel recovery / foot reach and the L/R values are
      hidden instead of guessed.
   5. Baseline comparison is one-sided for "lower is better" metrics (foot reach, bounce, contact time, L/R diff, heel
      recovery): being better than your usual is no longer flagged as a problem.
   6. Measurement confidence lists its seven ingredients and names the one holding it back.
   Earlier: foot-contact state machine, leg-length normalisation, per-leg L/R, stride curves, One Euro landmark filter,
   independent dimensions instead of one score.
   NOTE: all geometry is still 2D image-plane – film exactly side-on for valid angles. */

let PoseLandmarker, FilesetResolver;

const $ = s => document.querySelector(s);
const cv = $('#cv'), ctx = cv.getContext('2d'), vid = $('#vid');
let lm, mode = null, stream, facing = 'environment', running = false, rec, chunks = [], S, lastT = 0, fc = 0, live = { rows: [], conf: NaN };

const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN;
const av = a => avg(a.filter(Number.isFinite));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const pct = (a, p) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.floor(p * (s.length - 1))]; };
const med = a => pct(a, 0.5);
const stdev = a => { if (a.length < 3) return NaN; const m = avg(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
const rmse = (a, b) => a && b && a.length && a.length === b.length ? Math.sqrt(avg(a.map((x, i) => (x - b[i]) ** 2))) : NaN;
// The callback gets the video frame's own metadata (mediaTime = exact position in the file, captureTime = camera capture clock)
const next = f => mode !== 'demo' && vid.requestVideoFrameCallback
  ? vid.requestVideoFrameCallback((now, md) => f({ now, mt: md && md.mediaTime, ct: md && md.captureTime }))
  : requestAnimationFrame(() => f());
const say = t => $('#status').textContent = t || '';

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

/* ---------- Run types ---------- */

// Reference ranges per effort. These are coaching heuristics (starting points), NOT biomechanical truth –
// after 3 confident sessions of a run type your own baseline replaces them.
//   cad / lean : [bad below, warn below, good up to, warn up to (bad above)]
//   vo         : largest "good" bounce in cm        gct  : [good up to, warn up to] in ms
//   knee       : heel recovery = smallest knee angle, [good up to, warn up to] in degrees (lower = more heel fold)
const MODES = {
  easy:      { label: 'Easy',      cad: [152, 160, 172, 182], lean: [0, 0, 8, 12],  vo: 10, gct: [260, 300], knee: [110, 130] },
  tempo:     { label: 'Tempo',     cad: [162, 168, 188, 198], lean: [1, 4, 12, 16], vo: 9,  gct: [240, 280], knee: [105, 125] },
  threshold: { label: 'Threshold', cad: [168, 174, 192, 200], lean: [2, 5, 14, 18], vo: 8,  gct: [220, 260], knee: [100, 120] },
  intervals: { label: 'Intervals', cad: [172, 178, 196, 206], lean: [3, 6, 16, 20], vo: 8,  gct: [200, 240], knee: [92, 112] }
};
const MODE_KEY = 'zoomies.runMode';
const rmode = () => { const v = $('#runMode') && $('#runMode').value; return MODES[v] ? v : 'tempo'; };

// Build the run-type dropdown from MODES (creates it next to the side selector if index.html has none)
function initModes() {
  let sel = $('#runMode');
  if (!sel) {
    sel = document.createElement('select'); sel.id = 'runMode';
    const a = $('#side'); sel.className = a.className; a.insertAdjacentElement('afterend', sel);
  }
  let saved = null; try { saved = localStorage.getItem(MODE_KEY); } catch (e) {}
  const keep = sel.value;
  sel.innerHTML = Object.keys(MODES).map(k => `<option value="${k}">${MODES[k].label}</option>`).join('');
  sel.value = MODES[saved] ? saved : MODES[keep] ? keep : 'tempo';
  sel.onchange = () => { try { localStorage.setItem(MODE_KEY, sel.value); } catch (e) {} };
}
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

/* ---------- Personal baseline (per run type, Welford running mean/variance across sessions) ---------- */

const BK = 'zoomies.baseline.v3', BK_OLD = 'zoomies.baseline.v2';
// Smallest difference that is meaningful for each metric, so a very consistent runner doesn't get tiny-sigma false alarms.
const FLOOR = { cad: 3, lean: 1.5, ovs: 3, vo: 0.7, knee: 4, elbow: 6, rot: 3, asym: 2, gct: 12 };
const baseAll = () => { try { return JSON.parse(localStorage.getItem(BK)) || {}; } catch (e) { return {}; } };
const baseFor = m => baseAll()[m] || {};

// v2 -> v3: metrics whose measurement did not change keep their history. Foot reach, heel recovery, L/R diff and
// ground contact are measured differently now, so their baselines restart (general ranges are used for the first 3 sessions).
(function migrateBaseline() {
  try {
    if (localStorage.getItem(BK)) return;
    const old = JSON.parse(localStorage.getItem(BK_OLD) || 'null'); if (!old) return;
    const keep = ['cad', 'lean', 'vo', 'elbow', 'rot'], out = {};
    Object.keys(old).forEach(m => { out[m] = {}; keep.forEach(k => { if (old[m][k]) out[m][k] = old[m][k]; }); });
    localStorage.setItem(BK, JSON.stringify(out));
  } catch (e) {}
})();

function baseUpdate(m, vals) {
  const all = baseAll(), t = all[m] = all[m] || {};
  Object.keys(FLOOR).forEach(k => {
    const v = vals[k];
    if (!Number.isFinite(v)) return;
    const e = t[k] = t[k] || { n: 0, mean: 0, m2: 0 };
    e.n++; const d = v - e.mean; e.mean += d / e.n; e.m2 += d * (v - e.mean);
  });
  try { localStorage.setItem(BK, JSON.stringify(all)); } catch (e) {}
}

/* ---------- Pose Landmarker Loader ---------- */

let visionFs, modelKind = '';

const mkModel = (v, d) => PoseLandmarker.createFromOptions(visionFs, {
  baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${v}/float16/1/pose_landmarker_${v}.task`, delegate: d },
  runningMode: "VIDEO", numPoses: 1,
  minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5
});

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
  visionFs = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  // Full model is noticeably steadier on feet/ankles than lite; fall back if the device can't run it.
  for (const [v, d] of [['full', 'GPU'], ['lite', 'GPU'], ['lite', 'CPU']]) {
    try { lm = await mkModel(v, d); modelKind = v; break; } catch (e) {}
  }
  if (!lm) throw new Error('no pose model');
  say('');
}

// Live preview only: if the full model needs >40 ms per frame the analysis would run below ~25 fps, which hurts timing
// accuracy more than the lighter model hurts landmark accuracy. Swap to lite before the run starts.
let pm = [], swapping = false;
function watchSpeed(ms) {
  if (modelKind !== 'full' || swapping) return;
  pm.push(ms); if (pm.length < 45) return;
  const m = med(pm); pm = pm.slice(-30);
  if (m <= 40) return;
  swapping = true;
  mkModel('lite', 'GPU').then(n => {
    if (running) { n.close && n.close(); return; }
    const old = lm; lm = n; modelKind = 'lite'; old && old.close && old.close();
    say('Switched to the lighter pose model – this phone could not keep up at full quality.');
  }).catch(() => {}).finally(() => { swapping = false; });
}

/* ---------- Landmark filtering (One Euro, visibility-gated) ---------- */

let FL = {};

function oe(k, x, t, vis) {
  const s = FL[k] || (FL[k] = { x, dx: 0, t });
  const te = t - s.t;
  if (te <= 0) return s.x;
  if (te > 0.3) { s.x = x; s.dx = 0; s.t = t; return x; }          // gap / seek: restart
  if (vis < 0.25) { s.t = t; return s.x; }                           // unreliable point: hold
  const a = c => 1 / (1 + 1 / (2 * Math.PI * c * te));
  s.dx += a(1) * ((x - s.x) / te - s.dx);
  s.x += a(4 + 12 * Math.abs(s.dx)) * (x - s.x);                     // 4 Hz base cutoff, opens up with speed
  s.t = t;
  return s.x;
}

const smooth = (L, t) => L.map((p, i) => ({ x: oe(i + 'x', p.x, t, p.visibility), y: oe(i + 'y', p.y, t, p.visibility), visibility: p.visibility }));

/* ---------- Gait Kinematics ---------- */

// Stride channels (resampled to 0-100% of each leg's stride, touchdown to touchdown)
const CH = [['knee', 'Knee', '°'], ['hip', 'Hip', '°'], ['ankle', 'Ankle', '°'], ['foot', 'Foot reach', '% leg'], ['com', 'Vertical COM', '% leg'], ['trunk', 'Trunk', '°']];

const newIv = () => ({ hmin: 1e9, hmax: -1e9 });
const newC = () => ({ st: 0, sw: 0, tc: NaN, td: NaN, pt: NaN, pyn: NaN, pv: 0, buf: [], fh: [], lo: NaN, hi: NaN });

const newSession = () => {
  FL = {};
  S = {
    steps: [], str: [[], []], gct: [[], []], lean: [], elbow: [], rot: [], iv: newIv(), dir: 0, c: [newC(), newC()],
    vb: [], on: 0.2, off: 0.4, dts: [], lt: -1, tl: 0, ll: 0,
    fr: 0, det: 0, inf: 0, vsum: 0, vcnt: 0, nAtt: 0, base: baseFor(rmode())
  };
};

function ang(a, b, c) {
  const u = [a.x - b.x, a.y - b.y], v = [c.x - b.x, c.y - b.y];
  return Math.acos(Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1]) / ((Math.hypot(...u) * Math.hypot(...v)) || 1)))) * 180 / Math.PI;
}

// Linear resample of buf[key] onto 101 points between t0 and t1
function rs(buf, key, t0, t1) {
  const out = new Array(101); let j = 0;
  for (let i = 0; i <= 100; i++) {
    const tt = t0 + (t1 - t0) * i / 100;
    while (j < buf.length - 2 && buf[j + 1].t < tt) j++;
    const a = buf[j], b = buf[j + 1], u = b.t > a.t ? clamp((tt - a.t) / (b.t - a.t), 0, 1) : 0;
    out[i] = a[key] + (b[key] - a[key]) * u;
  }
  return out;
}
function process(L, w, h, t) {
  if (!(t > S.lt)) return;
  const dt = t - S.lt; S.lt = t;
  if (dt < 1) { S.dts.push(dt); if (S.dts.length > 150) S.dts.shift(); }
  S.det++;

  const P = i => ({ x: L[i].x * w, y: L[i].y * h });
  const side = $('#side').value, vis = o => [11, 23, 25, 27].reduce((s, i) => s + L[i + o].visibility, 0);
  const o = side === 'L' ? 0 : side === 'R' ? 1 : (vis(0) >= vis(1) ? 0 : 1);
  const sh = P(11 + o), hp = P(23 + o), kn = P(25 + o), an = P(27 + o);

  S.dir = S.dir * 0.95 + (L[31].x - L[29].x) + (L[32].x - L[30].x);
  const dir = S.dir < 0 ? -1 : 1;

  // Smoothed body scale (px) used for every normalisation
  const tl0 = Math.hypot(sh.x - hp.x, sh.y - hp.y) || 1;
  const ll0 = (Math.hypot(hp.x - kn.x, hp.y - kn.y) + Math.hypot(kn.x - an.x, kn.y - an.y)) || 1;
  S.tl = S.tl ? S.tl * 0.97 + tl0 * 0.03 : tl0;
  S.ll = S.ll ? S.ll * 0.97 + ll0 * 0.03 : ll0;
  const tl = S.tl, ll = S.ll;

  const lean = Math.atan2((sh.x - hp.x) * dir, hp.y - sh.y) * 180 / Math.PI;
  S.lean.push(lean);
  S.elbow.push(ang(sh, P(13 + o), P(15 + o)));
  S.rot.push(Math.abs(L[11].x - L[12].x) * w / tl);

  // Quality bookkeeping for the confidence model
  let vs = 0, inn = 1;
  for (let i = 23; i < 33; i++) {
    vs += L[i].visibility;
    if (L[i].x < 0.01 || L[i].x > 0.99 || L[i].y < 0.01 || L[i].y > 0.99) inn = 0;
  }
  S.vsum += vs / 10; S.vcnt++; S.inf += inn;

  const hcx = (L[23].x + L[24].x) / 2 * w, hcy = (L[23].y + L[24].y) / 2 * h, iv = S.iv;
  iv.hmin = Math.min(iv.hmin, hcy); iv.hmax = Math.max(iv.hmax, hcy);

  // Adaptive thresholds, refreshed every 15 frames from the runner's own data.
  // Velocity unit is leg-lengths per second (ll/s) so it is independent of camera distance, resolution and fps.
  if (S.det % 15 === 1) {
    const noise = S.vb.length > 60 ? pct(S.vb, 0.25) : 0.05;      // stance-phase foot speed noise floor
    S.on = clamp(3 * noise, 0.15, 0.6);                             // "foot is still" threshold
    S.off = Math.max(2 * S.on, 0.4);                                // "foot is lifting" threshold (hysteresis gap)
    S.c.forEach(c => { if (c.fh.length > 45) { c.lo = pct(c.fh, 0.1); c.hi = pct(c.fh, 0.9); } });
  }

  [0, 1].forEach(s => {
    const c = S.c[s], oth = S.c[1 - s];
    const H = P(23 + s), K = P(25 + s), A = P(27 + s), Hl = P(29 + s), T = P(31 + s);
    const fy = Math.max(Hl.y, T.y), fyn = fy / ll, fh = (fy - hcy) / ll;   // fyn: absolute foot y (ll units); fh: foot depth below hip (ll units)
    const lv = [23, 25, 27, 29, 31].reduce((a, i) => a + L[i + s].visibility, 0) / 5;
    const sm = {
      t, knee: ang(H, K, A), hip: Math.atan2((K.x - H.x) * dir, K.y - H.y) * 180 / Math.PI, ankle: ang(K, A, T),
      foot: (A.x - hcx) * dir / ll * 100, com: hcy / ll * 100, trunk: lean, vis: lv, fy: fyn
    };
    c.buf.push(sm); if (c.buf.length > 200) c.buf.shift();
    c.fh.push(fh); if (c.fh.length > 90) c.fh.shift();

    const dtl = t - c.pt, gap = !(dtl > 0 && dtl < 0.2);
    const v = gap ? 0 : 0.4 * c.pv + 0.6 * (fyn - c.pyn) / dtl;            // vertical foot velocity, ll/s (+ = moving down)
    if (!gap) { S.vb.push(Math.abs(v)); if (S.vb.length > 400) S.vb.shift(); }

    const rg = c.hi - c.lo, ok = c.fh.length > 45 && rg > 0.06;
    const lowIn = ok ? fh > c.lo + 0.7 * rg : fh > 0.8;                    // enter "foot is low" (strict)
    const lowOut = ok ? fh < c.lo + 0.5 * rg : fh < 0.7;                   // leave "foot is low" (lenient) -> hysteresis
    if (!c.st && lowOut) c.sw = 1;                                         // foot has swung since last contact

    if (!c.st) {
      // TOUCHDOWN: swung, low, nearly still, and respecting minimum stride / opposite-leg spacing
      if (c.sw && !gap && lowIn && Math.abs(v) < S.on && !(t - c.tc < 0.35) && !(t - oth.tc < 0.12)) {
        const pa = Math.abs(c.pv), ev = pa > S.on ? c.pt + (t - c.pt) * (pa - S.on) / (pa - Math.abs(v)) : t;

        if (isFinite(c.tc)) {                                              // close this leg's previous stride
          S.nAtt++;
          const dur = ev - c.tc, b = c.buf, bv = avg(b.map(x => x.vis));
          if (dur > 0.35 && dur < 1.6 && b.length > 5 && bv > 0.5) {
            const cu = {};
            CH.forEach(([k]) => { cu[k] = rs(b, k, c.tc, ev); });
            const mc = avg(cu.com); cu.com = cu.com.map(x => x - mc);
            S.str[s].push({ t0: c.tc, t1: ev, dur, vis: bv, kmin: Math.min(...b.map(x => x.knee)), cu });
          }
        }

        // Foot reach = furthest-forward ankle in the 0.25 s before contact. The velocity detector fires a few frames
        // AFTER the real touchdown and the foot is already travelling back by then, so reading the position at "ev" under-reports.
        const rb = c.buf.filter(x => x.t >= ev - 0.25);
        S.steps.push({ t: ev, s, ovs: Math.max(...rb.map(x => x.foot)), vis: avg(rb.map(x => x.vis)), vo: S.steps.length ? (iv.hmax - iv.hmin) / tl * 0.29 * (+$('#ht').value || 175) : NaN });

        // Refined touchdown (for ground-contact time): walk back while the foot was already within 0.05 leg-lengths of the ground (max 0.12 s)
        let td = ev;
        for (let i = c.buf.length - 1; i >= 0 && c.buf[i].t >= ev - 0.12; i--) { if (c.buf[i].fy < fyn - 0.05) break; td = Math.min(td, c.buf[i].t); }

        S.iv = newIv(); c.st = 1; c.sw = 0; c.tc = ev; c.td = td; c.buf = [sm];
      }
    } else {
      // TOE-OFF: foot rising fast or no longer low
      const age = t - c.tc;
      if (age > 0.6) { c.st = 0; c.sw = 0; }                               // lost lock (e.g. standing) – discard
      else if (age > 0.06 && (v < -S.off || lowOut)) {
        const ev = c.pv > -S.off && v < -S.off ? c.pt + (t - c.pt) * (c.pv + S.off) / (c.pv - v) : t;
        const d = (ev - (isFinite(c.td) ? c.td : c.tc)) * 1000;
        if (d > 80 && d < 450) S.gct[s].push(d);
        c.st = 0;
      }
    }
    c.pv = v; c.pyn = fyn; c.pt = t;
  });
}

/* ---------- Measurement layer ---------- */

const VIS_OK = 0.6;   // a leg tracked below this (mean landmark visibility over its strides) is not trusted for per-leg angles

// Strides whose duration is >30% off the median are a missed or extra touchdown, not a real stride – drop them
function gate(str) {
  const md = med(str.flatMap(a => a.map(x => x.dur)));
  return str.map(a => a.filter(x => !(Math.abs(x.dur - md) > 0.3 * md)));
}

function agg(n) {
  const all = n === Infinity, K = all ? 1e9 : Math.ceil(n / 2) + 1;
  const st = S.steps.slice(-(all ? 1e9 : n)), f = all ? 0 : -120;
  const sk = gate(S.str.map(a => a.slice(-K))), sg = S.gct.map(a => a.slice(-K));
  const fps = 1 / (med(S.dts) || 1), notes = [];
  const durs = sk.flatMap(a => a.map(x => x.dur));
  const lv = sk.map(a => avg(a.map(x => x.vis))), good = lv.map(v => v >= VIS_OK), ng = good.filter(Boolean).length;
  const use = s => ng ? good[s] : true;                                      // legs trusted for per-leg angle metrics
  const kn = sk.map(a => av(a.map(x => x.kmin)));
  const both = ng === 2 && sk[0].length > 1 && sk[1].length > 1;
  const si = (a, b) => Math.abs(a - b) / ((a + b) / 2) * 100;
  const m = {};

  m.cad = durs.length >= 2 ? 120 / med(durs) : NaN;                         // 2 steps per stride
  m.lean = av(S.lean.slice(f));
  m.ovs = med(st.filter(x => use(x.s)).map(x => x.ovs).filter(Number.isFinite));
  m.vo = av(st.map(x => x.vo));
  m.knee = av(sk.flatMap((a, s) => use(s) ? a : []).map(x => x.kmin));      // heel recovery: only from clearly tracked legs
  m.gct = fps >= 20 ? av(sg.flatMap((a, s) => use(s) ? a : [])) : NaN;      // contact time needs ~20+ fps
  m.elbow = av(S.elbow.slice(f));
  m.rot = av(S.rot.slice(f)) * 100;
  m.asym = both ? Math.abs(kn[0] - kn[1]) : NaN;                            // left-leg vs right-leg minimum knee angle

  m.cv = durs.length >= 4 ? stdev(durs) / avg(durs) * 100 : NaN;            // stride-time variability
  const sis = both ? [
    si(avg(sk[0].map(x => x.dur)), avg(sk[1].map(x => x.dur))),
    si(180 - kn[0], 180 - kn[1]),                                           // symmetry index on knee flexion
    sg[0].length && sg[1].length ? si(avg(sg[0]), avg(sg[1])) : NaN
  ].filter(Number.isFinite) : [];
  m.si = sis.length ? avg(sis) : NaN;

  // Say out loud what was withheld and why
  if (sk[0].length + sk[1].length >= 3) {
    if (ng === 1) {
      const bad = lv.findIndex(v => !(v >= VIS_OK)), q = lv[bad];
      notes.push(`One leg is hard to see (${Number.isFinite(q) ? Math.round(q * 100) + '% visibility' : 'barely tracked'}) – L/R knee diff and symmetry are hidden; heel recovery and foot reach use the clear leg only.`);
    } else if (ng === 0) notes.push('Neither leg is tracked clearly – treat all leg readings as rough.');
  }
  if (fps < 20 && S.det > 30) notes.push(`Analysis ran at only ${Math.round(fps)} fps – ground contact time is hidden (needs 20+).`);

  if (all) m.curves = [0, 1].map(s => {
    const o = {};
    CH.forEach(([k]) => { o[k] = sk[s].length ? Array.from({ length: 101 }, (_, i) => +avg(sk[s].map(x => x.cu[k][i])).toFixed(2)) : []; });
    return o;
  });
  m.nOk = sk[0].length + sk[1].length; m.fps = fps; m.notes = notes;
  return m;
}

// 0-100 confidence that the numbers above can be trusted (any weak factor drags it down hard).
// Returns the score AND its ingredients so the UI can say which one is holding it back.
function confInfo() {
  if (!S.det) return { score: 0, f: [], lim: null };
  const nOk = gate(S.str).reduce((s, a) => s + a.length, 0), fps = 1 / (med(S.dts) || 1), vis = S.vsum / S.vcnt;
  const rate = S.det / Math.max(S.fr, S.det), rot = av(S.rot);
  const fpsTip = mode === 'live'
    ? 'This phone cannot process frames fast enough live. Film normally, then use Upload video – uploads are analysed frame by frame.'
    : 'Under 30 fps. Film at 30–60 fps; uploads are slowed automatically so no frames are skipped, and a lower-resolution clip helps if it still cannot keep up.';
  const f = [
    ['Landmark visibility', vis, Math.round(vis * 100) + '% of leg points seen', 'Something is blocking a leg (treadmill rail or upright?). Put the camera at hip height with a clear line to both legs.'],
    ['Pose detected', rate, Math.round(rate * 100) + '% of frames', 'Keep your whole body in frame and well lit for the entire run.'],
    ['Legs inside frame', S.inf / S.det, Math.round(100 * S.inf / S.det) + '% of frames', 'Leave a margin around your head and feet.'],
    ['Stride count', nOk / 12, nOk + ' valid strides', 'Record at least 30 s of steady running.'],
    ['Stride detection', S.nAtt ? nOk / S.nAtt : 0, nOk + ' of ' + S.nAtt + ' strides accepted', 'Many strides were rejected: brighter light (less motion blur), steadier camera, clear view of both feet.'],
    ['Frame rate', (fps - 12) / 18, Math.round(fps) + ' fps', fpsTip],
    ['Side-on view', (0.95 - rot) / 0.55, Math.round(rot * 100) + '% shoulder spread', 'Camera must be exactly side-on so your shoulders overlap.']
  ].map(([name, v, info, tip]) => ({ name, f: clamp(Number.isFinite(v) ? v : 0, 0.02, 1), info, tip }));
  const score = 100 * Math.sqrt(Math.exp(avg(f.map(x => Math.log(x.f)))) * Math.min(...f.map(x => x.f)));
  const lim = f.reduce((a, b) => b.f < a.f ? b : a);
  return { score, f, lim: lim.f < 0.85 ? lim : null };
}
const conf = () => confInfo().score;

/* ---------- Interpretation layer ---------- */

const G = 0, W = 1, B = 2, cl = ['good', 'warn', 'bad'], LOWCONF = 40;
const M = () => MODES[rmode()];
// "Lower is better" metrics: against your own baseline only a reading ABOVE your usual is a problem
const UP = { ovs: 1, vo: 1, gct: 1, asym: 1, knee: 1 };

// Reference ranges: coaching heuristics, NOT biomechanical truth. Used until a personal baseline exists.
const R = {
  cad: ['Cadence', ' spm', 0, v => {
    const [bl, wl, gh, wh] = M().cad, n = M().label.toLowerCase();
    return v < bl ? [B, `Low cadence for ${n} running – aim for ${wl}+ spm; shorter, quicker steps cut braking forces.`] :
      v < wl ? [W, `Slightly low cadence for ${n} pace.`] :
      v <= gh ? [G, 'Cadence is dialed in for this effort.'] :
      v <= wh ? [W, 'Very high turnover – make sure you are driving, not just spinning.'] :
      [B, 'Unrealistically fast turnover – check the reading.'];
  }],
  lean: ['Trunk lean', '°', 1, v => {
    const [bl, wl, gh, wh] = M().lean;
    return v < bl ? [B, 'Leaning backwards – stay tall and lean from the ankles.'] :
      v < wl ? [W, 'Slightly upright for this pace – let gravity pull you forward from the ankles.'] :
      v <= gh ? [G, 'Good posture for this effort.'] :
      v <= wh ? [W, 'Heavy lean – check that you are not bending at the waist.'] :
      [B, 'Excessive lean – stay tall through the core.'];
  }],
  ovs: ['Foot reach', '% leg', 0, v =>
    v < -10 ? [W, 'Foot is landing well behind the hips, which is unusual – check the camera is level and side-on, then re-test.'] :
    v < 35 ? [G, 'Landing close to under your hips.'] :
    v < 45 ? [W, 'Reaching ahead of your hips – raise cadence ~5% and land with the foot under you.'] :
    [B, 'Severe overstriding – shorten the stride and raise cadence.']
  ],
  vo: ['Bounce', 'cm', 1, v => {
    const limit = M().vo;
    return v <= limit ? [G, 'Smooth, horizontal power distribution.'] :
      v <= limit + 3 ? [W, 'Moderate bounce – direct power backward into the belt, not upward.'] :
      [B, 'Excessive vertical oscillation – wasted energy.'];
  }],
  knee: ['Heel recovery', '° knee', 0, v => {
    const [g, w] = M().knee;
    return v <= g ? [G, 'Good heel fold – quick leg recovery.'] :
      v <= w ? [W, 'Heel is not folding much. It depends on pace, so compare like with like; think quick, relaxed feet and strengthen hamstrings and glutes.'] :
      [B, 'Swing leg stays straight – check the reading, then work on hip drive and a quicker heel recovery.'];
  }],
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
    const [g, w] = M().gct;
    return v <= g ? [G, 'Crisp, responsive ground contact.'] :
      v <= w ? [W, 'Slightly long ground contact – focus on light foot pull.'] :
      [B, 'Heavy ground contact time – increase turnover rate.'];
  }]
};

// Measurement -> rows. Colour/coaching come from the personal baseline (>=3 sessions per metric) or else the reference ranges,
// and are withheld entirely when measurement confidence is too low.
function analyze(n) {
  const m = agg(n), ci = confInfo(), c = ci.score, b = S.base || {};
  const rows = Object.keys(R).map(k => {
    const [l, u, d, f] = R[k], v = m[k];
    if (!isFinite(v)) return { k, label: l, val: '–', c: '', msg: '' };
    const val = v.toFixed(d) + u;
    if (c < LOWCONF) return { k, label: l, val, c: '', msg: '' };
    let [cc, msg] = f(v), z;
    const e = b[k];
    if (e && e.n >= 3 && !(k === 'ovs' && v < -10)) {                       // an implausible reading is reported as such, not compared
      z = (v - e.mean) / Math.max(Math.sqrt(e.m2 / (e.n - 1)), FLOOR[k]);
      const ref = cc !== G ? ' Reference range: ' + msg : '';
      const dev = UP[k] ? z : Math.abs(z);
      cc = dev < 1.5 ? G : dev < 2.5 ? W : B;
      msg = cc === G ? '' : `${l} is ${Math.abs(v - e.mean).toFixed(d)}${u.trim()} ${v > e.mean ? 'above' : 'below'} your usual for ${rmode()} runs (${z.toFixed(1)}σ).${ref}`;
    }
    return { k, label: l, val, c: cc, msg, z };
  });
  const rebuild = Math.min(...['ovs', 'knee', 'asym', 'gct'].map(k => b[k] ? b[k].n : 0));
  return { m, rows, conf: c, cf: ci, dims: dims(m, c), baseN: b.cad ? b.cad.n : 0, rebuild, notes: m.notes };
}

// Independent dimensions – deliberately NOT averaged into one score.
// Only dimensions that are NOT already a Measurements chip (vo / ovs / gct used to be repeated here).
function dims(m, c) {
  const band = (v, a, b, u) => isFinite(v) ? { val: v.toFixed(1) + u, c: v <= a ? G : v <= b ? W : B } : { val: '–', c: '' };
  return [
    { label: 'Measurement confidence', val: Math.round(c) + '%', c: c >= 75 ? G : c >= 50 ? W : B },
    { label: 'Stride consistency (CV)', ...band(m.cv, 3, 6, '%') },
    { label: 'L/R symmetry (index)', ...band(m.si, 5, 10, '%') }
  ].map((x, i) => i && c < LOWCONF ? { ...x, c: '' } : x);
}

const chips = r => r.map(x => `<div class="chip ${cl[x.c] || ''}"><b>${x.val}</b><small>${x.label}</small></div>`).join('');
const tag = s => s.conf != null ? `confidence ${Math.round(s.conf)}%` : `score ${s.score}`;

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
    // Green ring on a foot while the detector says it is in stance
    if (running && S && S.c[s].st && V(31 + s)) {
      ctx.strokeStyle = '#4ade80'; ctx.lineWidth = 3 * k;
      ctx.beginPath(); ctx.arc(...P(31 + s), 10 * k, 0, 7); ctx.stroke();
    }
  });
}

function hud(w, h) {
  if (!running) return;
  const k = w / 640, r = live.rows, fps = S && S.dts.length ? Math.round(1 / (med(S.dts) || 1)) : 0;
  ctx.fillStyle = '#000a'; ctx.fillRect(6 * k, 6 * k, 200 * k, r.length ? 74 * k : 24 * k);
  ctx.fillStyle = '#f87171'; ctx.beginPath(); ctx.arc(18 * k, 18 * k, 5 * k, 0, 7); ctx.fill();
  ctx.fillStyle = '#fff'; ctx.font = `bold ${13 * k}px sans-serif`;
  ctx.fillText('ANALYZING Conf ' + (r.length ? Math.round(live.conf) + '%' : '–') + (fps ? ' · ' + fps + 'fps' : ''), 30 * k, 23 * k);
  r.slice(0, 4).forEach((x, i) => {
    ctx.fillStyle = ['#4ade80', '#fbbf24', '#f87171'][x.c] || '#fff';
    ctx.fillText(x.label + ': ' + x.val, 14 * k, (42 + i * 14) * k);
  });
}

/* ---------- Main Frame Execution ---------- */

let pmark = 0, srcGaps = [], lastMt = NaN, paceN = 0;

// Uploaded video: slow playback so every source frame gets analysed (the pose model is slower than real time on most phones).
// Skipped frames were the main reason timing metrics (stride CV, contact time) and confidence were poor.
function pace(t, ms) {
  pmark = pmark ? pmark * 0.9 + ms * 0.1 : ms;                               // smoothed ms of work per frame
  if (Number.isFinite(lastMt) && t > lastMt && t - lastMt < 0.5) { srcGaps.push(t - lastMt); if (srcGaps.length > 90) srcGaps.shift(); }
  lastMt = t;
  if (++paceN % 15 || srcGaps.length < 15) return;
  const fi = pct(srcGaps, 0.1) * 1000, user = +$('#spd').value || 1;        // fi = source frame interval (ms of video time)
  const want = clamp(0.75 * fi / pmark, 0.15, user);                        // at rate r each frame gets fi / r ms of real time
  if (Math.abs(want - vid.playbackRate) > 0.1 * vid.playbackRate) vid.playbackRate = want;
}

function frame(meta) {
  let L, w, h, dMs = 0, ts = Math.max(performance.now(), lastT + 1); lastT = ts;
  const mf = meta || {}, t0 = performance.now();
  // Analysis clock: the frame's own timestamp where the browser provides one, otherwise processing time
  let tt = ts / 1000;
  if (mode === 'file') tt = Number.isFinite(mf.mt) ? mf.mt : vid.currentTime;
  else if (mode === 'live' && Number.isFinite(mf.ct) && mf.now - mf.ct >= 0 && mf.now - mf.ct < 500) tt = mf.ct / 1000;
  if (mode === 'demo') {
    w = 640; h = 360;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    bg(tt); L = synth(tt);
  } else {
    if (vid.readyState < 2 || !lm) return;
    w = vid.videoWidth; h = vid.videoHeight;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    ctx.drawImage(vid, 0, 0);
    const d0 = performance.now();
    L = lm.detectForVideo(vid, ts).landmarks[0];
    dMs = performance.now() - d0;
  }
  if (running && S) S.fr++;
  if (L) {
    L = smooth(L, tt);
    draw(L, w, h);
    if (running) {
      process(L, w, h, tt);
      if (++fc % 12 === 0) { live = analyze(8); paint(live); }
    }
  }
  hud(w, h);
  if (running && mode === 'file') pace(tt, performance.now() - t0);
  else if (!running && mode === 'live') watchSpeed(dMs);
}

function bg(t) {
  ctx.fillStyle = '#233648'; ctx.fillRect(0, 0, 640, 360);
  ctx.fillStyle = '#7a4636'; ctx.fillRect(0, 342, 640, 18);
  ctx.fillStyle = '#d9c3b8';
  for (let i = 0; i < 9; i++) ctx.fillRect(((i * 80 - t * 220) % 640 + 640) % 640, 348, 40, 3);
}

// Demo runner: 172 spm, ~34% stance, planted foot sliding back with the belt, fast heel recovery, 2-link leg IK
function synth(t) {
  const asp = 0.5625, ph = t * 172 / 120, a = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 1 }));
  const set = (i, x, y) => { a[i].x = 0.5 + x * asp; a[i].y = y; };
  const hy = 0.52 + 0.012 * Math.cos(4 * Math.PI * (ph - 0.17)), lean = 0.12, tl = 0.2, sy = hy - tl * Math.cos(lean), sx = tl * Math.sin(lean);
  set(23, 0, hy); set(24, 0, hy); set(11, sx, sy); set(12, sx, sy); set(0, sx + 0.05, sy - 0.07);
  [0, 1].forEach(s => {
    const u = ((ph + s * 0.5) % 1 + 1) % 1, p = 2 * Math.PI * u, st = 0.34, gy = 0.88;
    let ax, ay;
    if (u < st) { ax = 0.10 - 0.24 * u / st; ay = gy; }
    else { const w = (u - st) / (1 - st); ax = -0.14 + 0.24 * (0.5 - 0.5 * Math.cos(Math.PI * w)); ay = gy - 0.15 * Math.abs(Math.sin(Math.PI * w)) ** 0.6; }
    const dx = ax, dy = ay - hy, r = Math.hypot(dx, dy), k = Math.min(1, 0.398 / r), ex = dx * k, ey = dy * k, d = r * k, m = Math.sqrt(0.04 - d * d / 4);
    set(25 + s, ex / 2 + dy / r * m, hy + ey / 2 - dx / r * m); set(27 + s, ex, hy + ey);
    set(29 + s, ex - 0.02, hy + ey + 0.02); set(31 + s, ex + 0.05, hy + ey + 0.025);
    const b = -0.5 * Math.sin(p), elx = sx + 0.1 * Math.sin(b), ely = sy + 0.1 * Math.cos(b), wx = elx + 0.1 * Math.sin(b + 1.5), wy = ely + 0.1 * Math.cos(b + 1.5);
    set(13 + s, elx, ely); [15, 17, 19, 21].forEach(i => set(i + s, wx, wy));
  });
  return a;
}

let loopId = 0;

function tick(id, meta) {
  if (id !== loopId || !mode) return;
  if (!(mode === 'file' && vid.paused)) frame(meta);
  next(m => tick(id, m));
}

function paint(a) {
  $('#chips').innerHTML = chips(a.rows);
  const lim = a.cf && a.cf.lim;
  const warn = a.conf < 60 && S && S.steps.length >= 6 && lim ? [`<li class="warn">Confidence ${Math.round(a.conf)}% – held back by ${lim.name.toLowerCase()} (${lim.info}). ${lim.tip}</li>`] : [];
  $('#cm').innerHTML = warn.concat(a.rows.filter(x => x.msg).sort((x, y) => y.c - x.c).slice(0, 3).map(x => `<li class="${cl[x.c]}">${x.msg}</li>`)).join('');
}

/* ---------- Sources ---------- */

async function openLive() {
  try {
    await loadModel(); stopSrc();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, audio: false });
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

function ready() { FL = {}; pm = []; $('#bGo').disabled = false; $('#bFlip').hidden = mode !== 'live'; $('#scrub').hidden = mode !== 'file'; say(''); const id = ++loopId; next(m => tick(id, m)); }

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
  newSession(); live = { rows: [], conf: NaN }; fc = 0; running = true;
  const rm = $('#runMode'); if (rm) rm.disabled = true;                      // the baseline is per run type, so it is fixed for the session
  $('#sum').innerHTML = ''; $('#bGo').textContent = 'Stop and save'; $('#bGo').classList.add('stop');
  if ($('#rec').value === '1') startRec(); else rec = null;
  if (mode === 'file') { vid.playbackRate = +$('#spd').value; pmark = 0; srcGaps = []; lastMt = NaN; paceN = 0; if (vid.ended) vid.currentTime = 0; await vid.play(); }
}

async function finish() {
  if (!running) return;
  running = false; vid.pause();
  const rm = $('#runMode'); if (rm) rm.disabled = false;
  $('#bGo').textContent = 'Start analysis'; $('#bGo').classList.remove('stop');
  const an = analyze(Infinity), strides = an.m.nOk, ok = S.steps.length >= 4 && strides >= 2;
  const sum = { conf: an.conf, cf: an.cf, notes: an.notes, rebuild: an.rebuild, dims: an.dims, rows: an.rows, steps: S.steps.length, strides, ok, modeTarget: rmode(), curves: an.m.curves, baseN: an.baseN };
  if (ok && an.conf >= 60) baseUpdate(sum.modeTarget, an.m);     // only confident sessions shape the baseline
  let blob = null; if (rec && rec.state !== 'inactive') blob = await stopRec();
  // The summary card already holds the full readings + coaching, so clear the live panel instead of repainting it (was showing results twice)
  $('#sum').innerHTML = sumHTML(sum); $('#chips').innerHTML = $('#cm').innerHTML = '';
  if (blob) {
    await tx('readwrite', s => s.put({ id: Date.now(), date: new Date().toLocaleString(), mode, sum, blob }));
    say('Saved to History'); renderHist();
  }
  if (mode === 'file') { vid.currentTime = 0.05; }
}

function sumHTML(s) {
  if (!s.ok) return `<div class="card"><b>Not enough steps detected.</b><br>Keep full body in frame, film from side, and run at least 10 steps.</div>`;
  const tips = s.rows.filter(x => x.c > 0 && x.msg).sort((a, b) => b.c - a.c);
  const mt = s.modeTarget || 'tempo', nw = !!s.dims, lim = s.cf && s.cf.lim;
  const head = nw
    ? `<div class="score">${Math.round(s.conf)}<small style="font-size:16px;color:var(--mu)">% measurement confidence (${mt})</small></div>`
    : `<div class="score">${s.score}<small style="font-size:16px;color:var(--mu)"> / 100 form score (${mt})</small></div>`;
  const warn = nw && s.conf < 50 ? `<div class="note">Low confidence – treat readings as rough.${lim ? ` Biggest limit: ${lim.name.toLowerCase()} (${lim.info}).` : ' Film side-on, full body in frame, steady camera, 30+ fps.'}</div>` : '';
  const why = nw && s.cf && s.cf.f && s.cf.f.length
    ? `<details><summary class="note">Why ${Math.round(s.conf)}% confidence?</summary><ul>${s.cf.f.map(x => `<li class="${x.f >= 0.85 ? 'good' : x.f >= 0.6 ? 'warn' : 'bad'}">${x.name}: ${x.info}${x.f < 0.85 ? ' – ' + x.tip : ''}</li>`).join('')}</ul></details>` : '';
  const notes = (s.notes || []).map(t => `<div class="note">${t}</div>`).join('');
  const base = nw ? `<div class="note">${s.baseN >= 3 ? `Compared with your ${mt} baseline (${s.baseN} sessions).${s.rebuild != null && s.rebuild < 3 ? ` Foot reach, heel recovery, L/R diff and contact time are still rebuilding (${s.rebuild}/3) and use general ranges.` : ''}` : `Building your ${mt} baseline (${s.baseN}/3 sessions) – colours use general reference ranges until then.`}</div>` : '';
  return `<div class="card">${head}
<div class="note">${s.steps} steps${s.strides != null ? ' · ' + s.strides + ' valid strides' : ''} analysed</div>${warn}${why}${notes}${base}
${nw ? `<b>Form dimensions</b><div class="grid">${chips(s.dims.filter(x => !['Vertical motion', 'Braking / overstride', 'Ground contact'].includes(x.label)))}</div><b>Measurements</b>` : ''}<div class="grid">${chips(s.rows)}</div>
<b>Coaching notes</b><ul>${(tips.length ? tips : [{ c: 0, msg: 'No notable deviations for this effort.' }]).map(x => `<li class="${cl[x.c]}">${x.msg}</li>`).join('')}</ul></div>`;
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
      return `<div class="card"><b>${x.date}</b> <span class="note">${x.sum.modeTarget || 'tempo'} pace · ${tag(x.sum)}</span>
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
  // Stride-curve difference: RMSE between the two runs' mean 0-100% stride waveforms, per leg
  const ca = a.sum.curves, cb = b.sum.curves;
  const curves = ca && cb ? `<b>Stride-curve difference (RMSE over 0–100% stride)</b><table><tr><th></th><th>Left</th><th>Right</th></tr>${CH.map(([k, l, u]) => {
    const f = s => { const r = rmse(ca[s] && ca[s][k], cb[s] && cb[s][k]); return isFinite(r) ? r.toFixed(1) + ' ' + u.trim() : '–'; };
    return `<tr><td>${l}</td><td>${f(0)}</td><td>${f(1)}</td></tr>`;
  }).join('')}</table>` : '';
  $('#cmp').innerHTML = `<div class="card"><b>Side by side</b><div class="two"><div><div class="note">Earlier · ${tag(a.sum)}</div>${v(a)}</div><div><div class="note">Later · ${tag(b.sum)}</div>${v(b)}</div></div>
<div class="row"><button id="cp">Play both</button><button id="cx">Pause</button><button id="cc">Close</button></div>
<table><tr><th></th><th>Earlier</th><th>Later</th></tr>${a.sum.rows.map(r => `<tr><td>${r.label}</td>${td(r)}${td(b.sum.rows.find(q => q.label === r.label) || { val: '–' })}</tr>`).join('')}</table>${curves}</div>`;
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
  if (!running) { stopSrc(); vid.srcObject = null; mode = 'demo'; ready(); }
};

initModes();
renderHist();

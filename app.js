/* Zoomies gait analyzer – app.js

   Changes in this version (from the algorithm review notes):
   1. Gait events: steps now come from a per-leg foot-contact state machine (foot height + foot
      velocity, hysteresis, min stride/stance timing) instead of ankle-separation sign changes.
   2. Ground contact: foot velocity is normalised by leg length (leg-lengths/second) and the
      thresholds are derived from the runner's own foot-velocity distribution. Sub-frame interpolation kept.
   3. L/R: each leg keeps its own contacts, stance times and knee waveform; asymmetry is computed per leg.
   4. Layers separated: measurement (analyze/agg) vs interpretation (R reference ranges + personal baseline).
      The single 0-100 "form score" is gone; replaced by independent dimensions + measurement confidence.
   5. Stride normalisation: every valid stride is resampled to 0-100% (knee, hip, ankle, foot, vertical COM,
      trunk). Mean curves are saved and compared between runs with RMSE.
   6. Landmark filtering (visibility-gated One Euro filter) and full pose model (falls back to lite).
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
const next = f => mode !== 'demo' && vid.requestVideoFrameCallback ? vid.requestVideoFrameCallback(f) : requestAnimationFrame(f);
const say = t => $('#status').textContent = t || '';
const rmode = () => $('#runMode')?.value || 'tempo';

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

/* ---------- Personal baseline (per run type, Welford running mean/variance across sessions) ---------- */

const BK = 'zoomies.baseline.v2';
// Smallest difference that is meaningful for each metric, so a very consistent runner doesn't get tiny-sigma false alarms.
const FLOOR = { cad: 3, lean: 1.5, ovs: 3, vo: 0.7, knee: 4, elbow: 6, rot: 3, asym: 2, gct: 12 };
const baseAll = () => { try { return JSON.parse(localStorage.getItem(BK)) || {}; } catch (e) { return {}; } };
const baseFor = m => baseAll()[m] || {};

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
  const mk = (v, d) => PoseLandmarker.createFromOptions(fs, {
    baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${v}/float16/1/pose_landmarker_${v}.task`, delegate: d },
    runningMode: "VIDEO", numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5
  });
  // Full model is noticeably steadier on feet/ankles than lite; fall back if the device can't run it.
  for (const [v, d] of [['full', 'GPU'], ['lite', 'GPU'], ['lite', 'CPU']]) {
    try { lm = await mk(v, d); break; } catch (e) {}
  }
  if (!lm) throw new Error('no pose model');
  say('');
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
const newC = () => ({ st: 0, sw: 0, tc: NaN, pt: NaN, pyn: NaN, pv: 0, buf: [], fh: [], lo: NaN, hi: NaN });

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
      foot: (A.x - hcx) * dir / ll * 100, com: hcy / ll * 100, trunk: lean, vis: lv
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
          const dur = ev - c.tc, b = c.buf;
          if (dur > 0.35 && dur < 1.6 && b.length > 5 && avg(b.map(x => x.vis)) > 0.5) {
            const cu = {};
            CH.forEach(([k]) => { cu[k] = rs(b, k, c.tc, ev); });
            const mc = avg(cu.com); cu.com = cu.com.map(x => x - mc);
            S.str[s].push({ t0: c.tc, t1: ev, dur, kmin: Math.min(...b.map(x => x.knee)), cu });
          }
        }

        const rb = c.buf.filter(x => x.t >= ev - 0.1);                     // furthest reach just before contact
        S.steps.push({ t: ev, s, ovs: Math.max(...rb.map(x => x.foot)), vo: S.steps.length ? (iv.hmax - iv.hmin) / tl * 0.29 * (+$('#ht').value || 175) : NaN });
        S.iv = newIv(); c.st = 1; c.sw = 0; c.tc = ev; c.buf = [sm];
      }
    } else {
      // TOE-OFF: foot rising fast or no longer low
      const age = t - c.tc;
      if (age > 0.6) { c.st = 0; c.sw = 0; }                               // lost lock (e.g. standing) – discard
      else if (age > 0.06 && (v < -S.off || lowOut)) {
        const ev = c.pv > -S.off && v < -S.off ? c.pt + (t - c.pt) * (c.pv + S.off) / (c.pv - v) : t;
        const d = (ev - c.tc) * 1000;
        if (d > 80 && d < 450) S.gct[s].push(d);
        c.st = 0;
      }
    }
    c.pv = v; c.pyn = fyn; c.pt = t;
  });
}

/* ---------- Measurement layer ---------- */

function agg(n) {
  const all = n === Infinity, K = all ? 1e9 : Math.ceil(n / 2) + 1;
  const st = S.steps.slice(-(all ? 1e9 : n)), f = all ? 0 : -120;
  const sk = S.str.map(a => a.slice(-K)), sg = S.gct.map(a => a.slice(-K));
  const kn = sk.map(a => av(a.map(x => x.kmin))), durs = sk.flatMap(a => a.map(x => x.dur));
  const both = sk[0].length > 1 && sk[1].length > 1;
  const si = (a, b) => Math.abs(a - b) / ((a + b) / 2) * 100;
  const m = {};

  m.cad = durs.length >= 2 ? 120 / med(durs) : NaN;                         // 2 steps per stride
  m.lean = av(S.lean.slice(f));
  m.ovs = av(st.map(x => x.ovs));
  m.vo = av(st.map(x => x.vo));
  m.knee = av(sk.flat().map(x => x.kmin));
  m.gct = av(sg.flat());
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

  if (all) m.curves = [0, 1].map(s => {
    const o = {};
    CH.forEach(([k]) => { o[k] = S.str[s].length ? Array.from({ length: 101 }, (_, i) => +avg(S.str[s].map(x => x.cu[k][i])).toFixed(2)) : []; });
    return o;
  });
  return m;
}

// 0-100 confidence that the numbers above can be trusted (any weak factor drags it down hard)
function conf() {
  if (!S.det) return 0;
  const strides = S.str[0].length + S.str[1].length, fps = 1 / (med(S.dts) || 1);
  const f = [
    S.vsum / S.vcnt,                                  // landmark visibility
    S.det / Math.max(S.fr, S.det),                    // frames where a pose was found
    S.inf / S.det,                                    // legs fully inside the frame
    strides / 12,                                     // enough strides
    S.nAtt ? strides / S.nAtt : 0,                    // stride-segmentation acceptance
    (fps - 12) / 18,                                  // frame rate (30+ fps for GCT)
    (0.95 - av(S.rot)) / 0.55                         // side-on view (shoulders overlap)
  ].map(x => clamp(Number.isFinite(x) ? x : 0, 0.02, 1));
  return 100 * Math.sqrt(Math.exp(avg(f.map(Math.log))) * Math.min(...f));
}

/* ---------- Interpretation layer ---------- */

const G = 0, W = 1, B = 2, cl = ['good', 'warn', 'bad'], LOWCONF = 40;

// Reference ranges: coaching heuristics, NOT biomechanical truth. Used until a personal baseline exists.
const R = {
  cad: ['Cadence', ' spm', 0, v => {
    if (rmode() === 'easy') {
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
    if (rmode() === 'easy') {
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
    const limit = rmode() === 'easy' ? 10 : 8;
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
    const tG = rmode() === 'easy' ? 260 : 220, tW = rmode() === 'easy' ? 300 : 260;
    return v <= tG ? [G, 'Crisp, responsive ground contact.'] :
      v <= tW ? [W, 'Slightly long ground contact – focus on light foot pull.'] :
      [B, 'Heavy ground contact time – increase turnover rate.'];
  }]
};

// Measurement -> rows. Colour/coaching come from the personal baseline (>=3 sessions) or else the reference ranges,
// and are withheld entirely when measurement confidence is too low.
function analyze(n) {
  const m = agg(n), c = conf(), b = S.base || {};
  const rows = Object.keys(R).map(k => {
    const [l, u, d, f] = R[k], v = m[k];
    if (!isFinite(v)) return { k, label: l, val: '–', c: '', msg: '' };
    const val = v.toFixed(d) + u;
    if (c < LOWCONF) return { k, label: l, val, c: '', msg: '' };
    let [cc, msg] = f(v), z;
    const e = b[k];
    if (e && e.n >= 3) {
      z = (v - e.mean) / Math.max(Math.sqrt(e.m2 / (e.n - 1)), FLOOR[k]);
      const ref = cc !== G ? ' Reference range: ' + msg : '';
      cc = Math.abs(z) < 1.5 ? G : Math.abs(z) < 2.5 ? W : B;
      msg = cc === G ? '' : `${l} is ${Math.abs(v - e.mean).toFixed(d)}${u.trim()} ${v > e.mean ? 'above' : 'below'} your usual for ${rmode()} runs (${z.toFixed(1)}σ).${ref}`;
    }
    return { k, label: l, val, c: cc, msg, z };
  });
  return { m, rows, conf: c, dims: dims(m, c), baseN: b.cad ? b.cad.n : 0 };
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
  const k = w / 640, r = live.rows;
  ctx.fillStyle = '#000a'; ctx.fillRect(6 * k, 6 * k, 170 * k, r.length ? 74 * k : 24 * k);
  ctx.fillStyle = '#f87171'; ctx.beginPath(); ctx.arc(18 * k, 18 * k, 5 * k, 0, 7); ctx.fill();
  ctx.fillStyle = '#fff'; ctx.font = `bold ${13 * k}px sans-serif`;
  ctx.fillText('ANALYZING Conf ' + (r.length ? Math.round(live.conf) + '%' : '–'), 30 * k, 23 * k);
  r.slice(0, 4).forEach((x, i) => {
    ctx.fillStyle = ['#4ade80', '#fbbf24', '#f87171'][x.c] || '#fff';
    ctx.fillText(x.label + ': ' + x.val, 14 * k, (42 + i * 14) * k);
  });
}

/* ---------- Main Frame Execution ---------- */

function frame() {
  let L, w, h, ts = Math.max(performance.now(), lastT + 1); lastT = ts;
  const tt = mode === 'file' ? vid.currentTime : ts / 1000;
  if (mode === 'demo') {
    w = 640; h = 360;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    bg(tt); L = synth(tt);
  } else {
    if (vid.readyState < 2 || !lm) return;
    w = vid.videoWidth; h = vid.videoHeight;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    ctx.drawImage(vid, 0, 0);
    L = lm.detectForVideo(vid, ts).landmarks[0];
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

function tick(id) {
  if (id !== loopId || !mode) return;
  if (!(mode === 'file' && vid.paused)) frame();
  next(() => tick(id));
}

function paint(a) {
  $('#chips').innerHTML = chips(a.rows);
  const warn = a.conf < 50 && S && S.steps.length >= 6 ? [`<li class="warn">Measurement confidence ${Math.round(a.conf)}% – film side-on, full body in frame, steady camera.</li>`] : [];
  $('#cm').innerHTML = warn.concat(a.rows.filter(x => x.msg).sort((x, y) => y.c - x.c).slice(0, 3).map(x => `<li class="${cl[x.c]}">${x.msg}</li>`)).join('');
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

function ready() { FL = {}; $('#bGo').disabled = false; $('#bFlip').hidden = mode !== 'live'; $('#scrub').hidden = mode !== 'file'; say(''); const id = ++loopId; next(() => tick(id)); }

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
  $('#sum').innerHTML = ''; $('#bGo').textContent = 'Stop and save'; $('#bGo').classList.add('stop');
  if ($('#rec').value === '1') startRec(); else rec = null;
  if (mode === 'file') { vid.playbackRate = +$('#spd').value; if (vid.ended) vid.currentTime = 0; await vid.play(); }
}

async function finish() {
  if (!running) return;
  running = false; vid.pause();
  $('#bGo').textContent = 'Start analysis'; $('#bGo').classList.remove('stop');
  const an = analyze(Infinity), strides = S.str[0].length + S.str[1].length, ok = S.steps.length >= 4 && strides >= 2;
  const sum = { conf: an.conf, dims: an.dims, rows: an.rows, steps: S.steps.length, strides, ok, modeTarget: rmode(), curves: an.m.curves, baseN: an.baseN };
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
  const mt = s.modeTarget || 'tempo', nw = !!s.dims;
  const head = nw
    ? `<div class="score">${Math.round(s.conf)}<small style="font-size:16px;color:var(--mu)">% measurement confidence (${mt})</small></div>`
    : `<div class="score">${s.score}<small style="font-size:16px;color:var(--mu)"> / 100 form score (${mt})</small></div>`;
  const warn = nw && s.conf < 50 ? `<div class="note">Low confidence – treat readings as rough. Film side-on, full body in frame, steady camera, 30+ fps.</div>` : '';
  const base = nw ? `<div class="note">${s.baseN >= 3 ? `Compared with your ${mt} baseline (${s.baseN} sessions).` : `Building your ${mt} baseline (${s.baseN}/3 sessions) – colours use general reference ranges until then.`}</div>` : '';
  return `<div class="card">${head}
<div class="note">${s.steps} steps${s.strides != null ? ' · ' + s.strides + ' valid strides' : ''} analysed</div>${warn}${base}
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

renderHist();

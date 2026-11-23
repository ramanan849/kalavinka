'use strict';
/* P1 gold annotator: browser editor for Praat TextGrids (interval + point tiers). */
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const DEVA_FONT = '"Nirmala UI", Mangal, "Noto Sans Devanagari", "Segoe UI", sans-serif';
const MIN_DUR = 0.005;         // minimum interval length (s)
const LINK_EPS = 0.0015;       // boundaries closer than this are "the same" across tiers
const REQUIRED = ['words', 'syllables', 'vocables', 'singers'];
const VOCABLES = [['आ', '1'], ['ओ', '2'], ['ई', '3'], ['ए', '4'], ['ऊ', '5'], ['म्म', '6'], ['mel', '7']];
const SINGERS = [['A', '1'], ['B', '2'], ['C', '3'], ['D', '4'], ['E', '5'], ['AB', '6']];
const VOCABLE_SET = new Set(VOCABLES.map(v => v[0]));
const PALETTE = ['ं', 'ँ', '़', '्', 'ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ', 'ॉ', 'ः'];
const TIER_COL = { words: '#d6e4ff', syllables: '#dff2dc', vocables: '#fde2bd', singers: '#eee' };
const SING_COL = { A: '#8fb8ff', B: '#ffb38a', C: '#9be0a8', D: '#e4a3f0', E: '#f5d76e', F: '#8de0dc' };
const HINGLISH = new Set(('ek se mein main mai hai hain ki ke ka ko ye yeh wo woh de le na naa ho tu tum mera meri mere tera teri tere ' +
  'dil yahi maine leke hum toh to bhi jo kya kyun kyon aur pe par ne nahi nahin hi tha thi the raha rahi rahe sajna sanam pyar pyaar ' +
  'ishq jaan jana yaar yaara mujhe tujhe humko tumko apna apni koi kahin kabhi abhi sab baat dekho chal chalo aaja aaj kal ' +
  'zindagi dard sapna sapne aankhen ankhiyan mahi soniye dholna balle oye haan ha re o kuch kuchh jab ab phir saath hoon').split(' '));
const DEFAULT_SET = { src: 'v', pitchFloor: 70, pitchCeil: 1100, voicing: 0.75, specWin: 0.025, fmax: 5000, dyn: 50, formantMax: 5500, plo: 70, phi: 1000, unlabMin: 0.35, tierH: 44,
  grabPx: 6, snapPx: 8, dragPx: 4, nudgeMs: 5, bigNudgeMs: 20, wheelSpeed: 1, zoomSpeed: 1 };

// ------------------------------------------------------------------ state
const S = {
  songs: [], id: null, meta: null, tg: null, orig: null, origKeys: {}, D: 75, offset: 0,
  sr: 24000, x: { v: null, m: null }, buf: { v: null, m: null }, blocks: null,
  spec: null, specImg: null, pitch: null, form: null, voicedThr: 0, specKey: '',
  view: { a: 0, b: 10 }, zoomStack: [], cursor: 0, mark: null, sel: null, active: 0, selIv: -1, selB: null,
  undo: [], redo: [], savedJSON: null, savedAt: null, draftAt: null, hasCorrected: false,
  listened: null, ignored: new Set(), issues: [], seconds: 0, done: false, notes: '',
  editStart: null, editCk: false, dirtyDraw: true, loading: false, hover: null,
  blockTiers: new Set(['words', 'syllables']), queryDirty: true,
};
const opt = { follow: true, clickPlay: true, pitch: true, intensity: false, formants: false, allB: false, ghost: true, sylFollow: true, loop: false, autoSyl: true, rate: 1, blend: 0, vol: 1 };
S.set = Object.assign({}, DEFAULT_SET, JSON.parse(localStorage.getItem('p1set') || '{}'));

// ------------------------------------------------------------------ utils
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const fmt = t => (+t).toFixed(3);
const fmtMS = s => { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
function toast(msg, err) { const t = $('#toast'); t.textContent = msg; t.className = 'show' + (err ? ' err' : ''); clearTimeout(toast.h); toast.h = setTimeout(() => t.className = '', err ? 6000 : 2500); }
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
const tierIdx = name => S.tg ? S.tg.tiers.findIndex(t => t.name === name) : -1;
const T = name => { const i = tierIdx(name); return i >= 0 ? S.tg.tiers[i] : null; };
const isInt = t => t && t.kind === 'interval';
const ivTiers = () => S.tg.tiers.map((t, i) => i).filter(i => isInt(S.tg.tiers[i]));
function findIv(tier, t) { // interval containing t
  const it = tier.items; let lo = 0, hi = it.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (it[m].a <= t) lo = m; else hi = m - 1; }
  return lo;
}
function findPt(tier, t, eps) { // nearest point within eps, or -1
  let best = -1, bd = eps;
  tier.items.forEach((p, j) => { const d = Math.abs(p.p - t); if (d <= bd) { bd = d; best = j; } });
  return best;
}
function boundaryNear(tier, t, eps) { // k (items[k].a is the boundary) or -1
  const it = tier.items, i = findIv(tier, t);
  let best = -1, bd = eps;
  for (const k of [i, i + 1]) if (k > 0 && k < it.length) { const d = Math.abs(it[k].a - t); if (d <= bd) { bd = d; best = k; } }
  return best;
}
const stripQ = w => w.replace(/\?/g, '').trim();
const noteName = f => { const n = Math.round(12 * Math.log2(f / 440)) + 69; return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][((n % 12) + 12) % 12] + (Math.floor(n / 12) - 1); };
const ivKey = iv => `${iv.a.toFixed(4)}|${iv.b.toFixed(4)}|${iv.t}`;
const ptKey = p => `${p.p.toFixed(4)}|${p.t}`;
function download(name, data, type) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([data], { type })); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

// server <-> internal (point items are {p: time, t: mark} internally)
function fromServer(tg) {
  return { xmin: tg.xmin, xmax: tg.xmax, tiers: tg.tiers.map(t => t.kind === 'point'
    ? { name: t.name, kind: 'point', xmin: t.xmin, xmax: t.xmax, items: t.items.map(i => ({ p: i.t, t: i.mark })) }
    : { name: t.name, kind: 'interval', xmin: t.xmin, xmax: t.xmax, items: t.items.map(i => ({ a: i.a, b: i.b, t: i.t })) }) };
}
function toServer(tg) {
  return { xmin: tg.xmin, xmax: tg.xmax, tiers: tg.tiers.map(t => t.kind === 'point'
    ? { name: t.name, kind: 'point', xmin: t.xmin, xmax: t.xmax, items: t.items.map(i => ({ t: i.p, mark: i.t })) }
    : { name: t.name, kind: 'interval', xmin: t.xmin, xmax: t.xmax, items: t.items }) };
}
function textGridText(tg) { // Praat long text format
  const n = x => { const s = (+x).toFixed(6); return +s === 0 ? '0' : s; }, q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', '', `xmin = ${n(tg.xmin)} `, `xmax = ${n(tg.xmax)} `, 'tiers? <exists> ', `size = ${tg.tiers.length} `, 'item []: '];
  tg.tiers.forEach((t, i) => {
    L.push(`    item [${i + 1}]:`);
    if (t.kind === 'interval') {
      L.push('        class = "IntervalTier" ', `        name = ${q(t.name)} `, `        xmin = ${n(t.xmin)} `, `        xmax = ${n(t.xmax)} `, `        intervals: size = ${t.items.length} `);
      t.items.forEach((it, j) => L.push(`        intervals [${j + 1}]:`, `            xmin = ${n(it.a)} `, `            xmax = ${n(it.b)} `, `            text = ${q(it.t)} `));
    } else {
      L.push('        class = "TextTier" ', `        name = ${q(t.name)} `, `        xmin = ${n(t.xmin)} `, `        xmax = ${n(t.xmax)} `, `        points: size = ${t.items.length} `);
      t.items.forEach((it, j) => L.push(`        points [${j + 1}]:`, `            number = ${n(it.p)} `, `            mark = ${q(it.t)} `));
    }
  });
  return L.join('\n') + '\n';
}

// ------------------------------------------------------------------ undo / change tracking
function checkpoint() { S.undo.push(JSON.stringify(S.tg.tiers)); if (S.undo.length > 400) S.undo.shift(); S.redo = []; }
function restore(from, to) {
  if (!from.length) { toast('Nothing to ' + (from === S.undo ? 'undo' : 'redo')); return; }
  to.push(JSON.stringify(S.tg.tiers));
  S.tg.tiers = JSON.parse(from.pop());
  S.active = clamp(S.active, 0, S.tg.tiers.length - 1);
  const t = S.tg.tiers[S.active]; if (S.selIv >= t.items.length) S.selIv = t.items.length - 1;
  S.selB = null; changed(); updateEditor(); renderBlockTiers();
}
let draftTimer = null, checkTimer = null;
function changed() {
  S.dirtyDraw = true; S.queryDirty = true;
  clearTimeout(draftTimer); draftTimer = setTimeout(saveDraft, 1500);
  clearTimeout(checkTimer); checkTimer = setTimeout(runChecks, 250);
  updateStatus();
}
const isDirty = () => S.tg && JSON.stringify(S.tg.tiers) !== S.savedJSON;
async function saveDraft() {
  if (!S.id || !S.tg) return;
  clearTimeout(draftTimer);
  if (!isDirty()) { if (S.draftAt) { await api('/api/discard_draft/' + S.id, {}).catch(() => { }); S.draftAt = null; } updateStatus(); return; }
  try { const r = await api('/api/draft/' + S.id, { fmt: 2, tg: S.tg, at: Date.now() / 1000 }); S.draftAt = r.at; updateStatus(); }
  catch (e) { toast('Draft autosave failed: ' + e.message, true); }
}
function updateStatus() {
  const c = $('#save-status');
  if (!S.tg) { c.textContent = '—'; c.className = 'chip'; return; }
  const tm = t => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (!S.hasCorrected && S.orig && JSON.stringify(S.tg.tiers) === JSON.stringify(S.orig.tiers)) { c.textContent = 'Not saved to corrected/ yet (unchanged prediction)'; c.className = 'chip'; }
  else if (isDirty()) { c.textContent = 'Unsaved changes' + (S.draftAt ? ' · draft kept ' + tm(S.draftAt) : ''); c.className = 'chip dirty'; }
  else if (S.hasCorrected) { c.textContent = '✔ Saved ' + tm(S.savedAt); c.className = 'chip saved'; }
  else { c.textContent = 'Not saved to corrected/ yet'; c.className = 'chip dirty'; }
}

// ------------------------------------------------------------------ songs
async function loadSongs() {
  const j = await api('/api/songs');
  S.songs = j.songs; $('#corr-path').textContent = j.corrected_dir;
  renderSongList();
}
function renderSongList() {
  const L = $('#song-list'); L.innerHTML = '';
  let nSaved = 0, nDone = 0;
  for (const s of S.songs) {
    if (s.saved_at) nSaved++; if (s.done) nDone++;
    const d = document.createElement('div');
    d.className = 'song' + (s.id === S.id ? ' cur' : '');
    const st = s.done ? '✅' : s.draft_newer ? '✏️' : s.saved_at ? '💾' : '○';
    const tip = s.done ? 'done' : s.draft_newer ? 'unsaved draft' : s.saved_at ? 'saved, not marked done' : 'not started';
    const bits = [];
    if (s.listened != null) bits.push('heard ' + s.listened + '%');
    if (s.issues != null) bits.push(s.issues + ' checks');
    if (s.seconds) bits.push(fmtMS(s.seconds));
    d.innerHTML = `<span title="${tip}">${st}</span><span class="t"></span><span class="m">${bits.join(' · ') || tip}</span>`;
    d.querySelector('.t').textContent = `${s.year} ${s.title}`; d.querySelector('.t').title = s.id;
    d.onclick = () => openSong(s.id);
    L.appendChild(d);
  }
  $('#progress-total').textContent = `${nDone}/${S.songs.length} done · ${nSaved} saved`;
}

function parseWav(ab) {
  const dv = new DataView(ab); let p = 12, f = null, data = null;
  while (p + 8 <= dv.byteLength) {
    const id = String.fromCharCode(dv.getUint8(p), dv.getUint8(p + 1), dv.getUint8(p + 2), dv.getUint8(p + 3));
    const sz = dv.getUint32(p + 4, true);
    if (id === 'fmt ') f = { tag: dv.getUint16(p + 8, true), ch: dv.getUint16(p + 10, true), sr: dv.getUint32(p + 12, true), bits: dv.getUint16(p + 22, true) };
    if (id === 'data') data = { off: p + 8, len: Math.min(sz, dv.byteLength - p - 8) };
    p += 8 + sz + (sz & 1);
  }
  if (!f || !data) throw new Error('unsupported wav');
  const bps = f.bits / 8, n = Math.floor(data.len / bps / f.ch), x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < f.ch; c++) {
      const o = data.off + (i * f.ch + c) * bps;
      s += f.bits === 16 ? dv.getInt16(o, true) / 32768 : f.bits === 32 && f.tag === 3 ? dv.getFloat32(o, true) : f.bits === 32 ? dv.getInt32(o, true) / 2147483648 : f.bits === 24 ? ((dv.getUint8(o) | dv.getUint8(o + 1) << 8 | dv.getInt8(o + 2) << 16) / 8388608) : (dv.getUint8(o) - 128) / 128;
    }
    x[i] = s / f.ch;
  }
  return { x, sr: f.sr };
}

async function openSong(id) {
  if (S.loading) return;
  if (S.id && S.tg) { if (document.activeElement === txt) commitText(txt.value, false); await saveDraft(); await saveState(); }
  stop();
  S.loading = true; $('#loading').classList.remove('hidden'); $('#loading').textContent = 'Loading…';
  try {
    const j = await api('/api/song/' + encodeURIComponent(id));
    S.id = id; history.replaceState(null, '', '#' + encodeURIComponent(id));
    S.meta = j.meta; S.orig = fromServer(j.original);
    S.hasCorrected = !!j.corrected; S.savedAt = j.corrected_at;
    if (j.corrected_error) toast('corrected/ file could not be read: ' + j.corrected_error, true);
    const base = j.corrected ? fromServer(j.corrected) : S.orig;
    let tg = base, src = j.corrected ? 'your saved file in corrected/' : 'pipeline prediction';
    S.savedJSON = j.corrected ? JSON.stringify(base.tiers) : null;
    if (j.draft && j.draft.tg && (!j.corrected_at || j.draft_at > j.corrected_at + 0.5)) {
      tg = j.draft.fmt === 2 ? j.draft.tg : fromServer(j.draft.tg); src = 'autosaved draft (not yet saved)';
      setTimeout(() => toast('Restored your unsaved draft from ' + new Date(j.draft_at * 1000).toLocaleString()), 300);
    }
    S.draftAt = j.draft ? j.draft_at : null;
    S.tg = JSON.parse(JSON.stringify(tg));
    if (S.savedJSON === null) S.savedJSON = '__none__';
    S.D = S.tg.xmax - S.tg.xmin;
    S.origKeys = {};
    for (const t of S.orig.tiers) S.origKeys[t.name] = new Set(t.items.map(t.kind === 'point' ? ptKey : ivKey));
    S.undo = []; S.redo = []; S.sel = null; S.selB = null; S.selIv = -1; S.cursor = 0; S.mark = null; S.zoomStack = [];
    S.active = Math.max(0, tierIdx('words'));
    const st = j.state || {};
    S.ignored = new Set(st.ignored || []); S.seconds = st.seconds || 0; S.done = !!st.done; S.notes = st.notes || '';
    $('#chk-done').checked = S.done; $('#notes').value = S.notes;
    $('#song-title').textContent = S.meta.title || id;
    S.offset = S.meta.start_frame && S.meta.fps ? S.meta.start_frame / S.meta.fps : 0;
    $('#song-sub').textContent = `${id} · window ${fmtMS(S.offset)}–${fmtMS(S.offset + S.D)} of the song · loaded from ${src}`;
    renderSongList(); renderBlockTiers(); fillFindTiers();
    $('#loading').textContent = 'Loading audio…';
    const [va, ma] = await Promise.all(['vocals', 'mix'].map(k => fetch(`/audio/${encodeURIComponent(id)}/${k}.wav`).then(r => { if (!r.ok) throw new Error(k + '.wav: HTTP ' + r.status); return r.arrayBuffer(); })));
    const v = parseWav(va), m = parseWav(ma);
    S.x.v = v.x; S.x.m = m.x; S.sr = v.sr;
    const ctx = audioCtx();
    S.buf.v = ctx.createBuffer(1, v.x.length, v.sr); S.buf.v.copyToChannel(v.x, 0);
    S.buf.m = ctx.createBuffer(1, m.x.length, m.sr); S.buf.m.copyToChannel(m.x, 0);
    setupElements([va, ma]);
    S.listened = new Uint8Array(Math.ceil(S.D * 10));
    if (st.listened) for (let i = 0; i < Math.min(st.listened.length, S.listened.length); i++) S.listened[i] = st.listened[i] === '1' ? 1 : 0;
    S.view = { a: 0, b: Math.min(S.D, 12) };
    S.specKey = '';
    await analyse();
    selectInterval(S.active, 0, false);
    runChecks(); updateStatus(); updateEditor();
  } catch (e) { toast('Could not load: ' + e.message, true); console.error(e); }
  S.loading = false; $('#loading').classList.add('hidden'); S.dirtyDraw = true;
}

const srcX = () => S.set.src === 'm' ? S.x.m : S.x.v;
function buildBlocks() {
  const x = srcX(), B = 32, n = Math.ceil(x.length / B), mn = new Float32Array(n), mx = new Float32Array(n);
  for (let i = 0; i < n; i++) { let a = 1, b = -1; for (let j = i * B; j < Math.min(x.length, (i + 1) * B); j++) { const v = x[j]; if (v < a) a = v; if (v > b) b = v; } mn[i] = a; mx[i] = b; }
  let peak = 0; for (let i = 0; i < x.length; i += 5) peak = Math.max(peak, Math.abs(x[i]));
  S.blocks = { B, mn, mx, peak: peak || 1 };
}
let worker = null;
function analyse() {
  const key = [S.id, S.set.src, S.set.pitchFloor, S.set.pitchCeil, S.set.specWin, S.set.formantMax].join('|');
  if (key === S.specKey) { buildSpecImg(); S.dirtyDraw = true; return Promise.resolve(); }
  $('#loading').classList.remove('hidden'); $('#loading').textContent = 'Analysing (spectrogram, pitch, intensity, formants)…';
  buildBlocks();
  return new Promise(res => {
    if (worker) worker.terminate();
    worker = new Worker('dsp.js');
    worker.onmessage = e => {
      S.spec = e.data.spec; S.pitch = e.data.pitch; S.form = e.data.form; S.specKey = key;
      const r = Array.from(S.pitch.rms).sort((a, b) => a - b);
      S.voicedThr = r[Math.floor(r.length * 0.95)] * 0.1;
      buildSpecImg(); S.dirtyDraw = true; S.queryDirty = true; worker.terminate(); worker = null;
      if (!S.loading) $('#loading').classList.add('hidden');
      res();
    };
    worker.onerror = e => { console.error(e); toast('analysis failed', true); res(); };
    worker.postMessage({ x: srcX(), sr: S.sr, set: S.set });
  });
}
function buildSpecImg() {
  if (!S.spec) return;
  const { data, nf, nb, max } = S.spec, dyn = S.set.dyn;
  const c = document.createElement('canvas'); c.width = nf; c.height = nb;
  const g = c.getContext('2d'), img = g.createImageData(nf, nb), d = img.data;
  for (let f = 0; f < nf; f++) for (let b = 0; b < nb; b++) {
    let v = (data[f * nb + b] - (max - dyn)) / dyn; v = v < 0 ? 0 : v > 1 ? 1 : v;
    const g8 = 255 - Math.round(235 * Math.pow(v, 1.8)), o = ((nb - 1 - b) * nf + f) * 4;
    d[o] = g8; d[o + 1] = g8; d[o + 2] = g8; d[o + 3] = 255;
  }
  g.putImageData(img, 0, 0); S.specImg = c;
}
const isVoiced = f => S.pitch.f0[f] > 0 && S.pitch.clar[f] > S.set.voicing && S.pitch.rms[f] > S.voicedThr;
const frameAt = t => Math.floor(t / (S.pitch.hop / S.sr));

// ------------------------------------------------------------------ audio
let AC = null;
function audioCtx() { if (!AC) AC = new (window.AudioContext || window.webkitAudioContext)(); return AC; }
const P = { playing: false, pending: false, a: 0, b: 0, t0: 0, srcs: [], gains: null, mode: 'wa', els: null, elGains: null, raf: 0, kind: '', tok: 0 };
// Slow playback (pitch kept) uses <audio> elements. They are built from the bytes we already downloaded
// (blob URLs), so they are fully seekable at once; no second download, no "plays from 0" after a seek.
function setupElements(bytes) {
  const ctx = audioCtx();
  if (P.els) for (const e of P.els) { e.pause(); const u = e.src; e.removeAttribute('src'); e.load(); if (u.startsWith('blob:')) URL.revokeObjectURL(u); }
  if (P.elGains) P.elGains.forEach(g => { try { g.disconnect(); } catch (e) { } });
  P.els = bytes.map(ab => {
    const e = new Audio(URL.createObjectURL(new Blob([ab], { type: 'audio/wav' })));
    e.preload = 'auto'; e.preservesPitch = true; e.mozPreservesPitch = true; e.webkitPreservesPitch = true; return e;
  });
  P.elGains = P.els.map(e => { const s = ctx.createMediaElementSource(e), g = ctx.createGain(); s.connect(g).connect(ctx.destination); return g; });
}
const gainsFor = () => [(1 - opt.blend) * opt.vol, opt.blend * opt.vol];
function applyGains() {
  const g = gainsFor();
  if (P.gains) P.gains.forEach((n, i) => n.gain.value = g[i]);
  if (P.elGains) P.elGains.forEach((n, i) => n.gain.value = g[i]);
}
function waitEvent(el, ev, ms) {
  return new Promise(res => { let done = false; const f = () => { if (done) return; done = true; el.removeEventListener(ev, f); res(); }; el.addEventListener(ev, f); setTimeout(f, ms); });
}
async function seekEl(e, t) {
  if (e.readyState < 1) await waitEvent(e, 'loadedmetadata', 3000);
  if (Math.abs(e.currentTime - t) < 0.002) return;
  const w = waitEvent(e, 'seeked', 2000); e.currentTime = t; await w;
}
const PLAY_BTN = { all: '#btn-play', sel: '#btn-play-sel', view: '#btn-play-view', before: '#btn-play-before' };
let lastMark = null;
// play [a, b]. kind says which button/shortcut started it (for the toolbar highlight); returns false if nothing to play
function play(a, b, kind) {
  if (!S.buf.v) return false;
  stop();
  const tok = ++P.tok;
  const ctx = audioCtx(); if (ctx.state === 'suspended') ctx.resume();
  a = clamp(a, 0, S.D); b = clamp(b, a, S.D);
  if (b - a < 0.005) { if (a >= S.D - 0.01) toast('Already at the end of the clip'); return false; }
  P.a = a; P.b = b; P.playing = true; P.userView = false; P.kind = kind || 'all'; lastMark = a; P.mode = opt.rate === 1 ? 'wa' : 'el';
  const g = gainsFor();
  if (P.mode === 'wa') {
    P.gains = [ctx.createGain(), ctx.createGain()];
    P.srcs = [S.buf.v, S.buf.m].map((buf, i) => {
      P.gains[i].gain.value = g[i]; P.gains[i].connect(ctx.destination);
      const s = ctx.createBufferSource(); s.buffer = buf; s.connect(P.gains[i]);
      s.start(ctx.currentTime + 0.01, a, b - a); return s;
    });
    P.t0 = ctx.currentTime + 0.01;
    P.srcs[0].onended = () => { if (P.playing && P.mode === 'wa' && tok === P.tok) ended(); };
  } else {
    applyGains(); P.pending = true;
    const els = P.els; els.forEach(e => { e.pause(); e.playbackRate = opt.rate; });
    Promise.all(els.map(e => seekEl(e, a))).then(() => {
      if (tok !== P.tok || !P.playing) return;
      if (Math.abs(els[0].currentTime - a) > 0.3) { // the browser refused to seek: fall back to normal speed rather than playing the wrong place
        toast('Slow playback could not seek here, so this plays at 1×', true); P.pending = false; P.mode = 'wa';
        const k = P.kind, r = opt.rate; opt.rate = 1; play(a, b, k); opt.rate = r; return;
      }
      P.pending = false; els.forEach(e => e.play().catch(() => { }));
    });
  }
  updatePlayButtons();
  tick();
  return true;
}
function ended() { const a = P.a, b = P.b, k = P.kind; stop(); if (opt.loop) play(a, b, k); }
function stop() {
  P.tok++;
  if (P.srcs) for (const s of P.srcs) { try { s.onended = null; s.stop(); } catch (e) { } }
  if (P.gains) P.gains.forEach(n => { try { n.disconnect(); } catch (e) { } });
  P.srcs = []; P.gains = null; if (P.els) P.els.forEach(e => e.pause());
  P.playing = false; P.pending = false; P.kind = ''; cancelAnimationFrame(P.raf);
  updatePlayButtons(); S.dirtyDraw = true;
}
function updatePlayButtons() {
  for (const [k, id] of Object.entries(PLAY_BTN)) { const el = $(id); if (el) el.classList.toggle('playing', P.playing && P.kind === k); }
  const pb = $('#btn-play'); pb.textContent = P.playing ? '■' : '▶'; pb.title = P.playing ? 'Stop (Space)' : 'Play from the cursor / play the selection (Space)';
}
function playPos() {
  if (!P.playing) return null;
  if (P.pending) return P.a;
  if (P.mode === 'wa') return clamp(P.a + (audioCtx().currentTime - P.t0), P.a, P.b);
  return clamp(P.els[0].currentTime, P.a, P.b);
}
function tick() {
  if (!P.playing) return;
  const t = playPos();
  if (t != null && !P.pending) {
    if (lastMark != null && t >= lastMark) for (let i = Math.floor(lastMark * 10); i <= Math.floor(t * 10) && i < S.listened.length; i++) S.listened[i] = 1;
    lastMark = t;
    if (P.mode === 'el' && (t >= P.b - 0.003 || P.els[0].ended)) { ended(); return; }
    if (opt.follow && !P.userView && !dragActive() && (t > S.view.b || t < S.view.a)) { const w = S.view.b - S.view.a; S.view.a = clamp(t - w * 0.05, 0, Math.max(0, S.D - w)); S.view.b = S.view.a + w; }
  }
  S.dirtyDraw = true;
  P.raf = requestAnimationFrame(tick);
}
function selRange() {
  if (S.sel && S.sel.b - S.sel.a > 0.003) return [S.sel.a, S.sel.b];
  const t = S.tg && S.tg.tiers[S.active];
  if (S.selIv >= 0 && isInt(t)) { const it = t.items[S.selIv]; return [it.a, it.b]; }
  return null;
}
function playSelection() { const r = selRange(); if (r) return play(r[0], r[1], 'sel'); return play(S.cursor, S.cursor < S.view.b - 0.01 ? S.view.b : S.D, 'sel'); }
function playView() { return play(S.view.a, S.view.b, 'view'); }
function playBefore() { return play(Math.max(0, S.cursor - 1.5), S.cursor, 'before'); }
// Tab: play the selection; Tab again while playing = stop there and leave the reference line
function tabKey(shift) {
  if (P.playing && !shift) { const t = playPos(); stop(); setMark(t); return; }
  if (shift) playView(); else playSelection();
}
function setMark(t) {
  if (t == null) return;
  S.mark = t; S.cursor = t; ensureVisible(t, t); S.dirtyDraw = true; updatePos();
  toast(`Reference line at ${fmt(t)} s — drag boundaries to it (they snap), C moves the selected boundary there, K removes it`);
}
function togglePlay() { if (P.playing) { stop(); return; } const r = selRange(); if (r) play(r[0], r[1], 'all'); else play(S.cursor, S.D, 'all'); }

// ------------------------------------------------------------------ layout & drawing
const cv = $('#cv'), ctx2 = cv.getContext('2d'), ov = $('#overview'), og = ov.getContext('2d');
const LM = 82, AXIS = 20, GHOST_H = 24;
let L = null, W = 0, Hh = 0;
function ghostTier() {
  if (!opt.ghost || !S.tg) return null;
  const t = S.tg.tiers[S.active]; if (!t) return null;
  return S.orig.tiers.find(o => o.name === t.name && o.kind === t.kind) || null;
}
function layout() {
  // Every tier must always be fully visible: tiers get their room first, waveform/spectrogram share the rest.
  const nT = S.tg ? S.tg.tiers.length : 4;
  let gh = ghostTier() ? GHOST_H : 0, th = S.set.tierH;
  const MINW = 30, MINS = 40;
  const room = Hh - AXIS - gh - MINW - MINS;
  if (nT * th > room) th = Math.max(20, Math.floor(room / nT)); // short window: thinner tier rows
  if (Hh - AXIS - nT * th - gh < MINW + MINS && gh) { gh = 0; }   // very short: drop the prediction row before anything else
  const avail = Math.max(0, Hh - AXIS - nT * th - gh);
  const wave = avail >= MINW + MINS ? Math.max(MINW, Math.round(avail * 0.33)) : Math.round(avail * 0.4);
  const spec = avail - wave;
  let y = AXIS;
  const l = { wave: { y, h: wave } }; y += wave;
  l.spec = { y, h: spec }; y += spec;
  l.tiers = []; l.ghost = null;
  for (let i = 0; i < nT; i++) {
    l.tiers.push({ ti: i, y, h: th }); y += th;
    if (gh && i === S.active) { l.ghost = { y, h: gh }; y += gh; }
  }
  return l;
}
const tx = t => LM + (t - S.view.a) / (S.view.b - S.view.a) * (W - LM);
const xt = x => S.view.a + (x - LM) / (W - LM) * (S.view.b - S.view.a);
function resize() {
  const r = cv.parentElement.getBoundingClientRect(), d = devicePixelRatio || 1;
  W = r.width; Hh = r.height; cv.width = W * d; cv.height = Hh * d; ctx2.setTransform(d, 0, 0, d, 0, 0);
  const ow = ov.getBoundingClientRect().width; ov.width = ow * d; ov.height = 34 * d; og.setTransform(d, 0, 0, d, 0, 0);
  S.dirtyDraw = true;
}
new ResizeObserver(resize).observe($('#canvas-wrap'));

function fitText(g, s, w) {
  if (g.measureText(s).width <= w) return s;
  while (s.length > 1 && g.measureText(s + '…').width > w) s = s.slice(0, -1);
  return s.length > 1 ? s + '…' : '';
}
function draw() {
  requestAnimationFrame(draw);
  if (S.queryDirty && !$('#tab-query').classList.contains('hidden')) { S.queryDirty = false; updateQuery(); }
  if (!S.dirtyDraw) return; S.dirtyDraw = false;
  const g = ctx2; g.clearRect(0, 0, W, Hh);
  drawOverview();
  updateBlockRow();
  if (!S.tg || !S.blocks) return;
  L = layout();
  const { a: va, b: vb } = S.view, span = vb - va;
  // axis
  g.fillStyle = '#fafbfc'; g.fillRect(0, 0, W, AXIS); g.fillStyle = '#57606a'; g.font = '11px Consolas, monospace';
  const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].find(s => (W - LM) / (span / s) > 70) || 10;
  for (let t = Math.ceil(va / step) * step; t <= vb; t += step) { const x = tx(t); g.fillRect(x, AXIS - 5, 1, 5); g.fillText(t.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0) + 's', x + 2, 12); }
  // waveform
  const wv = L.wave; g.fillStyle = '#fff'; g.fillRect(LM, wv.y, W - LM, wv.h);
  const mid = wv.y + wv.h / 2, sc = (wv.h / 2 - 3) / S.blocks.peak, x = srcX(), sr = S.sr;
  g.fillStyle = S.set.src === 'm' ? '#7a5aa5' : '#4a6fa5';
  const spp = span * sr / (W - LM);
  for (let px = LM; px < W; px++) {
    const s0 = Math.floor(xt(px) * sr), s1 = Math.max(s0 + 1, Math.floor(xt(px + 1) * sr));
    let mn = 1, mx = -1;
    if (spp > S.blocks.B * 2) { const B = S.blocks.B; for (let k = Math.floor(s0 / B); k < Math.ceil(s1 / B) && k < S.blocks.mn.length; k++) { if (S.blocks.mn[k] < mn) mn = S.blocks.mn[k]; if (S.blocks.mx[k] > mx) mx = S.blocks.mx[k]; } }
    else for (let k = Math.max(0, s0); k < s1 && k < x.length; k++) { if (x[k] < mn) mn = x[k]; if (x[k] > mx) mx = x[k]; }
    if (mx >= mn) g.fillRect(px, mid - mx * sc, 1, Math.max(1, (mx - mn) * sc));
  }
  g.fillStyle = '#e1e4e8'; g.fillRect(LM, mid, W - LM, 0.5);
  if (opt.intensity && S.pitch) drawIntensity(g, wv);
  for (const is of S.issues) if (is.type === 'unlabelled' && !S.ignored.has(is.key)) {
    const x0 = Math.max(LM, tx(is.a)), x1 = tx(is.b); if (x1 < LM || x0 > W) continue;
    g.fillStyle = 'rgba(209,36,47,.22)'; g.fillRect(x0, wv.y + wv.h - 7, x1 - x0, 7);
  }
  // spectrogram
  const sp = L.spec;
  g.fillStyle = '#fff'; g.fillRect(LM, sp.y, W - LM, sp.h);
  if (S.specImg) {
    const hopS = S.spec.hop / S.sr, rows = Math.min(S.spec.nb, Math.ceil(S.set.fmax / S.spec.binHz));
    g.imageSmoothingEnabled = true;
    g.drawImage(S.specImg, va / hopS, S.spec.nb - rows, span / hopS, rows, LM, sp.y, W - LM, sp.h);
  }
  if (opt.formants && S.form) drawFormants(g, sp);
  if (opt.pitch && S.pitch) drawPitch(g, sp);
  const top = wv.y, bot = sp.y + sp.h;
  const r = selRange();
  if (r) { g.fillStyle = 'rgba(47,111,223,.13)'; const x0 = Math.max(LM, tx(r[0])), x1 = Math.min(W, tx(r[1])); if (x1 > x0) g.fillRect(x0, top, x1 - x0, Hh - top); }
  // boundaries through the signal panes
  const showTiers = opt.allB ? S.tg.tiers.map((t, i) => i) : [S.active];
  g.setLineDash([3, 3]);
  for (const ti of showTiers) {
    const t = S.tg.tiers[ti]; if (!t) continue;
    g.strokeStyle = ti === S.active ? 'rgba(47,111,223,.45)' : 'rgba(120,120,120,.3)'; g.beginPath();
    const times = isInt(t) ? t.items.slice(1).map(i => i.a) : t.items.map(i => i.p);
    for (const tt of times) { const bx = tx(tt); if (bx >= LM && bx <= W) { g.moveTo(bx + .5, top); g.lineTo(bx + .5, bot); } }
    g.stroke();
  }
  g.setLineDash([]);
  for (const lt of L.tiers) drawTier(g, lt);
  if (L.ghost) drawGhost(g, L.ghost);
  // label column
  g.fillStyle = '#fafbfc'; g.fillRect(0, AXIS, LM, Hh);
  g.font = '11px Segoe UI, sans-serif'; g.fillStyle = '#57606a';
  g.fillText(S.set.src === 'm' ? 'mix' : 'vocals', 6, wv.y + 14);
  if (opt.intensity) { g.fillStyle = '#c99700'; g.fillText('— intensity', 6, wv.y + 28); }
  g.fillStyle = '#57606a'; g.fillText('spectrogram', 6, sp.y + 14);
  g.font = '10px Consolas, monospace'; g.fillText(S.set.fmax + ' Hz', 6, sp.y + 28); g.fillText('0 Hz', 6, sp.y + sp.h - 3);
  if (opt.pitch) { g.fillStyle = '#d6336c'; g.font = '11px Segoe UI'; g.fillText('— pitch', 6, sp.y + 44); }
  if (opt.formants) { g.fillStyle = '#e03131'; g.font = '11px Segoe UI'; g.fillText('• formants', 6, sp.y + 58); }
  for (const lt of L.tiers) {
    const t = S.tg.tiers[lt.ti], act = lt.ti === S.active, blk = S.blockTiers.has(t.name);
    g.fillStyle = act ? '#2f6fdf' : '#f0f2f4'; g.fillRect(0, lt.y, LM - 2, lt.h - 1);
    g.fillStyle = act ? '#fff' : '#24292f'; g.font = (act ? 'bold ' : '') + '12px Segoe UI, sans-serif';
    g.fillText(fitText(g, `${lt.ti + 1} ${t.name}`, LM - 10), 6, lt.h < 36 ? lt.y + lt.h / 2 + 4 : lt.y + 17);
    g.font = '10px Segoe UI'; g.fillStyle = act ? '#dbe6fb' : '#8b949e';
    if (lt.h >= 36) g.fillText(`${t.kind === 'point' ? 'points' : 'intervals'} · ${t.items.filter(i => i.t).length}`, 6, lt.y + 31);
    if (blk && r) { g.fillStyle = '#e8590c'; g.fillRect(LM - 6, lt.y + 2, 3, lt.h - 5); }
  }
  if (L.ghost) { g.fillStyle = '#f6f6f6'; g.fillRect(0, L.ghost.y, LM - 2, L.ghost.h - 1); g.fillStyle = '#8b949e'; g.font = 'italic 10px Segoe UI'; g.fillText('prediction', 6, L.ghost.y + 15); }
  g.strokeStyle = '#d0d7de'; g.beginPath(); g.moveTo(LM - .5, 0); g.lineTo(LM - .5, Hh); g.stroke();
  const cx = tx(S.cursor); if (cx >= LM) { g.fillStyle = '#d1242f'; g.fillRect(cx, top, 1, Hh - top); }
  if (S.mark != null) { const mx = tx(S.mark); if (mx >= LM && mx <= W) {
    g.fillStyle = '#8250df'; g.fillRect(mx - 1, 0, 2, Hh);
    g.beginPath(); g.moveTo(mx - 6, 0); g.lineTo(mx + 6, 0); g.lineTo(mx, 8); g.closePath(); g.fill();
    g.font = 'bold 10px Segoe UI'; const lab = 'ref ' + fmt(S.mark), lw = g.measureText(lab).width + 6, lx = mx + lw + 4 > W ? mx - lw - 3 : mx + 3;
    g.fillRect(lx, 9, lw, 13); g.fillStyle = '#fff'; g.fillText(lab, lx + 3, 19); } }
  const pp = playPos(); if (pp != null) { const px = tx(pp); if (px >= LM) { g.fillStyle = '#1a7f37'; g.fillRect(px - 1, 0, 2, Hh); } }
  if (S.hover && S.hover.grab && !drag) { const gx = tx(S.hover.grab.t), lt = S.hover.grab.lt; g.fillStyle = 'rgba(232,89,12,.55)'; g.fillRect(gx - 2, lt.y, 4, lt.h); }
  if (S.hover && S.hover.snap != null) { const sx = tx(S.hover.snap); g.fillStyle = '#e8590c'; g.fillRect(sx - 1, top, 2, Hh - top); }
  updatePos();
}
function drawPitch(g, sp) {
  const p = S.pitch, hopS = p.hop / S.sr, lo = Math.log(S.set.plo), hi = Math.log(S.set.phi);
  const fy = f => sp.y + sp.h - (Math.log(f) - lo) / (hi - lo) * sp.h;
  const f0 = Math.max(1, Math.floor(S.view.a / hopS)), f1 = Math.min(p.nf - 1, Math.ceil(S.view.b / hopS));
  g.beginPath(); let pen = false;
  for (let f = f0; f <= f1; f++) {
    if (!isVoiced(f)) { pen = false; continue; }
    const x = tx((f + 0.5) * hopS), y = fy(p.f0[f]);
    if (pen && Math.abs(p.f0[f] / p.f0[f - 1] - 1) < 0.12) g.lineTo(x, y); else g.moveTo(x, y);
    pen = true;
  }
  g.lineJoin = 'round'; g.strokeStyle = 'rgba(255,255,255,.9)'; g.lineWidth = 5; g.stroke();
  g.strokeStyle = '#d6336c'; g.lineWidth = 2.5; g.stroke(); g.lineWidth = 1;
  g.fillStyle = 'rgba(214,51,108,.85)'; g.font = '10px Consolas, monospace';
  for (const f of [100, 200, 400, 800]) if (f > S.set.plo && f < S.set.phi) { const y = fy(f); g.fillRect(W - 6, y, 6, 1); g.fillText(f + '', W - 30, y + 3); }
}
function drawIntensity(g, wv) {
  const p = S.pitch, hopS = p.hop / S.sr;
  let mx = -1e9; for (let i = 0; i < p.nf; i++) if (p.db[i] > mx) mx = p.db[i];
  const lo = mx - 50, fy = d => wv.y + wv.h - clamp((d - lo) / 50, 0, 1) * (wv.h - 4) - 2;
  const f0 = Math.max(0, Math.floor(S.view.a / hopS)), f1 = Math.min(p.nf - 1, Math.ceil(S.view.b / hopS));
  g.beginPath();
  for (let f = f0; f <= f1; f++) { const X = tx((f + 0.5) * hopS), Y = fy(p.db[f]); f === f0 ? g.moveTo(X, Y) : g.lineTo(X, Y); }
  g.strokeStyle = '#e0a800'; g.lineWidth = 2; g.stroke(); g.lineWidth = 1;
  g.fillStyle = '#b08800'; g.font = '10px Consolas'; g.fillText(Math.round(mx) + ' dB', W - 44, wv.y + 10); g.fillText(Math.round(lo) + ' dB', W - 44, wv.y + wv.h - 3);
}
function drawFormants(g, sp) {
  const F = S.form, hopS = F.hop / S.sr, f0 = Math.max(0, Math.floor(S.view.a / hopS)), f1 = Math.min(F.nf - 1, Math.ceil(S.view.b / hopS));
  const step = Math.max(1, Math.floor((f1 - f0) / (W - LM) * 2));
  g.fillStyle = 'rgba(224,49,49,.85)';
  for (let f = f0; f <= f1; f += step) {
    if (!isVoiced(f)) continue;
    const X = tx((f + 0.5) * hopS);
    for (let k = 0; k < 4; k++) { const hz = F.data[f * 4 + k]; if (hz > 0 && hz < S.set.fmax) g.fillRect(X - 1, sp.y + sp.h - hz / S.set.fmax * sp.h - 1, 2.5, 2.5); }
  }
}
function drawTier(g, lt) {
  const tier = S.tg.tiers[lt.ti], y = lt.y, h = lt.h, act = lt.ti === S.active;
  g.fillStyle = act ? '#fbfcff' : '#fff'; g.fillRect(LM, y, W - LM, h);
  g.fillStyle = '#d0d7de'; g.fillRect(LM, y, W - LM, 1);
  const keys = S.origKeys[tier.name] || new Set();
  g.font = `15px ${DEVA_FONT}`; g.textBaseline = 'middle';
  if (tier.kind === 'point') {
    tier.items.forEach((pt, j) => {
      const X = tx(pt.p); if (X < LM - 50 || X > W + 50) return;
      const sel = act && j === S.selIv;
      g.fillStyle = sel ? '#d1242f' : '#5a32a3'; g.fillRect(X - (sel ? 1 : 0), y + 1, sel ? 3 : 1.5, h - 1);
      g.beginPath(); g.arc(X, y + 6, 3.5, 0, 7); g.fill();
      if (!keys.has(ptKey(pt))) { g.fillStyle = '#e8590c'; g.fillRect(X + 2, y + 2, 5, 3); }
      if (pt.t) { g.fillStyle = sel ? '#d1242f' : '#1f2328'; g.fillText(fitText(g, pt.t, 120), X + 5, y + h / 2 + 3); }
    });
    g.textBaseline = 'alphabetic'; return;
  }
  const it = tier.items, i0 = findIv(tier, S.view.a), i1 = findIv(tier, S.view.b);
  const issueIv = new Set(S.issues.filter(x => x.ti === lt.ti && x.iv != null && !S.ignored.has(x.key) && x.sev !== 'info').map(x => x.iv));
  for (let i = i0; i <= i1; i++) {
    const iv = it[i], x0 = Math.max(LM, tx(iv.a)), x1 = Math.min(W, tx(iv.b));
    if (iv.t) {
      g.fillStyle = tier.name === 'singers' ? (SING_COL[iv.t] || (iv.t.length > 1 ? '#c9b3ff' : '#ccc')) : (TIER_COL[tier.name] || '#e8eaed');
      if (iv.t === '~') g.fillStyle = '#eef6ec';
      if (iv.t.includes('?')) g.fillStyle = '#ffe3e3';
      g.fillRect(x0, y + 3, x1 - x0, h - 6);
    }
    if (act && i === S.selIv) { g.fillStyle = 'rgba(255,200,0,.45)'; g.fillRect(x0, y + 1, x1 - x0, h - 1); g.strokeStyle = '#e0a800'; g.strokeRect(x0 + .5, y + 1.5, x1 - x0 - 1, h - 3); }
    if (!keys.has(ivKey(iv))) { g.fillStyle = '#e8590c'; g.fillRect(x0, y + 3, Math.min(6, x1 - x0), 3); g.fillRect(x0, y + 3, 2, 8); }
    if (issueIv.has(i)) { g.fillStyle = '#d1242f'; g.fillRect(x0 + 2, y + h - 6, Math.max(2, x1 - x0 - 4), 3); }
    if (iv.t && x1 - x0 > 8) {
      g.fillStyle = '#1f2328';
      const s = fitText(g, iv.t, x1 - x0 - 4);
      g.fillText(s, (x0 + x1) / 2 - g.measureText(s).width / 2, y + h / 2 + 1);
    }
  }
  for (let k = Math.max(1, i0); k <= i1 && k < it.length; k++) {
    const bx = tx(it[k].a); if (bx < LM || bx > W) continue;
    const selB = S.selB && S.selB.ti === lt.ti && S.selB.k === k;
    g.fillStyle = selB ? '#d1242f' : '#24292f'; g.fillRect(bx - (selB ? 1 : 0), y + 1, selB ? 3 : 1, h - 1);
  }
  g.textBaseline = 'alphabetic';
}
function drawGhost(g, gl) {
  const ot = ghostTier(); if (!ot) return;
  g.fillStyle = '#f7f7f7'; g.fillRect(LM, gl.y, W - LM, gl.h);
  g.font = `12px ${DEVA_FONT}`; g.textBaseline = 'middle';
  if (ot.kind === 'point') {
    for (const pt of ot.items) { const X = tx(pt.p); if (X < LM || X > W) continue; g.fillStyle = '#aaa'; g.fillRect(X, gl.y, 1, gl.h); g.fillText(pt.t, X + 3, gl.y + gl.h / 2); }
  } else {
    const i0 = findIv(ot, S.view.a), i1 = findIv(ot, S.view.b);
    for (let i = i0; i <= i1; i++) {
      const iv = ot.items[i], x0 = Math.max(LM, tx(iv.a)), x1 = Math.min(W, tx(iv.b));
      g.fillStyle = '#b5b5b5'; if (i > 0 && x0 > LM) g.fillRect(x0, gl.y, 1, gl.h);
      if (iv.t && x1 - x0 > 8) { g.fillStyle = '#8b949e'; const s = fitText(g, iv.t, x1 - x0 - 4); g.fillText(s, (x0 + x1) / 2 - g.measureText(s).width / 2, gl.y + gl.h / 2 + 1); }
    }
  }
  g.textBaseline = 'alphabetic';
}
function drawOverview() {
  const w = ov.width / (devicePixelRatio || 1), h = 34, g = og;
  g.clearRect(0, 0, w, h); g.fillStyle = '#fff'; g.fillRect(0, 0, w, h);
  if (!S.tg) return;
  const X = t => t / S.D * w;
  const sg = T('singers'); if (isInt(sg)) for (const iv of sg.items) if (iv.t) { g.fillStyle = SING_COL[iv.t] || '#c9b3ff'; g.fillRect(X(iv.a), 2, X(iv.b) - X(iv.a), 9); }
  const wd = T('words'); if (isInt(wd)) { g.fillStyle = '#7d9ccf'; for (const iv of wd.items) if (iv.t) g.fillRect(X(iv.a), 13, Math.max(1, X(iv.b) - X(iv.a) - 1), 7); }
  const vc = T('vocables'); if (isInt(vc)) { g.fillStyle = '#e8a33d'; for (const iv of vc.items) if (iv.t) g.fillRect(X(iv.a), 13, X(iv.b) - X(iv.a), 7); }
  for (const is of S.issues) if (!S.ignored.has(is.key) && is.sev !== 'info') { g.fillStyle = is.sev === 'error' ? '#d1242f' : '#e8590c'; g.fillRect(X(is.a), 21, 2, 6); }
  if (S.listened) { g.fillStyle = '#2da44e'; for (let i = 0; i < S.listened.length; i++) if (S.listened[i]) g.fillRect(X(i / 10), 29, Math.ceil(w / S.listened.length) + .5, 4); }
  const r = selRange(); if (r) { g.fillStyle = 'rgba(47,111,223,.25)'; g.fillRect(X(r[0]), 0, Math.max(2, X(r[1]) - X(r[0])), h); }
  g.strokeStyle = '#2f6fdf'; g.lineWidth = 2; g.strokeRect(X(S.view.a) + 1, 1, X(S.view.b) - X(S.view.a) - 2, h - 2); g.lineWidth = 1;
  if (S.mark != null) { g.fillStyle = '#8250df'; g.fillRect(X(S.mark) - 1, 0, 2, h); }
  const pp = playPos(); if (pp != null) { g.fillStyle = '#1a7f37'; g.fillRect(X(pp) - 1, 0, 2, h); }
}
function updatePos() {
  const h = S.hover; let s = `cursor ${fmt(S.cursor)} s`;
  const pp = playPos(); if (pp != null) s = `▶ ${fmt(pp)} s (${fmt(P.a)}–${fmt(P.b)}${opt.rate !== 1 ? ', ' + opt.rate + '×' : ''}) · ` + s;
  if (S.mark != null) s += ` · ref ${fmt(S.mark)} s`;
  const r = selRange(); if (r) s += ` · sel ${fmt(r[0])}–${fmt(r[1])} (${Math.round((r[1] - r[0]) * 1000)} ms)`;
  if (h && h.t != null) {
    s += ` · ⌖ ${fmt(h.t)} s (song ${fmtMS(h.t + S.offset)})`;
    if (S.mark != null) { const dm = Math.round(((h.snap != null ? h.snap : h.t) - S.mark) * 1000); s += ` · ${dm >= 0 ? '+' : ''}${dm} ms from ref`; }
    if (S.pitch) { const f = frameAt(h.t); if (f >= 0 && f < S.pitch.nf && isVoiced(f)) s += ` · ${Math.round(S.pitch.f0[f])} Hz ${noteName(S.pitch.f0[f])}`; }
  }
  $('#pos').textContent = s;
}
requestAnimationFrame(draw);

// ------------------------------------------------------------------ view control
function setView(a, b, noPush) {
  if (!noPush) { S.zoomStack.push({ ...S.view }); if (S.zoomStack.length > 50) S.zoomStack.shift(); }
  let w = clamp(b - a, 0.05, S.D); a = clamp(a, 0, S.D - w);
  S.view = { a, b: a + w }; S.dirtyDraw = true;
}
const scrollView = (a, b) => setView(a, b, true);
const zoom = (f, at) => { const { a, b } = S.view; at = at ?? (a + b) / 2; setView(at - (at - a) * f, at + (b - at) * f); };
function zoomBack() { const v = S.zoomStack.pop(); if (v) setView(v.a, v.b, true); }
function ensureVisible(a, b) {
  const { a: va, b: vb } = S.view, w = vb - va;
  if (b - a > w * 0.9) { scrollView(a - 0.3, b + 0.3); return; }
  if (a < va || b > vb) scrollView((a + b) / 2 - w / 2, (a + b) / 2 + w / 2);
}

// ------------------------------------------------------------------ basic editing ops
function selectInterval(ti, i, playIt) {
  const tier = S.tg.tiers[ti]; if (!tier) return;
  S.active = ti; S.selB = null;
  if (!tier.items.length) { S.selIv = -1; updateEditor(); S.dirtyDraw = true; return; }
  i = clamp(i, 0, tier.items.length - 1); S.selIv = i;
  if (isInt(tier)) {
    const iv = tier.items[i]; S.sel = { a: iv.a, b: iv.b }; S.cursor = iv.a;
    ensureVisible(iv.a, iv.b);
    if (playIt) play(iv.a, iv.b);
  } else {
    const p = tier.items[i].p; S.sel = null; S.cursor = p; ensureVisible(p - 0.2, p + 0.2);
    if (playIt) play(Math.max(0, p - 0.3), p + 0.3);
  }
  S.dirtyDraw = true; S.queryDirty = true; updateEditor();
}
function addBoundary(ti, t, opts = {}) {
  const tier = S.tg.tiers[ti]; if (!tier) return false;
  if (tier.kind === 'point') {
    if (findPt(tier, t, 0.001) >= 0) { if (!opts.quiet) toast('A point is already there'); return false; }
    if (!opts.noCk) checkpoint();
    const j = tier.items.findIndex(p => p.p > t); const ins = j < 0 ? tier.items.length : j;
    tier.items.splice(ins, 0, { p: t, t: '' });
    if (!opts.noCk) { S.active = ti; S.selIv = ins; changed(); updateEditor(); }
    return true;
  }
  const sylI = tierIdx('syllables');
  if (tier.name === 'words' && sylI >= 0 && !opts.noLink) { const k = boundaryNear(S.tg.tiers[sylI], t, 0.015); if (k > 0) t = S.tg.tiers[sylI].items[k].a; }
  const i = findIv(tier, t), iv = tier.items[i];
  if (t - iv.a < MIN_DUR || iv.b - t < MIN_DUR) { if (!opts.quiet) toast('Boundary already there (or too close)'); return false; }
  if (!opts.noCk) checkpoint();
  let left = iv.t, right = '';
  const sp = iv.t.trim().indexOf(' ');
  if (sp > 0 && !opts.keepText) { left = iv.t.trim().slice(0, sp); right = iv.t.trim().slice(sp + 1); }
  tier.items.splice(i + 1, 0, { a: t, b: iv.b, t: right }); iv.b = t; iv.t = left;
  if (tier.name === 'words' && sylI >= 0 && !opts.noLink) addBoundary(sylI, t, { noCk: true, quiet: true, noLink: true, keepText: true });
  if (!opts.noCk) { S.selB = { ti, k: i + 1 }; changed(); }
  return true;
}
function mergeText(tier, a, b) { if (!a) return b; if (!b || a === b) return a; return tier.name === 'words' || !REQUIRED.includes(tier.name) ? a + ' ' + b : a + b; }
function removeBoundary(ti, k, noCk) {
  const tier = S.tg.tiers[ti]; if (!isInt(tier) || k <= 0 || k >= tier.items.length) return;
  if (!noCk) checkpoint();
  const A = tier.items[k - 1], B = tier.items[k];
  A.t = mergeText(tier, A.t, B.t); A.b = B.b; tier.items.splice(k, 1);
  if (!noCk) { S.selB = null; if (S.active === ti) S.selIv = k - 1; changed(); updateEditor(); }
}
function removePoint(ti, j) { checkpoint(); S.tg.tiers[ti].items.splice(j, 1); S.selIv = -1; changed(); updateEditor(); }
function linkedBoundaries(ti, k) {
  const t = S.tg.tiers[ti].items[k].a, out = [{ ti, k }];
  for (const j of ivTiers()) if (j !== ti) { const kk = boundaryNear(S.tg.tiers[j], t, LINK_EPS); if (kk > 0) out.push({ ti: j, k: kk }); }
  return out;
}
// When a word boundary moves (with its linked syllable boundary), the syllable boundaries inside the two
// neighbouring words are rescaled in proportion, so a word's syllables keep their relative sizes.
function sylPlan(list) {
  if (!opt.sylFollow) return null;
  const wi = tierIdx('words'), si = tierIdx('syllables'); if (wi < 0 || si < 0) return null;
  const w = list.find(b => b.ti === wi), sb = list.find(b => b.ti === si); if (!w || !sb) return null;
  const W = S.tg.tiers[wi].items, it = S.tg.tiers[si].items, x = W[w.k].a, a = W[w.k - 1].a, b = W[w.k].b;
  const left = [], right = [];
  for (let j = sb.k - 1; j > 0 && it[j].a > a + LINK_EPS; j--) left.push(j);
  for (let j = sb.k + 1; j < it.length && it[j].a < b - LINK_EPS; j++) right.push(j);
  if (!left.length && !right.length) return null;
  // smallest syllable on each side limits how far the word can shrink (every piece stays >= MIN_DUR)
  const segs = (pts, from, to) => { const q = [from, ...pts.map(j => it[j].a).sort((m, n) => m - n), to]; return Math.min(...q.slice(1).map((v, i) => v - q[i])); };
  const minL = segs(left, a, x), minR = segs(right, x, b);
  const lo = left.length ? (minL >= MIN_DUR ? a + (x - a) * MIN_DUR / minL : a + MIN_DUR * (left.length + 1)) : -Infinity;
  const hi = right.length ? (minR >= MIN_DUR ? b - (b - x) * MIN_DUR / minR : b - MIN_DUR * (right.length + 1)) : Infinity;
  return { si, a, b, x, left, right, lo, hi };
}
function moveBoundaries(list, t) {
  const sp = sylPlan(list);
  let lo = 0, hi = S.D;
  for (const { ti, k } of list) {
    const it = S.tg.tiers[ti].items;
    if (sp && ti === sp.si) { // syllable neighbours get rescaled, so only the outer limits count
      if (!sp.left.length) lo = Math.max(lo, it[k - 1].a + MIN_DUR);
      if (!sp.right.length) hi = Math.min(hi, it[k].b - MIN_DUR);
      continue;
    }
    lo = Math.max(lo, it[k - 1].a + MIN_DUR); hi = Math.min(hi, it[k].b - MIN_DUR);
  }
  if (sp) { lo = Math.max(lo, sp.lo); hi = Math.min(hi, sp.hi); }
  t = clamp(t, lo, Math.max(lo, hi));
  if (sp) {
    const it = S.tg.tiers[sp.si].items, { a, b, x } = sp;
    for (const j of sp.left) { const v = a + (it[j].a - a) * (t - a) / (x - a); it[j].a = v; it[j - 1].b = v; }
    for (const j of sp.right) { const v = b - (b - it[j].a) * (b - t) / (b - x); it[j].a = v; it[j - 1].b = v; }
  }
  for (const { ti, k } of list) { const it = S.tg.tiers[ti].items; it[k - 1].b = t; it[k].a = t; }
  return t;
}
function movePoint(ti, j, t) {
  const it = S.tg.tiers[ti].items;
  const lo = j > 0 ? it[j - 1].p + 0.001 : 0, hi = j + 1 < it.length ? it[j + 1].p - 0.001 : S.D;
  it[j].p = clamp(t, lo, hi); return it[j].p;
}
function setText(ti, i, text, noCk) { const iv = S.tg.tiers[ti].items[i]; if (iv.t === text) return; if (!noCk) checkpoint(); iv.t = text; changed(); }
function mergeEmptyNeighbours(ti, i) {
  const tier = S.tg.tiers[ti]; let idx = i;
  if (tier.items[idx].t) return idx;
  if (idx + 1 < tier.items.length && !tier.items[idx + 1].t) removeBoundary(ti, idx + 1, true);
  if (idx > 0 && !tier.items[idx - 1].t) { removeBoundary(ti, idx, true); idx--; }
  return idx;
}
function labelRange(ti, a, b, text, noCk) { // make [a,b] one interval labelled text
  const tier = S.tg.tiers[ti];
  const snap = t => { const k = boundaryNear(tier, t, 0.012); return k > 0 ? tier.items[k].a : t; };
  a = clamp(snap(a), 0, S.D); b = clamp(snap(b), 0, S.D);
  if (a < 0.012) a = 0; if (S.D - b < 0.012) b = S.D;
  if (!noCk) checkpoint();
  if (a > 0) addBoundary(ti, a, { noCk: true, quiet: true, keepText: true });
  if (b < S.D) addBoundary(ti, b, { noCk: true, quiet: true, keepText: true });
  let i = findIv(tier, a + 1e-6);
  while (i + 1 < tier.items.length && tier.items[i + 1].a < b - 1e-6) removeBoundary(ti, i + 1, true);
  tier.items[i].t = text;
  if (!text && (tier.name === 'vocables' || tier.name === 'singers')) i = mergeEmptyNeighbours(ti, i);
  S.active = ti; S.selIv = i; S.sel = { a: tier.items[i].a, b: tier.items[i].b }; S.selB = null;
  changed(); updateEditor();
  return i;
}
const selIsInterval = () => { const t = S.tg.tiers[S.active]; return S.selIv >= 0 && isInt(t) && S.sel && Math.abs(t.items[S.selIv].a - S.sel.a) < 1e-6 && Math.abs(t.items[S.selIv].b - S.sel.b) < 1e-6; };
function quickLabel(text) {
  const ti = S.active, tier = S.tg.tiers[ti];
  if (!isInt(tier)) { if (S.selIv >= 0) { setText(ti, S.selIv, text); updateEditor(); } return; }
  if (S.sel && S.sel.b - S.sel.a > 0.02 && !selIsInterval()) labelRange(ti, S.sel.a, S.sel.b, text);
  else if (S.selIv >= 0) {
    checkpoint(); tier.items[S.selIv].t = text;
    if (!text && (tier.name === 'vocables' || tier.name === 'singers')) S.selIv = mergeEmptyNeighbours(ti, S.selIv);
    changed(); updateEditor();
  } else toast('Drag over the waveform to select a time range first');
}
function createFromSelection() {
  const ti = S.active, tier = S.tg.tiers[ti];
  if (!isInt(tier) || !S.sel) return;
  const { a, b } = S.sel;
  const inside = tier.items.filter(iv => iv.t && iv.a < b - 0.005 && iv.b > a + 0.005);
  if (inside.length && !confirm(`The selection overlaps ${inside.length} labelled interval(s) on "${tier.name}" (${inside.slice(0, 4).map(i => i.t).join(', ')}). They will be merged into one new interval. Continue?`)) return;
  checkpoint();
  const i = labelRange(ti, a, b, inside.map(x => x.t).join(' '), true);
  const syI = tierIdx('syllables');
  if (tier.name === 'words' && syI >= 0) { const iv = tier.items[i]; for (const t of [iv.a, iv.b]) if (t > 0 && t < S.D) addBoundary(syI, t, { noCk: true, quiet: true, noLink: true, keepText: true }); }
  changed(); selectInterval(ti, i, false); txt.focus(); txt.select();
}
function zeroCrossing(t) {
  const x = S.x.v, sr = S.sr, c = Math.round(t * sr), R = Math.round(0.01 * sr);
  for (let d = 0; d < R; d++) for (const s of [c + d, c - d]) if (s > 0 && s < x.length && (x[s - 1] <= 0) !== (x[s] <= 0)) return s / sr;
  return t;
}

// ------------------------------------------------------------------ block shift / stretch
function blockPlan(a, b) {
  const eps = 1e-6, refs = [];
  S.tg.tiers.forEach((t, ti) => {
    if (!S.blockTiers.has(t.name)) return;
    if (isInt(t)) { for (let k = 1; k < t.items.length; k++) { const v = t.items[k].a; if (v >= a - eps && v <= b + eps) refs.push({ ti, k, t0: v }); } }
    else t.items.forEach((p, j) => { if (p.p >= a - eps && p.p <= b + eps) refs.push({ ti, j, t0: p.p }); });
  });
  return { a, b, refs };
}
function applyBlock(plan, f) { // f maps original time -> new time; all-or-nothing
  const byTier = {};
  for (const r of plan.refs) (byTier[r.ti] = byTier[r.ti] || []).push(r);
  const updates = [];
  for (const ti in byTier) {
    const t = S.tg.tiers[ti], moved = {};
    for (const r of byTier[ti]) moved[r.k ?? r.j] = f(r.t0);
    if (isInt(t)) {
      const n = t.items.length, bnd = [t.xmin];
      for (let k = 1; k < n; k++) bnd.push(k in moved ? moved[k] : t.items[k].a);
      bnd.push(t.xmax);
      for (let k = 1; k <= n; k++) if (!(bnd[k] - bnd[k - 1] >= MIN_DUR - 1e-9)) return false;
      updates.push(() => { for (let k = 1; k < n; k++) { t.items[k - 1].b = bnd[k]; t.items[k].a = bnd[k]; } });
    } else {
      const ps = t.items.map((p, j) => j in moved ? moved[j] : p.p);
      for (let j = 0; j < ps.length; j++) { if (ps[j] < t.xmin || ps[j] > t.xmax) return false; if (j && !(ps[j] - ps[j - 1] > 0.0005)) return false; }
      updates.push(() => t.items.forEach((p, j) => p.p = ps[j]));
    }
  }
  updates.forEach(u => u());
  return true;
}
function bestFraction(plan, mk, full) { // largest s in [0,1] such that mk(s*full) is valid
  if (applyBlock(plan, mk(full))) return full;
  let lo = 0, hi = 1;
  for (let i = 0; i < 16; i++) { const m = (lo + hi) / 2; if (applyBlock(plan, mk(full * m))) lo = m; else hi = m; }
  applyBlock(plan, mk(full * lo));
  return full * lo;
}
function shiftBlock(dt) {
  const r = selRange(); if (!r) { toast('Select a time range first: drag on the waveform, or click a word and Shift+click another'); return; }
  const plan = blockPlan(r[0], r[1]);
  if (!plan.refs.length) { toast('No boundaries of the ticked tiers inside the selection'); return; }
  checkpoint();
  const done = bestFraction(plan, d => t => t + d, dt);
  if (Math.abs(done) < 1e-6) { S.undo.pop(); toast('Cannot move further: an interval at the edge of the block would become too short'); return; }
  if (Math.abs(done - dt) > 1e-6) toast(`Moved ${Math.round(done * 1000)} ms (limit reached)`);
  S.sel = { a: r[0] + done, b: r[1] + done }; S.selIv = -1; S.cursor = S.sel.a; S.selB = null;
  $('#blk-info').textContent = `moved ${plan.refs.length} boundaries by ${Math.round(done * 1000)} ms`;
  changed(); updateEditor();
}
function stretchBlock(dEnd) {
  const r = selRange(); if (!r) { toast('Select a time range first'); return; }
  const [a, b] = r, plan = blockPlan(a, b);
  if (!plan.refs.length) { toast('No boundaries of the ticked tiers inside the selection'); return; }
  checkpoint();
  const done = bestFraction(plan, d => t => a + (t - a) * (b + d - a) / (b - a), dEnd);
  if (Math.abs(done) < 1e-6) { S.undo.pop(); toast('Cannot stretch further'); return; }
  S.sel = { a, b: b + done }; S.selIv = -1; S.selB = null;
  $('#blk-info').textContent = `stretched ${plan.refs.length} boundaries; end ${done > 0 ? '+' : ''}${Math.round(done * 1000)} ms`;
  changed(); updateEditor();
}
// ------------------------------------------------------------------ copy / paste of annotation blocks
function copyBlock() {
  const r = selRange(); if (!r) { toast('Select a time range (or an interval) to copy'); return; }
  const [a, b] = r, clip = { dur: b - a, tiers: {} };
  for (const t of S.tg.tiers) {
    if (!S.blockTiers.has(t.name)) continue;
    clip.tiers[t.name] = isInt(t)
      ? { kind: 'interval', items: t.items.filter(iv => iv.b > a + 1e-6 && iv.a < b - 1e-6).map(iv => ({ a: Math.max(a, iv.a) - a, b: Math.min(b, iv.b) - a, t: iv.t })) }
      : { kind: 'point', items: t.items.filter(p => p.p >= a && p.p <= b).map(p => ({ p: p.p - a, t: p.t })) };
  }
  if (!Object.keys(clip.tiers).length) { toast('Tick at least one tier in the Block row'); return; }
  S.clip = clip;
  const w = clip.tiers.words ? clip.tiers.words.items.map(i => i.t).filter(Boolean).join(' ') : '';
  if (w && navigator.clipboard) navigator.clipboard.writeText(w).catch(() => { });
  toast(`Copied ${clip.dur.toFixed(2)} s of ${Object.keys(clip.tiers).join(' + ')}${w ? ': “' + w.slice(0, 40) + '”' : ''}. Click where it should go and press Ctrl+V`);
  updateClipInfo();
}
function pasteTier(ti, c, A, scale, dur) {
  const t = S.tg.tiers[ti], B = Math.min(S.D, A + dur * scale);
  if (t.kind === 'point') {
    if (c.kind !== 'point') return;
    t.items = t.items.filter(p => p.p < A || p.p > B);
    for (const p of c.items) { const tt = A + p.p * scale; if (tt <= S.D) t.items.push({ p: tt, t: p.t }); }
    t.items.sort((x, y) => x.p - y.p); return;
  }
  if (c.kind !== 'interval') return;
  if (A > 1e-6) addBoundary(ti, A, { noCk: true, quiet: true, noLink: true, keepText: true });
  if (B < S.D - 1e-6) addBoundary(ti, B, { noCk: true, quiet: true, noLink: true, keepText: true });
  const i = findIv(t, A + 1e-7);
  while (i + 1 < t.items.length && t.items[i + 1].a < B - 1e-6) removeBoundary(ti, i + 1, true);
  const span = t.items[i], ins = [];
  for (const it of c.items) {
    const a = Math.max(span.a, A + it.a * scale), b = Math.min(span.b, A + it.b * scale);
    if (b - a > 1e-6) ins.push({ a, b, t: it.t });
  }
  if (!ins.length) return;
  ins[0].a = span.a; ins[ins.length - 1].b = span.b;          // keep the tier contiguous
  for (let k = 1; k < ins.length; k++) ins[k].a = ins[k - 1].b;
  t.items.splice(i, 1, ...ins.filter(x => x.b - x.a > 1e-6));
}
function pasteBlock(fit) {
  const c = S.clip; if (!c) { toast('Nothing copied yet: select a stretch and press Ctrl+C'); return; }
  let A = S.cursor, scale = 1;
  const r = selRange();
  if (fit) { if (!r) { toast('Select the target range to fit into'); return; } A = r[0]; scale = (r[1] - r[0]) / c.dur; }
  else if (r) A = r[0];
  if (P.playing) A = playPos();
  if (A + c.dur * scale > S.D + 1e-6) { toast('Not enough room before the end of the file', true); return; }
  checkpoint();
  const names = Object.keys(c.tiers).filter(n => tierIdx(n) >= 0);
  for (const n of names) pasteTier(tierIdx(n), c.tiers[n], A, scale, c.dur);
  S.sel = { a: A, b: A + c.dur * scale }; S.selIv = -1; S.selB = null; S.cursor = A;
  changed(); updateEditor(); ensureVisible(S.sel.a, S.sel.b);
  toast(`Pasted ${names.join(' + ')} at ${fmt(A)} s${Math.abs(scale - 1) > 1e-6 ? ` (stretched ×${scale.toFixed(2)})` : ''}. The pasted block is selected: [ ] shifts it`);
}
async function pasteTextIntoIntervals() {
  let text = '';
  try { text = await navigator.clipboard.readText(); } catch (e) { text = prompt('Paste the text here (words separated by spaces):') || ''; }
  text = text.replace(/\s+/g, ' ').trim(); if (!text) return;
  const ti = S.active, t = S.tg.tiers[ti]; if (!isInt(t)) return;
  const words = text.split(' ');
  const r = S.sel && !selIsInterval() ? [S.sel.a, S.sel.b] : null;
  let idx;
  if (r) {
    idx = t.items.map((iv, i) => i).filter(i => t.items[i].a >= r[0] - 1e-6 && t.items[i].b <= r[1] + 1e-6);
    const lab = idx.filter(i => t.items[i].t); if (lab.length >= words.length) idx = lab;
  } else if (S.selIv >= 0) idx = words.length === 1 ? [S.selIv] : t.items.map((iv, i) => i).filter(i => i >= S.selIv && (t.items[i].t || i === S.selIv)).slice(0, words.length);
  else { toast('Select an interval, or drag over the intervals to fill'); return; }
  if (!idx.length) { toast('No intervals in the selection'); return; }
  if (idx.length !== words.length && !confirm(`${words.length} word(s) on the clipboard but ${idx.length} interval(s) to fill. Fill them in order${words.length > idx.length ? ' (the last interval gets the remaining words)' : ''}?`)) return;
  checkpoint();
  idx.forEach((i, j) => {
    if (j >= words.length) return;
    t.items[i].t = j === idx.length - 1 ? words.slice(j).join(' ') : words[j];
    if (t.name === 'words' && opt.autoSyl) autoSyllabify(i, true);
  });
  changed(); updateEditor(); toast(`Filled ${Math.min(idx.length, words.length)} interval(s) from the clipboard`);
}
function updateClipInfo() { const c = S.clip; $('#clip-info').textContent = c ? `clipboard: ${c.dur.toFixed(2)} s · ${Object.keys(c.tiers).join(', ')}` : 'clipboard empty'; }

function renderBlockTiers() {
  const box = $('#blk-tiers'); box.innerHTML = '';
  if (!S.tg) return;
  for (const t of S.tg.tiers) {
    const l = document.createElement('label'); l.className = 'tog';
    l.innerHTML = `<input type="checkbox" ${S.blockTiers.has(t.name) ? 'checked' : ''}> ${t.name}`;
    l.querySelector('input').onchange = e => { e.target.checked ? S.blockTiers.add(t.name) : S.blockTiers.delete(t.name); S.dirtyDraw = true; };
    box.appendChild(l);
  }
}
function updateBlockRow() {
  const r = S.tg && selRange();
  $('#block-row').classList.toggle('off', !r);
  if (r && !$('#blk-info').dataset.keep) {
    const n = blockPlan(r[0], r[1]).refs.length;
    $('#blk-info').textContent = `${n} boundaries in ${fmt(r[0])}–${fmt(r[1])}`;
  }
}

// ------------------------------------------------------------------ syllables of a word
function wordSyllables(wi) {
  const w = T('words').items[wi], sy = T('syllables'); if (!isInt(sy)) return [];
  const out = [];
  for (let k = findIv(sy, w.a + 1e-6); k < sy.items.length && sy.items[k].a < w.b - 1e-6; k++) out.push(k);
  return out;
}
function setSyllables(wi, list, noCk) {
  const syI = tierIdx('syllables'), sy = S.tg.tiers[syI], w = T('words').items[wi];
  list = list.filter(s => s.length);
  if (!list.length || !isInt(sy)) return;
  if (!noCk) checkpoint();
  let ks = wordSyllables(wi);
  if (!ks.length || Math.abs(sy.items[ks[0]].a - w.a) > LINK_EPS) addBoundary(syI, w.a, { noCk: true, quiet: true, noLink: true, keepText: true });
  ks = wordSyllables(wi);
  if (Math.abs(sy.items[ks[ks.length - 1]].b - w.b) > LINK_EPS) addBoundary(syI, w.b, { noCk: true, quiet: true, noLink: true, keepText: true });
  ks = wordSyllables(wi);
  if (ks.length === list.length) ks.forEach((k, j) => sy.items[k].t = list[j]);
  else {
    const first = ks[0];
    while (wordSyllables(wi).length > 1) removeBoundary(syI, first + 1, true);
    const n = list.length, d = (w.b - w.a) / n;
    for (let j = 1; j < n; j++) addBoundary(syI, w.a + d * j, { noCk: true, quiet: true, noLink: true, keepText: true });
    wordSyllables(wi).forEach((k, j) => sy.items[k].t = list[j]);
    toast(`Syllable count changed (${ks.length} → ${n}): boundaries spread evenly, drag them to the onsets`);
  }
  changed(); updateEditor();
}
function autoSyllabify(wi, noCk) {
  const w = T('words').items[wi], word = stripQ(w.t); if (!word) return;
  const sy = T('syllables'); if (!isInt(sy)) return;
  const ks = wordSyllables(wi), cur = ks.map(k => sy.items[k].t), holds = cur.filter(t => t === '~').length;
  let list = /[A-Za-z]/.test(word) ? [word] : Deva.syllabify(word);
  if (holds && cur.length - holds === list.length) { let j = 0; list = cur.map(t => t === '~' ? '~' : list[j++]); }
  setSyllables(wi, list, noCk);
}

// ------------------------------------------------------------------ checks
function runChecks() {
  if (!S.tg) return;
  const out = [], W_ = T('words'), SY = T('syllables'), VO = T('vocables'), SG = T('singers');
  const wi = tierIdx('words'), si = tierIdx('syllables'), vi = tierIdx('vocables'), gi = tierIdx('singers');
  const add = (sev, type, msg, a, b, ti, iv, fix) => out.push({ sev, type, msg, a, b, ti, iv, fix, key: `${type}|${a.toFixed(2)}|${msg}` });
  for (const n of REQUIRED) { const t = T(n); if (!t) add('error', 'tier', `Required tier “${n}” is missing (Tier menu → add / rename)`, 0, 0, null, null); else if (!isInt(t)) add('error', 'tier', `Tier “${n}” must be an interval tier`, 0, 0, null, null); }
  const singerAt = t => isInt(SG) ? SG.items[findIv(SG, t)].t : 'x';
  const edge = iv => iv.a < 0.001 || iv.b > S.D - 0.001;
  const noSing = [];
  if (isInt(W_)) W_.items.forEach((iv, i) => {
    const w = iv.t.trim(); if (!w) return;
    const ww = stripQ(w);
    if (isInt(SY)) {
      for (const t of [iv.a, iv.b]) if (t > 0.001 && t < S.D - 0.001 && boundaryNear(SY, t, LINK_EPS) < 0)
        add('error', 'nosylb', `Word boundary at ${fmt(t)} has no syllable boundary`, t, t, wi, i, { label: 'add', fn: () => addBoundary(si, t, { noLink: true, keepText: true }) });
      const parts = wordSyllables(i).map(k => SY.items[k].t);
      const cat = parts.filter(p => p !== '~').join('').replace(/\?/g, '');
      if (ww && cat !== ww && !(edge(iv) && cat && (ww.endsWith(cat) || ww.startsWith(cat))))
        add('warn', 'sylspell', `Syllables “${parts.join('·')}” don't spell “${ww}”`, iv.a, iv.b, wi, i, { label: '✨ re-syllabify', fn: () => autoSyllabify(i) });
      if (parts.some(p => !p)) add('warn', 'sylempty', `Empty syllable inside “${ww}”`, iv.a, iv.b, wi, i);
    }
    if (/\s/.test(w)) add('warn', 'space', `“${w}” has a space: two words in one interval? Put the cursor between them and press B`, iv.a, iv.b, wi, i);
    const lat = /[A-Za-z]/.test(ww), dev = /[ऀ-ॿ]/.test(ww);
    if (lat && dev) add('error', 'mixed', `“${w}” mixes Latin and Devanagari`, iv.a, iv.b, wi, i);
    else if (lat && HINGLISH.has(ww.toLowerCase())) {
      const sug = Deva.roman2deva(ww.toLowerCase());
      add('warn', 'roman', `“${w}” looks like romanized Hindi: write it in Devanagari (${sug}?)`, iv.a, iv.b, wi, i, { label: '→ ' + sug, fn: () => { checkpoint(); W_.items[i].t = sug + (w.includes('?') ? '?' : ''); if (opt.autoSyl) autoSyllabify(i, true); changed(); updateEditor(); } });
    } else if (lat) add('info', 'english', `“${w}”: Latin script. OK if it is an English word`, iv.a, iv.b, wi, i);
    if (/[^ऀ-ॿA-Za-z'?\s\-]/.test(w)) add('warn', 'chars', `“${w}” contains unusual characters`, iv.a, iv.b, wi, i);
    if (w.includes('?')) add('info', 'unsure', `Marked uncertain: “${w}”`, iv.a, iv.b, wi, i);
    if (iv.b - iv.a < 0.04) add('warn', 'short', `Very short word “${w}” (${Math.round((iv.b - iv.a) * 1000)} ms)`, iv.a, iv.b, wi, i);
    if (isInt(SG) && !singerAt((iv.a + iv.b) / 2)) { const r = noSing[noSing.length - 1]; if (r && iv.a - r.b < 1.0) { r.b = iv.b; r.n++; r.words.push(w); } else noSing.push({ a: iv.a, b: iv.b, n: 1, i, words: [w] }); }
  });
  for (const r of noSing) add('warn', 'nosinger', r.n === 1 ? `“${r.words[0]}” is sung but the singers tier is empty here` : `${r.n} words sung (“${r.words.slice(0, 3).join(' ')}…”) but the singers tier is empty`, r.a, r.b, r.n === 1 ? wi : gi, r.n === 1 ? r.i : null,
    { label: 'select range', fn: () => { S.active = gi; S.selIv = -1; S.sel = { a: r.a, b: r.b }; S.cursor = r.a; ensureVisible(r.a, r.b); updateEditor(); S.dirtyDraw = true; toast('Now press 1–6 (A, B, C, D, E, AB) to label who sings'); } });
  if (isInt(SY) && isInt(W_)) SY.items.forEach((iv, k) => { if (iv.t && !W_.items[findIv(W_, (iv.a + iv.b) / 2)].t) add('warn', 'orphan', `Syllable “${iv.t}” outside any word`, iv.a, iv.b, si, k); });
  if (isInt(VO)) VO.items.forEach((iv, k) => {
    if (!iv.t) return;
    if (!VOCABLE_SET.has(iv.t)) add('warn', 'vocab', `Vocable “${iv.t}” is not one of आ ओ ई ए ऊ म्म mel`, iv.a, iv.b, vi, k);
    if (isInt(SG) && !singerAt((iv.a + iv.b) / 2)) add('warn', 'nosinger', `Vocable “${iv.t}” but the singers tier is empty here`, iv.a, iv.b, vi, k);
    if (isInt(W_)) { let ov = 0; for (let j = findIv(W_, iv.a); j < W_.items.length && W_.items[j].a < iv.b; j++) if (W_.items[j].t) ov += Math.min(iv.b, W_.items[j].b) - Math.max(iv.a, W_.items[j].a); if (ov > 0.5 * (iv.b - iv.a)) add('info', 'vocover', `Vocable “${iv.t}” overlaps words`, iv.a, iv.b, vi, k); }
  });
  if (isInt(SG)) {
    const used = new Set();
    SG.items.forEach((iv, k) => {
      if (!iv.t) return;
      if (!/^[A-Z]{1,4}$/.test(iv.t)) add('error', 'singerlbl', `Singer label “${iv.t}” should be capital letters (A, B, AB…)`, iv.a, iv.b, gi, k);
      else for (const c of iv.t) used.add(c);
      if (iv.t.length > 1 && new Set(iv.t).size !== iv.t.length) add('warn', 'singerlbl', `Singer label “${iv.t}” repeats a letter`, iv.a, iv.b, gi, k);
      const hasVoice = (isInt(W_) && W_.items.some(w => w.t && w.a < iv.b && w.b > iv.a)) || (isInt(VO) && VO.items.some(w => w.t && w.a < iv.b && w.b > iv.a));
      if (!hasVoice && iv.b - iv.a > 0.3) add('info', 'singernovoice', `Singer “${iv.t}” but no word/vocable here`, iv.a, iv.b, gi, k);
    });
    const letters = [...used].sort().join('');
    if (letters && !'ABCDEF'.startsWith(letters)) add('info', 'singerletters', `Singer letters used: ${letters} (normally A, B, … without gaps)`, 0, 0, gi, null);
  }
  if (S.pitch && isInt(W_)) {
    const hop = S.pitch.hop / S.sr, cov = new Uint8Array(S.pitch.nf);
    for (const tt of [W_, VO]) if (isInt(tt)) for (const iv of tt.items) if (iv.t) for (let f = Math.floor(iv.a / hop); f < Math.ceil(iv.b / hop) && f < cov.length; f++) cov[f] = 1;
    let start = -1, gap = 0;
    const tgt = vi >= 0 ? vi : wi;
    const flush = endF => { const a = start * hop, b = endF * hop; if (b - a >= S.set.unlabMin) add('warn', 'unlabelled', `Voice heard with no word or vocable (${(b - a).toFixed(1)} s): aakar? missed word? bleed?`, a, b, tgt, null, { label: 'select', fn: () => { S.active = tgt; S.selIv = -1; S.sel = { a, b }; S.cursor = a; ensureVisible(a, b); updateEditor(); S.dirtyDraw = true; } }); };
    for (let f = 0; f < S.pitch.nf; f++) {
      if (isVoiced(f) && !cov[f]) { if (start < 0) start = f; gap = 0; }
      else if (start >= 0) { gap++; if (gap > 6) { flush(f - gap + 1); start = -1; gap = 0; } }
    }
    if (start >= 0) flush(S.pitch.nf);
  }
  out.sort((x, y) => x.a - y.a);
  S.issues = out; S.dirtyDraw = true; renderIssues();
}
function sungMask() {
  const m = new Uint8Array(S.listened.length);
  for (const n of ['words', 'vocables']) { const t = T(n); if (isInt(t)) for (const iv of t.items) if (iv.t) for (let i = Math.floor(iv.a * 10); i < Math.ceil(iv.b * 10) && i < m.length; i++) m[i] = 1; }
  return m;
}
function listenedPct() {
  if (!S.listened || !S.tg) return 0; const m = sungMask(); let n = 0, h = 0;
  for (let i = 0; i < m.length; i++) if (m[i]) { n++; if (S.listened[i]) h++; }
  return n ? Math.round(100 * h / n) : 100;
}
const openIssues = () => S.issues.filter(i => !S.ignored.has(i.key) && i.sev !== 'info');
function renderIssues() {
  const box = $('#issues'); box.innerHTML = '';
  const sev = new Set($$('#issue-filters input[data-sev]').filter(c => c.checked).map(c => c.dataset.sev));
  const showIgn = $('#chk-show-ignored').checked, open = openIssues();
  $('#issue-count').textContent = open.length; $('#issue-count').className = 'badge' + (open.length ? '' : ' zero');
  const lp = listenedPct(), nErr = open.filter(i => i.sev === 'error').length;
  let changedN = 0;
  if (S.tg) for (const t of S.tg.tiers) { const k = S.origKeys[t.name] || new Set(); changedN += t.items.filter(iv => !k.has(isInt(t) ? ivKey(iv) : ptKey(iv))).length; }
  $('#issue-summary').innerHTML = `Listened to <b id="lp">${lp}%</b> of the sung audio<div class="bar"><div style="width:${lp}%"></div></div><b>${nErr}</b> errors · <b>${open.length - nErr}</b> warnings · ${changedN} intervals edited vs prediction`;
  let shown = 0;
  for (const is of S.issues) {
    const ign = S.ignored.has(is.key);
    if (!sev.has(is.sev) || (ign && !showIgn)) continue;
    if (++shown > 400) break;
    const d = document.createElement('div'); d.className = `issue ${is.sev}${ign ? ' ignored' : ''}`;
    const tn = is.ti != null && S.tg.tiers[is.ti] ? S.tg.tiers[is.ti].name : '';
    d.innerHTML = `<div class="ih"><span class="im"></span><span class="it">${fmt(is.a)}s ${tn}</span></div><div class="ia"></div>`;
    d.querySelector('.im').textContent = is.msg;
    const ia = d.querySelector('.ia');
    if (is.fix) { const b = document.createElement('button'); b.textContent = is.fix.label; b.onclick = e => { e.stopPropagation(); is.fix.fn(); }; ia.appendChild(b); }
    const bi = document.createElement('button'); bi.textContent = ign ? 'un-ignore' : 'ignore'; bi.title = 'False alarm: hide it (remembered for this song)';
    bi.onclick = e => { e.stopPropagation(); ign ? S.ignored.delete(is.key) : S.ignored.add(is.key); renderIssues(); S.dirtyDraw = true; saveStateSoon(); };
    ia.appendChild(bi);
    d.onclick = () => gotoIssue(is);
    box.appendChild(d);
  }
  if (!shown) box.innerHTML = '<p class="muted small">Nothing to show. 🎉</p>';
}
function gotoIssue(is) {
  if (is.ti != null && is.iv != null) selectInterval(is.ti, is.iv, opt.clickPlay);
  else { S.sel = is.b > is.a ? { a: is.a, b: is.b } : null; S.selIv = -1; S.cursor = is.a; if (is.ti != null) S.active = is.ti; ensureVisible(is.a, Math.max(is.b, is.a + 0.5)); updateEditor(); S.dirtyDraw = true; if (opt.clickPlay && S.sel) play(is.a, is.b); }
}
function nextIssue() { const open = openIssues(); if (!open.length) { toast('No open checks'); return; } gotoIssue(open.find(i => i.a > S.cursor + 0.01) || open[0]); }

// ------------------------------------------------------------------ editor panel
const txt = $('#txt');
function updateEditor() {
  const h = $('#helpers'); h.innerHTML = '';
  if (!S.tg) return;
  const tier = S.tg.tiers[S.active]; if (!tier) return;
  const iv = S.selIv >= 0 ? tier.items[S.selIv] : null, pt = tier.kind === 'point';
  if (document.activeElement !== txt) txt.value = iv ? iv.t : '';
  txt.disabled = !iv;
  $('#edit-where').textContent = !iv ? `${tier.name}: nothing selected` : pt ? `${tier.name} point #${S.selIv + 1} at ${fmt(iv.p)}` : `${tier.name} #${S.selIv + 1}  ${fmt(iv.a)}–${fmt(iv.b)}  (${Math.round((iv.b - iv.a) * 1000)} ms)`;
  const ta = $('#t-a'), tb = $('#t-b');
  if (document.activeElement !== ta) ta.value = iv ? fmt(pt ? iv.p : iv.a) : '';
  if (document.activeElement !== tb) tb.value = iv && !pt ? fmt(iv.b) : '';
  ta.disabled = !iv || (!pt && S.selIv === 0); tb.disabled = !iv || pt || S.selIv === tier.items.length - 1;
  // prediction at this position
  const ot = S.orig.tiers.find(t => t.name === tier.name && t.kind === tier.kind);
  let o = '', oText = null;
  if (iv && ot && !pt) { const oi = ot.items[findIv(ot, (iv.a + iv.b) / 2)]; oText = oi.t; o = ivKey(oi) === ivKey(iv) ? 'unchanged from prediction' : `prediction here: “${oi.t || '∅'}” ${fmt(oi.a)}–${fmt(oi.b)}`; }
  $('#orig').textContent = o; $('#orig').title = o; txt.classList.toggle('changed', o.startsWith('prediction'));
  const grp = label => { const g = document.createElement('div'); g.className = 'hg'; if (label) { const l = document.createElement('span'); l.className = 'hl'; l.textContent = label; g.appendChild(l); } h.appendChild(g); return g; };
  const btn = (g, html, title, fn, cls = 'q') => { const b = document.createElement('button'); b.className = cls; b.innerHTML = html; b.title = title; b.onclick = () => { fn(); cv.focus(); }; g.appendChild(b); return b; };
  if (iv && oText != null && oText !== iv.t) btn(grp(), '↺ prediction text', `Put back “${oText}”`, () => commitText(oText, false), '');
  if (tier.name === 'vocables' || tier.name === 'singers') {
    const list = tier.name === 'vocables' ? VOCABLES : SINGERS;
    const g = grp(S.sel && !selIsInterval() ? 'label drag-selection:' : 'label interval:');
    for (const [lab, key] of list) btn(g, `${lab}<small>${key}</small>`, `Label (${key})`, () => quickLabel(lab));
    btn(g, `∅<small>0</small>`, 'Clear label (0)', () => quickLabel(''));
    if (tier.name === 'singers') { const used = [...new Set(tier.items.map(i => i.t).filter(Boolean))].sort(); if (used.length) { const g2 = grp('in this song:'); for (const u of used) btn(g2, u, 'Label ' + u, () => quickLabel(u)); } }
    grp().innerHTML = `<span class="hl">${tier.name === 'vocables' ? 'Drag over the waveform, then 1–7. <b>mel</b> = long vowel continuing the previous syllable.' : 'AB = two or more voices at once · empty = nobody'}</span>`;
  }
  if (S.sel && !selIsInterval() && isInt(tier)) btn(grp(), '＋ interval from selection', 'Create one interval exactly over the selection (Enter)', createFromSelection, '');
  if (tier.name === 'words' && iv) {
    const ks = wordSyllables(S.selIv), sy = T('syllables');
    const g = grp('syllables:');
    const chips = document.createElement('span'); chips.className = 'sylchips';
    for (const k of ks) { const s = document.createElement('span'); s.textContent = sy.items[k].t || '∅'; chips.appendChild(s); }
    g.appendChild(chips);
    const inp = document.createElement('input'); inp.size = 14; inp.title = 'Type syllables separated by spaces (~ = hold), Enter to apply';
    inp.value = ks.map(k => sy.items[k].t).join(' ');
    inp.onkeydown = e => { e.stopPropagation(); if (e.key === 'Enter') { setSyllables(S.selIv, inp.value.trim().split(/\s+/)); cv.focus(); } };
    g.appendChild(inp);
    const sug = stripQ(iv.t) ? (/[A-Za-z]/.test(iv.t) ? [stripQ(iv.t)] : Deva.syllabify(stripQ(iv.t))) : [];
    btn(g, '✨', `Auto syllables: ${sug.join(' · ')} (Y)`, () => autoSyllabify(S.selIv));
    const lab = document.createElement('label'); lab.className = 'tog';
    lab.innerHTML = `<input type="checkbox" ${opt.autoSyl ? 'checked' : ''}> auto-syllabify`;
    lab.querySelector('input').onchange = e => opt.autoSyl = e.target.checked; g.appendChild(lab);
    btn(grp(), '?', 'Toggle uncertain mark (U)', toggleUnsure);
  }
  if (iv && (tier.name === 'words' || tier.name === 'syllables' || !REQUIRED.includes(tier.name))) {
    if (tier.name === 'syllables') btn(grp('quick:'), '~', 'Hold: continuation of the previous syllable', () => { setText(S.active, S.selIv, '~'); updateEditor(); });
    const g = grp('roman → देवनागरी:');
    const r = document.createElement('input'); r.size = 11; r.placeholder = 'pyaar';
    r.title = 'aa ii/ee uu/oo e ai o au · kh gh ch chh jh T Th D Dh N th dh ph bh sh · .n/M = ं · ~ = ँ · z f q = nukta · common words (main, mein, haan…) are looked up';
    const pv = document.createElement('span'); pv.className = 'preview';
    r.oninput = () => pv.textContent = Deva.roman2deva(r.value);
    r.onkeydown = e => { e.stopPropagation(); if (e.key === 'Enter' && r.value) { commitText(Deva.roman2deva(r.value), false); r.value = ''; pv.textContent = ''; } if (e.key === 'Escape') cv.focus(); };
    g.appendChild(r); g.appendChild(pv);
    const gp = grp(); for (const ch of PALETTE) btn(gp, '◌' + ch, 'insert ' + ch, () => insertAtCaret(ch));
  }
}
function insertAtCaret(ch) {
  if (S.selIv < 0) return;
  const s = txt.selectionStart ?? txt.value.length, e = txt.selectionEnd ?? s;
  const v = txt.value.slice(0, s) + ch + txt.value.slice(e);
  commitText(v, false, true);
  txt.value = v; txt.focus(); txt.setSelectionRange(s + ch.length, s + ch.length);
}
function commitText(value, advance, stay) {
  if (S.selIv < 0) return;
  const ti = S.active, tier = S.tg.tiers[ti], i = S.selIv; if (!tier.items[i]) return;
  const before = S.editStart ?? tier.items[i].t;
  value = value.replace(/\s+/g, ' ').trim();
  if (!S.editCk && tier.items[i].t !== value) checkpoint();
  tier.items[i].t = value; S.editCk = false; S.editStart = null;
  if (tier.name === 'words' && opt.autoSyl && stripQ(value) && stripQ(value) !== stripQ(before)) autoSyllabify(i, true);
  if ((tier.name === 'vocables' || tier.name === 'singers') && !value) S.selIv = mergeEmptyNeighbours(ti, i);
  changed();
  if (!stay) txt.blur();
  if (advance && S.selIv + 1 < tier.items.length) selectInterval(ti, S.selIv + 1, opt.clickPlay);
  else updateEditor();
}
txt.addEventListener('focus', () => { if (S.selIv >= 0) { S.editStart = S.tg.tiers[S.active].items[S.selIv].t; S.editCk = false; } });
txt.addEventListener('input', () => {
  if (S.selIv < 0) return;
  if (!S.editCk) { checkpoint(); S.editCk = true; }
  S.tg.tiers[S.active].items[S.selIv].t = txt.value; S.dirtyDraw = true;
});
txt.addEventListener('keydown', e => {
  e.stopPropagation();
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); commitText(txt.value, false); save(); return; }
  if (e.key === 'Enter') { e.preventDefault(); commitText(txt.value, !e.shiftKey); cv.focus(); }
  else if (e.key === 'Escape') { if (S.editStart != null && S.selIv >= 0) { S.tg.tiers[S.active].items[S.selIv].t = S.editStart; txt.value = S.editStart; if (S.editCk) S.undo.pop(); } S.editCk = false; S.editStart = null; txt.blur(); cv.focus(); S.dirtyDraw = true; }
  else if (e.key === 'Tab') { e.preventDefault(); tabKey(e.shiftKey); }
});
txt.addEventListener('blur', () => { if (S.editStart != null && S.selIv >= 0) commitText(txt.value, false, true); });
$('#btn-commit').onclick = () => { commitText(txt.value, true); cv.focus(); };
function onTimeEdit(which) {
  const el = which === 'a' ? $('#t-a') : $('#t-b'), v = parseFloat(el.value);
  const tier = S.tg.tiers[S.active]; if (S.selIv < 0 || isNaN(v)) return;
  checkpoint();
  if (tier.kind === 'point') { movePoint(S.active, S.selIv, v); }
  else {
    const k = which === 'a' ? S.selIv : S.selIv + 1; if (k <= 0 || k >= tier.items.length) { S.undo.pop(); return; }
    const got = moveBoundaries(linkedBoundaries(S.active, k), v);
    if (Math.abs(got - v) > 1e-6) toast('Limited by the neighbouring boundary');
    const iv = tier.items[S.selIv]; S.sel = { a: iv.a, b: iv.b };
  }
  changed(); updateEditor();
}
$('#t-a').addEventListener('change', () => onTimeEdit('a'));
$('#t-b').addEventListener('change', () => onTimeEdit('b'));
['#t-a', '#t-b', '#blk-ms'].forEach(s => $(s).addEventListener('keydown', e => { e.stopPropagation(); if (e.key === 'Enter') { e.target.dispatchEvent(new Event('change')); if (s === '#blk-ms') $('#blk-apply').click(); cv.focus(); } }));
function toggleUnsure() {
  const wi = tierIdx('words'); if (S.active !== wi || S.selIv < 0) { toast('Select a word first'); return; }
  const iv = S.tg.tiers[wi].items[S.selIv];
  checkpoint(); iv.t = iv.t.includes('?') ? stripQ(iv.t) : (iv.t ? iv.t + '?' : '?'); changed(); updateEditor();
}

// ------------------------------------------------------------------ mouse
cv.tabIndex = 0;
let drag = null;
function hit(e) {
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const lt = L ? L.tiers.find(l => y >= l.y && y < l.y + l.h) : null;
  const ghost = L && L.ghost && y >= L.ghost.y && y < L.ghost.y + L.ghost.h;
  return { x, y, t: clamp(xt(x), 0, S.D), lt, ghost };
}
const pxEps = px => px * (S.view.b - S.view.a) / (W - LM);
function boundaryHit(lt, x) {
  const tier = S.tg.tiers[lt.ti];
  const eps = pxEps(S.set.grabPx);
  if (tier.kind === 'point') return findPt(tier, xt(x), eps);
  return boundaryNear(tier, xt(x), eps);
}
cv.addEventListener('mousedown', e => {
  if (!S.tg || e.button !== 0) return;
  cv.focus();
  const h = hit(e);
  if (h.x < LM) { if (h.lt) { S.active = h.lt.ti; S.selIv = -1; S.selB = null; updateEditor(); S.dirtyDraw = true; } return; }
  const r = selRange();
  if ((e.ctrlKey || e.metaKey) && r && h.t >= r[0] - pxEps(8) && h.t <= r[1] + pxEps(8)) { // block drag
    const mode = Math.abs(tx(r[1]) - h.x) < 8 ? 'right' : Math.abs(tx(r[0]) - h.x) < 8 ? 'left' : 'move';
    const plan = blockPlan(r[0], r[1]);
    if (!plan.refs.length) { toast('No boundaries of the ticked tiers inside the selection'); return; }
    drag = { kind: 'block', mode, plan, a: r[0], b: r[1], t0: h.t, snap: JSON.stringify(S.tg.tiers), moved: false };
    return;
  }
  if (h.ghost) { const ot = ghostTier(); if (ot && isInt(ot)) { const oi = ot.items[findIv(ot, h.t)]; S.sel = { a: oi.a, b: oi.b }; S.selIv = -1; S.cursor = oi.a; if (opt.clickPlay) play(oi.a, oi.b); updateEditor(); S.dirtyDraw = true; } return; }
  if (h.lt) {
    const tier = S.tg.tiers[h.lt.ti], k = boundaryHit(h.lt, h.x);
    if (tier.kind === 'point' && k >= 0) {
      S.active = h.lt.ti; S.selIv = k; S.sel = null; updateEditor();
      drag = { kind: 'pt', ti: h.lt.ti, j: k, snap: JSON.stringify(S.tg.tiers), moved: false, x0: h.x, off: h.t - tier.items[k].p }; S.dirtyDraw = true; return;
    }
    if (tier.kind !== 'point' && k > 0) {
      S.active = h.lt.ti; S.selB = { ti: h.lt.ti, k };
      drag = { kind: 'b', list: e.altKey ? [{ ti: h.lt.ti, k }] : linkedBoundaries(h.lt.ti, k), snap: JSON.stringify(S.tg.tiers), moved: false, x0: h.x, off: h.t - tier.items[k].a };
      S.dirtyDraw = true; return;
    }
    drag = { kind: 'tier', lt: h.lt, x0: h.x, t0: h.t, shift: e.shiftKey };
    return;
  }
  if (e.shiftKey) { const a0 = r ? r[0] : S.cursor, b0 = r ? r[1] : S.cursor; S.sel = { a: Math.min(a0, h.t), b: Math.max(b0, h.t) }; if (h.t < a0) S.sel = { a: h.t, b: b0 }; S.selIv = -1; S.dirtyDraw = true; updateEditor(); return; }
  drag = { kind: 'sel', x0: h.x, t0: h.t };
});
const dragActive = () => !!drag;
function snapTime(t, exclude) {
  if (!S.set.snapPx) return null;
  const eps = pxEps(S.set.snapPx); let best = null, bd = eps;
  for (const j of ivTiers()) { const tier = S.tg.tiers[j]; const k = boundaryNear(tier, t, eps); if (k > 0 && !exclude.has(j + ':' + k)) { const d = Math.abs(tier.items[k].a - t); if (d < bd) { bd = d; best = tier.items[k].a; } } }
  if (Math.abs(S.cursor - t) < bd) { bd = Math.abs(S.cursor - t); best = S.cursor; }
  if (S.mark != null && Math.abs(S.mark - t) < pxEps(S.set.snapPx * 1.3)) best = S.mark; // the reference line wins
  return best;
}
window.addEventListener('mousemove', e => {
  if (!S.tg) return;
  const h = hit(e);
  S.hover = { t: h.x >= LM ? h.t : null, snap: null };
  if (!drag) {
    let cur = 'default';
    const r = selRange();
    if ((e.ctrlKey || e.metaKey) && r && h.t >= r[0] && h.t <= r[1]) cur = (Math.abs(tx(r[1]) - h.x) < 8 || Math.abs(tx(r[0]) - h.x) < 8) ? 'col-resize' : 'grab';
    else if (h.lt && h.x >= LM && boundaryHit(h.lt, h.x) >= (S.tg.tiers[h.lt.ti].kind === 'point' ? 0 : 1)) { cur = 'ew-resize'; const tr = S.tg.tiers[h.lt.ti], k = boundaryHit(h.lt, h.x); S.hover.grab = { lt: h.lt, t: tr.kind === 'point' ? tr.items[k].p : tr.items[k].a }; }
    else if (h.x >= LM && L && h.y < L.spec.y + L.spec.h) cur = 'text'; else if (h.lt || h.ghost) cur = 'pointer';
    cv.style.cursor = cur; S.dirtyDraw = true; return;
  }
  if (drag.kind === 'block') {
    const d = h.t - drag.t0; drag.moved = true;
    const { a, b, mode } = drag;
    const f = mode === 'move' ? (t => t + d) : mode === 'right' ? (t => a + (t - a) * (b + d - a) / (b - a)) : (t => b - (b - t) * (b - (a + d)) / (b - a));
    if (applyBlock(drag.plan, f)) { S.sel = { a: f(a), b: f(b) }; S.selIv = -1; S.selB = null; drag.last = d; }
    cv.style.cursor = mode === 'move' ? 'grabbing' : 'col-resize';
    S.dirtyDraw = true; return;
  }
  if (drag.kind === 'b' || drag.kind === 'pt') {
    if (!drag.moved && Math.abs(h.x - drag.x0) < S.set.dragPx) return; // small hand jitter on a click must not move anything
    drag.moved = true;
    let t = clamp(h.t - drag.off, 0, S.D);
    if (!e.shiftKey) { const ex = new Set(drag.kind === 'b' ? drag.list.map(b => b.ti + ':' + b.k) : []); const s = snapTime(t, ex); if (s != null) { t = s; S.hover.snap = s; } }
    if (drag.kind === 'b') moveBoundaries(drag.list, t); else movePoint(drag.ti, drag.j, t);
    S.dirtyDraw = true; updateEditorLight();
  } else if ((drag.kind === 'sel' && Math.abs(h.x - drag.x0) >= S.set.dragPx) || (drag.kind === 'tier' && Math.abs(h.x - drag.x0) >= S.set.dragPx)) {
    if (drag.kind === 'tier') drag.kind = 'sel';
    S.sel = { a: Math.min(drag.t0, h.t), b: Math.max(drag.t0, h.t) }; S.selIv = -1; S.selB = null; S.dirtyDraw = true;
  }
});
window.addEventListener('mouseup', e => {
  if (!drag) return;
  const h = hit(e), d = drag; drag = null;
  if (d.kind === 'block') {
    if (d.moved && d.last !== undefined) { S.undo.push(d.snap); S.redo = []; changed(); $('#blk-info').textContent = `${d.mode === 'move' ? 'moved' : 'stretched'} ${d.plan.refs.length} boundaries (${Math.round(d.last * 1000)} ms)`; }
    updateEditor(); return;
  }
  if (d.kind === 'b' || d.kind === 'pt') {
    if (d.moved) { S.undo.push(d.snap); S.redo = []; changed(); }
    S.hover = null; updateEditor(); return;
  }
  if (d.kind === 'tier') {
    const tier = S.tg.tiers[d.lt.ti];
    if (isInt(tier)) {
      const i = findIv(tier, d.t0);
      if (d.shift) { // extend selection over intervals
        const r = selRange(), iv = tier.items[i];
        S.active = d.lt.ti; S.sel = r ? { a: Math.min(r[0], iv.a), b: Math.max(r[1], iv.b) } : { a: iv.a, b: iv.b }; S.selIv = -1; S.selB = null;
        updateEditor(); S.dirtyDraw = true; return;
      }
      selectInterval(d.lt.ti, i, opt.clickPlay); S.cursor = d.t0;
    } else { S.active = d.lt.ti; S.selIv = -1; S.cursor = d.t0; S.sel = null; updateEditor(); }
    S.dirtyDraw = true; return;
  }
  if (d.kind === 'sel') {
    if (!S.sel || Math.abs(h.x - d.x0) < S.set.dragPx) { S.sel = null; S.cursor = h.t; S.selIv = -1; if (P.playing) stop(); }
    else { S.cursor = S.sel.a; if (opt.clickPlay) play(S.sel.a, S.sel.b); }
    S.queryDirty = true; updateEditor(); S.dirtyDraw = true;
  }
});
cv.addEventListener('dblclick', e => {
  const h = hit(e); if (!h.lt || h.x < LM) return;
  const tier = S.tg.tiers[h.lt.ti];
  if (boundaryHit(h.lt, h.x) >= (tier.kind === 'point' ? 0 : 1)) return;
  stop();
  if (addBoundary(h.lt.ti, h.t)) { S.active = h.lt.ti; if (isInt(tier)) S.selIv = findIv(tier, h.t); S.sel = null; S.cursor = h.t; updateEditor(); }
});
cv.addEventListener('mouseleave', () => { S.hover = null; S.dirtyDraw = true; });
cv.addEventListener('wheel', e => {
  if (!S.tg) return; e.preventDefault();
  const h = hit(e);
  if (P.playing) P.userView = true; // you scrolled/zoomed by hand: stop following until the next play
  const unit = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1, dx = e.deltaX * unit, dy = e.deltaY * unit;
  if (e.ctrlKey || e.metaKey) zoom(Math.exp(clamp(dy, -250, 250) * 0.0022 * S.set.zoomSpeed), h.x >= LM ? h.t : undefined);
  else { const w = S.view.b - S.view.a, dd = clamp((Math.abs(dx) > Math.abs(dy) ? dx : dy), -400, 400) / 600 * w * S.set.wheelSpeed; scrollView(S.view.a + dd, S.view.b + dd); }
}, { passive: false });
function updateEditorLight() {
  const tier = S.tg.tiers[S.active], iv = S.selIv >= 0 ? tier.items[S.selIv] : null; if (!iv) return;
  $('#edit-where').textContent = tier.kind === 'point' ? `${tier.name} point #${S.selIv + 1} at ${fmt(iv.p)}` : `${tier.name} #${S.selIv + 1}  ${fmt(iv.a)}–${fmt(iv.b)}  (${Math.round((iv.b - iv.a) * 1000)} ms)`;
}
let ovDrag = false;
const ovMove = e => { if (P.playing) P.userView = true; const r = ov.getBoundingClientRect(), t = (e.clientX - r.left) / r.width * S.D, w = S.view.b - S.view.a; scrollView(t - w / 2, t + w / 2); };
ov.addEventListener('mousedown', e => { if (!S.tg) return; ovDrag = true; ovMove(e); });
ov.addEventListener('wheel', e => { if (!S.tg) return; e.preventDefault(); if (P.playing) P.userView = true; const u = e.deltaMode === 1 ? 33 : 1, w = S.view.b - S.view.a, dd = clamp((e.deltaY || e.deltaX) * u, -400, 400) / 600 * w * S.set.wheelSpeed; scrollView(S.view.a + dd, S.view.b + dd); }, { passive: false });
window.addEventListener('mousemove', e => { if (ovDrag) ovMove(e); });
window.addEventListener('mouseup', () => ovDrag = false);

// ------------------------------------------------------------------ keyboard
const isTextField = el => !!el && (el.tagName === 'TEXTAREA' || el.isContentEditable || (el.tagName === 'INPUT' && /^(text|search|number|email|url|password|)$/i.test(el.type)));
const REPEATABLE = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ',', '.', '<', '>', '[', ']', '{', '}', '+', '=', '-', '_']);
window.addEventListener('keydown', e => {
  const tag = (document.activeElement && document.activeElement.tagName) || '';
  const k = e.key, ctrl = e.ctrlKey || e.metaKey;
  if (ctrl && k.toLowerCase() === 's') { e.preventDefault(); save(); return; }
  if (k === 'F1') { e.preventDefault(); $('#help').classList.toggle('hidden'); return; }
  if (ctrl && k.toLowerCase() === 'f') { e.preventDefault(); openFind(); return; }
  if (k === 'F3') { e.preventDefault(); findNext(e.shiftKey ? -1 : 1); return; }
  if (isTextField(document.activeElement)) { if (k === 'Escape') { document.activeElement.blur(); cv.focus(); } return; }
  if (tag === 'BUTTON' || tag === 'INPUT' || tag === 'SELECT') document.activeElement.blur(); // a clicked control must not eat Space/arrows
  const modal = $$('.modal').find(m => !m.classList.contains('hidden'));
  if (modal) { if (k === 'Escape') modal.classList.add('hidden'); return; }
  if (!S.tg) return;
  const tier = S.tg.tiers[S.active];
  const handled = () => e.preventDefault();
  if (e.repeat && !REPEATABLE.has(k) && !(ctrl && /^[zy]$/i.test(k))) { e.preventDefault(); return; } // holding Tab/Space/B… must not fire again and again
  if (ctrl && k.toLowerCase() === 'z') { handled(); e.shiftKey ? restore(S.redo, S.undo) : restore(S.undo, S.redo); return; }
  if (ctrl && k.toLowerCase() === 'y') { handled(); restore(S.redo, S.undo); return; }
  if (ctrl && k.toLowerCase() === 'c') { handled(); if (selRange()) copyBlock(); else if (S.selIv >= 0) navigator.clipboard.writeText(tier.items[S.selIv].t).then(() => toast('Copied label text')); return; }
  if (ctrl && k.toLowerCase() === 'v') { handled(); e.shiftKey ? pasteTextIntoIntervals() : pasteBlock(false); return; }
  if (ctrl && k.toLowerCase() === 'a') { handled(); S.sel = { a: 0, b: S.D }; S.selIv = -1; S.selB = null; updateEditor(); S.dirtyDraw = true; return; }
  if (ctrl && /^[1-9]$/.test(k)) { handled(); const ti = +k - 1; if (ti < S.tg.tiers.length) { const t = P.playing ? playPos() : S.cursor; addBoundary(ti, t); S.active = ti; updateEditor(); } return; }
  switch (k) {
    case ' ': handled(); togglePlay(); return;
    case 'Tab': handled(); tabKey(e.shiftKey); return;
    case 'Enter': handled(); if (S.sel && !selIsInterval() && isInt(tier)) createFromSelection(); else if (S.selIv >= 0) { txt.focus(); txt.select(); } return;
    case 'Escape': S.sel = null; S.selB = null; stop(); S.dirtyDraw = true; updateEditor(); return;
    case 'Home': handled(); { const r = selRange(); if (r) { S.cursor = r[0]; ensureVisible(r[0], r[0]); S.dirtyDraw = true; } } return;
    case 'End': handled(); { const r = selRange(); if (r) { S.cursor = r[1]; ensureVisible(r[1], r[1]); S.dirtyDraw = true; } } return;
    case 'ArrowRight': case 'ArrowLeft': {
      handled(); const dir = k === 'ArrowRight' ? 1 : -1;
      if (tier.kind === 'point') { let j = S.selIv >= 0 ? S.selIv + dir : (dir > 0 ? tier.items.findIndex(p => p.p > S.cursor) : tier.items.map(p => p.p < S.cursor).lastIndexOf(true)); if (j >= 0 && j < tier.items.length) selectInterval(S.active, j, opt.clickPlay); return; }
      let i = S.selIv >= 0 ? S.selIv + dir : findIv(tier, S.cursor);
      if (ctrl) while (i >= 0 && i < tier.items.length && !tier.items[i].t) i += dir;
      if (i >= 0 && i < tier.items.length) selectInterval(S.active, i, opt.clickPlay);
      return;
    }
    case 'ArrowUp': case 'ArrowDown': {
      handled(); const n = S.tg.tiers.length, ni = (S.active + (k === 'ArrowDown' ? 1 : n - 1)) % n;
      const keepSel = S.sel && !selIsInterval() ? S.sel : null;
      const t = S.selIv >= 0 && isInt(tier) ? (tier.items[S.selIv].a + tier.items[S.selIv].b) / 2 : S.cursor;
      S.active = ni; S.selB = null;
      if (keepSel || !isInt(S.tg.tiers[ni])) { S.selIv = -1; S.sel = keepSel; updateEditor(); S.dirtyDraw = true; }
      else selectInterval(ni, findIv(S.tg.tiers[ni], t), false);
      return;
    }
    case 'Delete': case 'Backspace': handled();
      if (tier.kind === 'point' && S.selIv >= 0) removePoint(S.active, S.selIv);
      else if (S.selB) removeBoundary(S.selB.ti, S.selB.k); else toast('Click a boundary (or point) to select it first');
      return;
    case ',': case '.': case '<': case '>': {
      handled(); const dt = (k === ',' || k === '<' ? -1 : 1) * (k === '<' || k === '>' ? S.set.bigNudgeMs : S.set.nudgeMs) / 1000;
      if (tier.kind === 'point' && S.selIv >= 0) { checkpoint(); movePoint(S.active, S.selIv, tier.items[S.selIv].p + dt); changed(); updateEditorLight(); return; }
      if (!S.selB) { toast('Click a boundary to select it first'); return; }
      checkpoint(); moveBoundaries(e.altKey ? [S.selB] : linkedBoundaries(S.selB.ti, S.selB.k), S.tg.tiers[S.selB.ti].items[S.selB.k].a + dt); changed(); updateEditorLight(); return;
    }
    case '[': handled(); shiftBlock(-0.01); return;
    case ']': handled(); shiftBlock(0.01); return;
    case '{': handled(); shiftBlock(-0.05); return;
    case '}': handled(); shiftBlock(0.05); return;
    case '+': case '=': handled(); zoom(0.66); return;
    case '-': case '_': handled(); zoom(1.5); return;
  }
  const lk = k.toLowerCase();
  if (ctrl || e.altKey) return;
  if (lk === 'b') {
    handled(); const t = P.playing ? playPos() : S.cursor;
    if (e.shiftKey) { checkpoint(); let n = 0; for (const ti of ivTiers()) if (addBoundary(ti, t, { noCk: true, quiet: true, noLink: true })) n++; if (n) changed(); else S.undo.pop(); toast(`Added boundary on ${n} tier(s)`); return; }
    if (addBoundary(S.active, t) && !P.playing && isInt(tier)) S.selIv = findIv(tier, t);
    updateEditor(); return;
  }
  if (lk === 'c') { handled(); if (!S.selB) { toast('Select a boundary first'); return; } checkpoint(); const tgt = S.mark != null ? S.mark : S.cursor, got = moveBoundaries(linkedBoundaries(S.selB.ti, S.selB.k), tgt); changed(); updateEditor();
    toast(Math.abs(got - tgt) > 1e-4 ? `Stopped ${Math.round(Math.abs(got - tgt) * 1000)} ms short: a neighbouring boundary is in the way` : `Boundary moved to the ${S.mark != null ? 'reference line' : 'cursor'}`); return; }
  if (lk === 'o') {
    handled();
    if (S.selB) { checkpoint(); const b = S.tg.tiers[S.selB.ti].items[S.selB.k].a; moveBoundaries(linkedBoundaries(S.selB.ti, S.selB.k), zeroCrossing(b)); changed(); toast('Boundary moved to nearest zero crossing'); }
    else { S.cursor = zeroCrossing(S.cursor); S.dirtyDraw = true; toast('Cursor moved to nearest zero crossing'); }
    return;
  }
  if (lk === 'q') { handled(); playBefore(); return; }
  if (lk === 'z') { handled(); const r = selRange(); if (r) setView(r[0] - 0.05, r[1] + 0.05); return; }
  if (lk === 'x') { handled(); zoomBack(); return; }
  if (lk === 'a') { handled(); setView(0, S.D); return; }
  if (lk === 'm') { handled(); setBlend(opt.blend > 0.5 ? 0 : 1); return; }
  if (lk === 's') { handled(); setRate(opt.rate === 1 ? 0.7 : 1); return; }
  if (lk === 'u') { handled(); toggleUnsure(); return; }
  if (lk === 'n') { handled(); nextIssue(); return; }
  if (lk === 'k') { handled(); if (S.mark != null) { S.mark = null; S.dirtyDraw = true; toast('Reference line removed'); } else setMark(P.playing ? playPos() : S.cursor); return; }
  if (lk === 'r') { handled(); if (S.mark == null) { toast('No reference line (press Tab while playing, or K)'); return; } S.cursor = S.mark; ensureVisible(S.mark, S.mark); S.dirtyDraw = true; return; }
  if (lk === 'y') { handled(); if (tier.name === 'words' && S.selIv >= 0) autoSyllabify(S.selIv); return; }
  if (/^[0-9]$/.test(k) && (tier.name === 'vocables' || tier.name === 'singers')) {
    handled(); if (k === '0') { quickLabel(''); return; }
    const ent = (tier.name === 'vocables' ? VOCABLES : SINGERS).find(x => x[1] === k);
    if (ent) quickLabel(ent[0]);
  }
});

// ------------------------------------------------------------------ find / replace
function fillFindTiers() {
  const s = $('#f-tier'), cur = s.value; s.innerHTML = '<option value="*">all tiers</option>';
  if (S.tg) for (const t of S.tg.tiers) { const o = document.createElement('option'); o.value = t.name; o.textContent = t.name; s.appendChild(o); }
  s.value = [...s.options].some(o => o.value === cur) ? cur : 'words';
}
function openFind() { $('#findbar').classList.remove('hidden'); fillFindTiers(); $('#f-q').focus(); $('#f-q').select(); }
function matcher() {
  const q = $('#f-q').value; if (!q) return null;
  let re;
  try { re = $('#f-re').checked ? new RegExp($('#f-whole').checked ? `^(?:${q})$` : q, 'u') : null; } catch (e) { $('#f-info').textContent = 'bad regex'; return null; }
  return s => re ? re.test(s) : ($('#f-whole').checked ? s === q : s.includes(q));
}
function findHits() {
  const m = matcher(); if (!m || !S.tg) return [];
  const want = $('#f-tier').value, hits = [];
  S.tg.tiers.forEach((t, ti) => { if (want !== '*' && t.name !== want) return; t.items.forEach((it, i) => { if (it.t && m(it.t)) hits.push({ ti, i, t: isInt(t) ? it.a : it.p }); }); });
  return hits.sort((a, b) => a.t - b.t || a.ti - b.ti);
}
function findNext(dir) {
  if ($('#findbar').classList.contains('hidden')) { openFind(); return; }
  const hits = findHits(); $('#f-info').textContent = hits.length + ' match(es)';
  if (!hits.length) return;
  const cur = S.selIv >= 0 ? { ti: S.active, i: S.selIv } : null;
  const idx = cur ? hits.findIndex(h => h.ti === cur.ti && h.i === cur.i) : -1;
  let n;
  if (idx >= 0) n = (idx + dir + hits.length) % hits.length;
  else { n = dir > 0 ? hits.findIndex(h => h.t > S.cursor) : hits.map(h => h.t < S.cursor).lastIndexOf(true); if (n < 0) n = dir > 0 ? 0 : hits.length - 1; }
  selectInterval(hits[n].ti, hits[n].i, opt.clickPlay);
  $('#f-info').textContent = `${n + 1} / ${hits.length}`;
}
function replaceIn(s) {
  const q = $('#f-q').value, r = $('#f-r').value;
  if ($('#f-whole').checked) return r;
  if ($('#f-re').checked) return s.replace(new RegExp(q, 'gu'), r);
  return s.split(q).join(r);
}
function afterTextChange(ti, i) { const t = S.tg.tiers[ti]; if (t.name === 'words' && opt.autoSyl) autoSyllabify(i, true); }
$('#f-next').onclick = () => findNext(1);
$('#f-prev').onclick = () => findNext(-1);
$('#f-q').addEventListener('keydown', e => { e.stopPropagation(); if (e.key === 'Enter') findNext(e.shiftKey ? -1 : 1); if (e.key === 'Escape') $('#findbar').classList.add('hidden'); });
$('#f-r').addEventListener('keydown', e => e.stopPropagation());
$('#f-rep').onclick = () => {
  const m = matcher(); if (!m || S.selIv < 0) return;
  const it = S.tg.tiers[S.active].items[S.selIv]; if (!it.t || !m(it.t)) { findNext(1); return; }
  checkpoint(); it.t = replaceIn(it.t); afterTextChange(S.active, S.selIv); changed(); updateEditor(); findNext(1);
};
$('#f-repall').onclick = () => {
  const hits = findHits(); if (!hits.length) { $('#f-info').textContent = '0 matches'; return; }
  if (!confirm(`Replace ${hits.length} label(s)?`)) return;
  checkpoint(); for (const h of hits) { const it = S.tg.tiers[h.ti].items[h.i]; it.t = replaceIn(it.t); afterTextChange(h.ti, h.i); }
  changed(); updateEditor(); $('#f-info').textContent = `replaced ${hits.length}`;
};
$('#f-close').onclick = () => $('#findbar').classList.add('hidden');

// ------------------------------------------------------------------ tiers menu
function tierAction(act) {
  if (!S.tg) return;
  const ti = S.active, t = S.tg.tiers[ti];
  const uniqueName = base => { let n = base, i = 2; while (tierIdx(n) >= 0) n = base + i++; return n; };
  if (act === 'tier-add-int' || act === 'tier-add-pt') {
    const name = prompt('Name of the new tier:', uniqueName(act === 'tier-add-int' ? 'notes' : 'points')); if (!name) return;
    if (tierIdx(name) >= 0) { toast('A tier with that name exists', true); return; }
    checkpoint();
    S.tg.tiers.splice(ti + 1, 0, act === 'tier-add-int' ? { name, kind: 'interval', xmin: S.tg.xmin, xmax: S.tg.xmax, items: [{ a: S.tg.xmin, b: S.tg.xmax, t: '' }] } : { name, kind: 'point', xmin: S.tg.xmin, xmax: S.tg.xmax, items: [] });
    S.active = ti + 1; S.selIv = -1;
  } else if (act === 'tier-dup') { checkpoint(); const c = JSON.parse(JSON.stringify(t)); c.name = uniqueName(t.name + '-copy'); S.tg.tiers.splice(ti + 1, 0, c); S.active = ti + 1; }
  else if (act === 'tier-rename') {
    const name = prompt(`Rename tier “${t.name}” to:`, t.name); if (!name || name === t.name) return;
    if (tierIdx(name) >= 0) { toast('A tier with that name exists', true); return; }
    if (REQUIRED.includes(t.name) && !confirm(`“${t.name}” is a tier the pipeline reads. Saving will be refused while it is missing. Rename anyway?`)) return;
    checkpoint(); S.blockTiers.has(t.name) && (S.blockTiers.delete(t.name), S.blockTiers.add(name)); t.name = name;
  } else if (act === 'tier-up' || act === 'tier-down') {
    const j = act === 'tier-up' ? ti - 1 : ti + 1; if (j < 0 || j >= S.tg.tiers.length) return;
    checkpoint(); [S.tg.tiers[ti], S.tg.tiers[j]] = [S.tg.tiers[j], S.tg.tiers[ti]]; S.active = j;
  } else if (act === 'tier-clear') {
    if (!confirm(`Remove all text from tier “${t.name}”? (boundaries stay; undo works)`)) return;
    checkpoint(); t.items.forEach(i => i.t = '');
  } else if (act === 'tier-reset') {
    const o = S.orig.tiers.find(x => x.name === t.name && x.kind === t.kind); if (!o) { toast('The prediction has no tier with this name', true); return; }
    if (!confirm(`Replace tier “${t.name}” with the pipeline prediction? (other tiers unchanged; undo works)`)) return;
    checkpoint(); t.items = JSON.parse(JSON.stringify(o.items)); S.selIv = -1;
  } else if (act === 'tier-remove') {
    if (S.tg.tiers.length <= 1) return;
    if (!confirm(`Remove tier “${t.name}”?${REQUIRED.includes(t.name) ? '\n\nWARNING: the pipeline needs this tier; saving will be refused without it.' : ''}`)) return;
    checkpoint(); S.tg.tiers.splice(ti, 1); S.active = clamp(ti, 0, S.tg.tiers.length - 1); S.selIv = -1;
  }
  S.selB = null; changed(); updateEditor(); renderBlockTiers(); fillFindTiers();
}

// ------------------------------------------------------------------ exports
function encodeWav(x, sr) {
  const n = x.length, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); dv.setUint32(4, 36 + n * 2, true); w(8, 'WAVE'); w(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true); w(36, 'data'); dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) dv.setInt16(44 + i * 2, clamp(Math.round(x[i] * 32767), -32768, 32767), true);
  return buf;
}
function partTextGrid(a, b) { // selection only, times preserved
  return { xmin: a, xmax: b, tiers: S.tg.tiers.map(t => {
    if (!isInt(t)) return { ...t, xmin: a, xmax: b, items: t.items.filter(p => p.p >= a && p.p <= b) };
    const items = t.items.filter(iv => iv.b > a && iv.a < b).map(iv => ({ a: Math.max(a, iv.a), b: Math.min(b, iv.b), t: iv.t })).filter(iv => iv.b - iv.a > 1e-6);
    return { ...t, xmin: a, xmax: b, items };
  }) };
}
function fileAction(act) {
  if (!S.tg) return;
  const r = selRange(), base = S.id;
  if (act === 'dl-tg') download(base + '.TextGrid', textGridText(S.tg), 'text/plain;charset=utf-8');
  else if (act === 'dl-tg-sel') { if (!r) return toast('Select a time range first'); download(`${base}_${fmt(r[0])}-${fmt(r[1])}.TextGrid`, textGridText(partTextGrid(r[0], r[1])), 'text/plain;charset=utf-8'); }
  else if (act === 'dl-tsv') {
    const rows = ['tier\tstart\tend\tduration\ttext'];
    for (const t of S.tg.tiers) for (const it of t.items) rows.push(isInt(t) ? `${t.name}\t${fmt(it.a)}\t${fmt(it.b)}\t${fmt(it.b - it.a)}\t${it.t}` : `${t.name}\t${fmt(it.p)}\t\t\t${it.t}`);
    download(base + '.tsv', '﻿' + rows.join('\n') + '\n', 'text/tab-separated-values;charset=utf-8');
  } else if (act === 'dl-wav-v' || act === 'dl-wav-m') {
    if (!r) return toast('Select a time range first');
    const x = act === 'dl-wav-v' ? S.x.v : S.x.m;
    download(`${base}_${act === 'dl-wav-v' ? 'vocals' : 'mix'}_${fmt(r[0])}-${fmt(r[1])}.wav`, encodeWav(x.subarray(Math.floor(r[0] * S.sr), Math.ceil(r[1] * S.sr)), S.sr), 'audio/wav');
  } else if (act === 'reload') { if (isDirty() && !confirm('Reload from disk? Unsaved changes stay in the autosaved draft and will be restored.')) return; saveDraft().then(() => { const id = S.id; S.id = null; S.tg = null; openSong(id); }); }
  else if (act === 'praat') praatOpen();
  else if (act === 'revert') { if (!confirm('Replace ALL tiers with the original pipeline prediction? (Undo still works.)')) return; checkpoint(); S.tg.tiers = JSON.parse(JSON.stringify(S.orig.tiers)); S.selIv = -1; changed(); updateEditor(); renderBlockTiers(); }
  else if (act === 'discard') {
    if (!confirm('Throw away all unsaved changes and the autosaved draft, and reload the last saved version (or the prediction)?')) return;
    api('/api/discard_draft/' + S.id, {}).then(() => { S.tg.tiers = S.savedJSON !== '__none__' ? JSON.parse(S.savedJSON) : JSON.parse(JSON.stringify(S.orig.tiers)); const id = S.id; S.id = null; S.tg = null; openSong(id); });
  }
}
async function praatOpen() { if (isDirty() || !S.hasCorrected) { if (!await save()) return; } try { const r = await api('/api/praat/' + S.id, {}); toast('Opened in Praat: ' + r.file + '. If you edit there, save over the same file, then File → Reload from disk.'); } catch (e) { toast('Praat: ' + e.message, true); } }
$$('.menu').forEach(m => {
  m.querySelector('.menu-btn').onclick = e => { e.stopPropagation(); const p = m.querySelector('.menu-pop'), was = p.classList.contains('open'); $$('.menu-pop').forEach(x => x.classList.remove('open')); if (!was) p.classList.add('open'); };
  m.querySelectorAll('.menu-pop button').forEach(b => b.onclick = () => { $$('.menu-pop').forEach(x => x.classList.remove('open')); const a = b.dataset.act; a.startsWith('tier-') ? tierAction(a) : fileAction(a); cv.focus(); });
});
document.addEventListener('click', e => $$('.menu-pop').forEach(x => { if (!(x.classList.contains('keep-open') && x.contains(e.target))) x.classList.remove('open'); }));

// ------------------------------------------------------------------ query panel
function statsIn(a, b) {
  const p = S.pitch; if (!p) return null;
  const f0 = Math.max(0, frameAt(a)), f1 = Math.min(p.nf - 1, frameAt(b));
  let n = 0, nv = 0, s = 0, mn = 1e9, mx = 0, sdb = 0;
  for (let f = f0; f <= f1; f++) { n++; sdb += Math.pow(10, p.db[f] / 10); if (isVoiced(f)) { nv++; s += Math.log(p.f0[f]); mn = Math.min(mn, p.f0[f]); mx = Math.max(mx, p.f0[f]); } }
  return { n, nv, mean: nv ? Math.exp(s / nv) : 0, mn, mx, db: n ? 10 * Math.log10(sdb / n) : 0 };
}
function updateQuery() {
  if (!S.tg || !S.pitch) return;
  const rows = [], add = (k, v) => rows.push(`<tr><td>${k}</td><td>${v}</td></tr>`);
  const c = S.cursor, f = clamp(frameAt(c), 0, S.pitch.nf - 1);
  add('cursor', `${fmt(c)} s (song ${fmtMS(c + S.offset)})`);
  add('pitch at cursor', isVoiced(f) ? `${S.pitch.f0[f].toFixed(1)} Hz · ${noteName(S.pitch.f0[f])}` : 'unvoiced');
  add('intensity at cursor', `${S.pitch.db[f].toFixed(1)} dB`);
  if (S.form) { const F = [0, 1, 2, 3].map(k => S.form.data[f * 4 + k]).filter(x => x > 0); add('formants at cursor', F.length ? F.map((x, i) => `F${i + 1} ${Math.round(x)}`).join(' · ') : '—'); }
  const r = selRange();
  if (r) {
    const st = statsIn(r[0], r[1]);
    add('selection', `${fmt(r[0])} – ${fmt(r[1])} s`); add('duration', `${((r[1] - r[0]) * 1000).toFixed(0)} ms`);
    add('mean pitch', st.nv ? `${st.mean.toFixed(1)} Hz · ${noteName(st.mean)}` : '—');
    add('pitch range', st.nv ? `${st.mn.toFixed(0)}–${st.mx.toFixed(0)} Hz (${noteName(st.mn)}–${noteName(st.mx)}, ${(12 * Math.log2(st.mx / st.mn)).toFixed(1)} st)` : '—');
    add('voiced', `${Math.round(100 * st.nv / Math.max(1, st.n))}% of frames`);
    add('mean intensity', `${st.db.toFixed(1)} dB`);
  }
  const t = S.tg.tiers[S.active], it = S.selIv >= 0 ? t.items[S.selIv] : null;
  if (it) add('selected', `${t.name}: “${it.t}”`);
  for (const tn of S.tg.tiers) if (isInt(tn)) { const iv = tn.items[findIv(tn, c)]; add(tn.name + ' at cursor', iv.t ? `“${iv.t}” (${fmt(iv.a)}–${fmt(iv.b)})` : '∅'); }
  $('#q-table').innerHTML = rows.join('');
  drawSlice(c);
}
function drawSlice(t) {
  const cvs = $('#slice'), g = cvs.getContext('2d'), w = cvs.width, h = cvs.height;
  g.clearRect(0, 0, w, h); g.fillStyle = '#fff'; g.fillRect(0, 0, w, h);
  const x = srcX(), N = 2048, win = Math.round(S.sr * 0.04), c = Math.round(t * S.sr);
  const re = new Float32Array(N), im = new Float32Array(N);
  for (let i = 0; i < win; i++) { const k = c - (win >> 1) + i; if (k > 0 && k < x.length) re[i] = (x[k] - 0.95 * x[k - 1]) * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win - 1))); }
  fftMain(re, im);
  const nb = Math.floor(S.set.fmax / (S.sr / N)), db = [];
  let mx = -1e9; for (let b = 0; b < nb; b++) { const v = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-12); db.push(v); if (v > mx) mx = v; }
  g.strokeStyle = '#4a6fa5'; g.beginPath();
  db.forEach((v, b) => { const X = b / nb * w, Y = h - 12 - clamp((v - (mx - 70)) / 70, 0, 1) * (h - 16); b ? g.lineTo(X, Y) : g.moveTo(X, Y); });
  g.stroke();
  g.fillStyle = '#888'; g.font = '10px Consolas'; for (let k = 1000; k < S.set.fmax; k += 1000) { const X = k / S.set.fmax * w; g.fillRect(X, h - 12, 1, 4); g.fillText(k / 1000 + 'k', X - 6, h - 1); }
  g.fillText(`${fmt(t)} s · 70 dB range`, 4, 10);
}
function fftMain(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < len / 2; k++) { const a = i + k, b = a + len / 2, xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr; re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } }
  }
}
function listing(kind) {
  const r = selRange() || [S.view.a, S.view.b], p = S.pitch, hopS = p.hop / S.sr, rows = [];
  rows.push(kind === 'pitch' ? 'time\tf0_Hz' : kind === 'int' ? 'time\tdB' : 'time\tF1\tF2\tF3\tF4');
  for (let f = Math.max(0, frameAt(r[0])); f <= Math.min(p.nf - 1, frameAt(r[1])); f++) {
    const t = ((f + 0.5) * hopS).toFixed(3);
    if (kind === 'pitch') rows.push(`${t}\t${isVoiced(f) ? p.f0[f].toFixed(2) : '--undefined--'}`);
    else if (kind === 'int') rows.push(`${t}\t${p.db[f].toFixed(2)}`);
    else rows.push(t + '\t' + [0, 1, 2, 3].map(k => S.form.data[f * 4 + k] ? S.form.data[f * 4 + k].toFixed(0) : '--').join('\t'));
  }
  navigator.clipboard.writeText(rows.join('\n')).then(() => toast(`Copied ${rows.length - 1} rows to the clipboard`), () => download(`${S.id}_${kind}.tsv`, rows.join('\n'), 'text/plain'));
}
$('#q-pitch-list').onclick = () => listing('pitch');
$('#q-int-list').onclick = () => listing('int');
$('#q-form-list').onclick = () => listing('form');

// ------------------------------------------------------------------ settings
function openSettings() {
  const s = S.set;
  $('#s-src').value = s.src; $('#s-pf').value = s.pitchFloor; $('#s-pc').value = s.pitchCeil; $('#s-vt').value = s.voicing; $('#s-sw').value = s.specWin;
  $('#s-fm').value = s.fmax; $('#s-dr').value = s.dyn; $('#s-fc').value = s.formantMax; $('#s-plo').value = s.plo; $('#s-phi').value = s.phi; $('#s-ul').value = s.unlabMin; $('#s-th').value = s.tierH;
  $('#s-grab').value = s.grabPx; $('#s-snap').value = s.snapPx; $('#s-drag').value = s.dragPx; $('#s-nud').value = s.nudgeMs; $('#s-bnud').value = s.bigNudgeMs; $('#s-wheel').value = s.wheelSpeed; $('#s-zoom').value = s.zoomSpeed;
  $('#settings').classList.remove('hidden');
}
$('#s-reset').onclick = () => { const keep = S.set; S.set = Object.assign({}, DEFAULT_SET); openSettings(); S.set = keep; toast('Defaults filled in: press Apply to use them'); };
$('#s-apply').onclick = async () => {
  const num = (id, lo, hi, d) => { const v = parseFloat($(id).value); return isNaN(v) ? d : clamp(v, lo, hi); };
  S.set = { src: $('#s-src').value, pitchFloor: num('#s-pf', 30, 500, 70), pitchCeil: num('#s-pc', 200, 2000, 1100), voicing: num('#s-vt', 0.3, 0.99, 0.75), specWin: num('#s-sw', 0.002, 0.1, 0.025),
    fmax: num('#s-fm', 1000, 8000, 5000), dyn: num('#s-dr', 20, 100, 50), formantMax: num('#s-fc', 3000, 8000, 5500), plo: num('#s-plo', 30, 500, 70), phi: num('#s-phi', 200, 2000, 1000), unlabMin: num('#s-ul', 0.1, 3, 0.35), tierH: num('#s-th', 26, 90, 44),
    grabPx: num('#s-grab', 2, 20, 6), snapPx: num('#s-snap', 0, 30, 8), dragPx: num('#s-drag', 1, 20, 4), nudgeMs: num('#s-nud', 0.5, 50, 5), bigNudgeMs: num('#s-bnud', 2, 500, 20), wheelSpeed: num('#s-wheel', 0.1, 5, 1), zoomSpeed: num('#s-zoom', 0.1, 5, 1) };
  if (S.set.bigNudgeMs < S.set.nudgeMs) S.set.bigNudgeMs = S.set.nudgeMs;
  localStorage.setItem('p1set', JSON.stringify(S.set));
  $('#settings').classList.add('hidden');
  if (S.x.v) { await analyse(); runChecks(); }
  S.dirtyDraw = true;
};

// ------------------------------------------------------------------ save / state
async function save() {
  if (!S.tg) return false;
  if (document.activeElement === txt) commitText(txt.value, false);
  runChecks();
  const errs = openIssues().filter(i => i.sev === 'error');
  if (errs.length && !confirm(`${errs.length} error(s) still open, e.g.\n• ${errs.slice(0, 3).map(e => e.msg).join('\n• ')}\n\nSave anyway? (you can keep working afterwards)`)) return false;
  try {
    const r = await api('/api/save/' + S.id, toServer(S.tg));
    S.savedJSON = JSON.stringify(S.tg.tiers); S.savedAt = r.at; S.hasCorrected = true; S.draftAt = null;
    updateStatus(); toast('Saved → ' + r.path);
    await saveState(); await loadSongs();
    return true;
  } catch (e) { toast('SAVE FAILED: ' + e.message, true); return false; }
}
let stateTimer = null;
const saveStateSoon = () => { clearTimeout(stateTimer); stateTimer = setTimeout(saveState, 1500); };
const stateObj = () => ({ done: S.done, notes: S.notes, seconds: Math.round(S.seconds), ignored: [...S.ignored], listened: Array.from(S.listened || []).join(''), listened_pct: listenedPct(), issues: openIssues().length });
async function saveState() {
  if (!S.id || !S.listened) return;
  clearTimeout(stateTimer);
  const st = stateObj();
  await api('/api/state/' + S.id, st).catch(() => { });
  const s = S.songs.find(x => x.id === S.id); if (s) Object.assign(s, { done: S.done, listened: st.listened_pct, issues: st.issues, seconds: st.seconds });
}
$('#btn-save').onclick = save;
$('#chk-done').onchange = async e => {
  if (e.target.checked) {
    if (isDirty() || !S.hasCorrected) { if (!await save()) { e.target.checked = false; return; } }
    const open = openIssues(), lp = listenedPct();
    if ((open.length || lp < 95) && !confirm(`Still ${open.length} open check(s), and you have listened to ${lp}% of the sung audio.\nMark as done anyway?`)) { e.target.checked = false; return; }
  }
  S.done = e.target.checked; await saveState(); renderSongList();
};
$('#notes').oninput = e => { S.notes = e.target.value; saveStateSoon(); };
$('#btn-open-folder').onclick = () => { const id = S.id || (S.songs[0] && S.songs[0].id); if (id) api('/api/open_folder/' + id, {}).catch(() => { }); };
window.addEventListener('beforeunload', e => {
  if (S.id && S.tg && isDirty()) navigator.sendBeacon('/api/draft/' + S.id, new Blob([JSON.stringify({ fmt: 2, tg: S.tg, at: Date.now() / 1000 })], { type: 'application/json' }));
  if (S.id && S.listened) navigator.sendBeacon('/api/state/' + S.id, new Blob([JSON.stringify(stateObj())], { type: 'application/json' }));
});
let lastAct = Date.now(), tickN = 0;
['mousedown', 'keydown', 'wheel'].forEach(ev => window.addEventListener(ev, () => lastAct = Date.now(), true));
setInterval(() => {
  if (!S.tg || document.hidden) return;
  if (Date.now() - lastAct < 90000 || P.playing) S.seconds += 1;
  $('#timer').textContent = '⏱ ' + fmtMS(S.seconds);
  if (++tickN % 20 === 0) saveState();
  if (P.playing && tickN % 2 === 0) { const lp = listenedPct(), el = $('#lp'); if (el) { el.textContent = lp + '%'; const bar = $('#issue-summary .bar div'); if (bar) bar.style.width = lp + '%'; } }
}, 1000);

// ------------------------------------------------------------------ toolbar wiring
function setBlend(v) { opt.blend = v; $$('#src-seg button').forEach(b => b.classList.toggle('on', +b.dataset.src === v)); applyGains(); }
function setRate(r) { opt.rate = r; $('#sel-rate').value = String(r); if (P.playing) { const t = playPos(), b = P.b, k = P.kind; play(t, b, k); } toast(r === 1 ? 'Normal speed' : `Speed ${r}× (pitch kept)`); }
$$('#src-seg button').forEach(b => b.onclick = () => setBlend(+b.dataset.src));
$('#sel-rate').onchange = e => { setRate(+e.target.value); e.target.blur(); };
$('#vol').oninput = e => { opt.vol = +e.target.value; applyGains(); };
$('#btn-play').onclick = togglePlay;
// each play button toggles: pressing the one that is playing stops it
const toggleOr = (k, f) => () => { if (P.playing && P.kind === k) stop(); else f(); };
$('#btn-play-sel').onclick = toggleOr('sel', playSelection);
$('#btn-play-view').onclick = toggleOr('view', playView);
$('#btn-play-before').onclick = toggleOr('before', playBefore);
const bindChk = (id, key, cb) => { $(id).checked = opt[key]; $(id).onchange = e => { opt[key] = e.target.checked; S.dirtyDraw = true; cb && cb(); }; };
bindChk('#chk-loop', 'loop'); bindChk('#chk-follow', 'follow'); bindChk('#chk-clickplay', 'clickPlay'); bindChk('#chk-pitch', 'pitch');
bindChk('#chk-sylf', 'sylFollow'); bindChk('#chk-int', 'intensity'); bindChk('#chk-form', 'formants'); bindChk('#chk-allb', 'allB'); bindChk('#chk-ghost', 'ghost');
$('#btn-zin').onclick = () => zoom(0.66);
$('#btn-zout').onclick = () => zoom(1.5);
$('#btn-zsel').onclick = () => { const r = selRange(); if (r) setView(r[0] - 0.05, r[1] + 0.05); };
$('#btn-zback').onclick = zoomBack;
$('#btn-zall').onclick = () => setView(0, S.D);
$('#btn-help').onclick = () => $('#help').classList.remove('hidden');
$('#btn-settings').onclick = openSettings;
$$('.modal-close').forEach(b => b.onclick = () => b.closest('.modal').classList.add('hidden'));
$$('.tabs button').forEach(b => b.onclick = () => { $$('.tabs button').forEach(x => x.classList.toggle('on', x === b)); $$('.tab-body').forEach(x => x.classList.toggle('hidden', x.id !== 'tab-' + b.dataset.tab)); S.queryDirty = true; });
$$('#issue-filters input').forEach(c => c.onchange = renderIssues);
$$('#block-row button[data-dt]').forEach(b => b.onclick = () => { shiftBlock(+b.dataset.dt); cv.focus(); });
$$('#block-row button[data-st]').forEach(b => b.onclick = () => { stretchBlock(+b.dataset.st); cv.focus(); });
$('#blk-copy').onclick = () => { copyBlock(); cv.focus(); };
$('#blk-paste').onclick = () => { pasteBlock(false); cv.focus(); };
$('#blk-paste-fit').onclick = () => { pasteBlock(true); cv.focus(); };
$('#blk-paste-text').onclick = () => { pasteTextIntoIntervals(); cv.focus(); };
// splitter between the display and the editor panel (height remembered)
(() => {
  const ed = $('#editor'), wrap = $('#canvas-wrap'), sp = $('#splitter');
  const apply = h => { if (h == null) { ed.style.height = ''; ed.style.maxHeight = ''; wrap.style.minHeight = ''; return; }
    const main = $('#main').getBoundingClientRect().height, maxH = Math.max(60, main - wrap.getBoundingClientRect().top + $('#main').getBoundingClientRect().top - 150);
    h = clamp(h, 50, maxH); ed.style.height = h + 'px'; ed.style.maxHeight = 'none'; wrap.style.minHeight = '150px'; return h; };
  const saved = parseFloat(localStorage.getItem('p1edH')); if (saved) apply(saved);
  let st = null;
  sp.addEventListener('mousedown', e => { e.preventDefault(); st = { y: e.clientY, h: ed.getBoundingClientRect().height }; sp.classList.add('drag'); document.body.classList.add('rowdrag'); });
  window.addEventListener('mousemove', e => { if (st) apply(st.h - (e.clientY - st.y)); });
  window.addEventListener('mouseup', () => { if (!st) return; st = null; sp.classList.remove('drag'); document.body.classList.remove('rowdrag'); localStorage.setItem('p1edH', ed.getBoundingClientRect().height); });
  sp.addEventListener('dblclick', () => { localStorage.removeItem('p1edH'); apply(null); toast('Layout reset'); });
})();
$('#blk-apply').onclick = () => { const v = parseFloat($('#blk-ms').value); if (!isNaN(v) && v) shiftBlock(v / 1000); cv.focus(); };

// ------------------------------------------------------------------ boot
(async () => {
  try { await loadSongs(); } catch (e) { toast('Server not reachable: ' + e.message, true); return; }
  const want = decodeURIComponent(location.hash.slice(1));
  const first = S.songs.find(s => s.id === want) || S.songs.find(s => !s.done) || S.songs[0];
  $('#loading').textContent = 'Select a song';
  if (first) openSong(first.id);
  window.addEventListener('hashchange', () => { const id = decodeURIComponent(location.hash.slice(1)); if (id && id !== S.id && S.songs.some(s => s.id === id)) openSong(id); });
})();

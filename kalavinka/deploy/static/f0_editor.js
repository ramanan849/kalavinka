/* Kalavinka F0 editor: edit a composition (notes, lyrics, ornaments, hand-drawn pitch) on its F0 curve.

   Plain JS + canvas, no build step. Loaded into the Gradio page through the page's <head>; it mounts itself into
   <div id="kv-editor"> when that appears. It talks to Python through three hidden Gradio textboxes:

     #kv-comp  the composition JSON. The editor writes it after every edit (and fires "input"); Python writes it when a
               new piece is loaded (start from a tune, sargam, JSON ...), and the editor reads it back.
     #kv-view  what to draw, computed by Python (kalavinka_app.compose.view): note boxes and the F0 curve in cents.
     #kv-sung  after a take: the pitch the model actually sang (pYIN), drawn as a thin orange line.

   All the curve maths (ornaments, smoothing, strokes) lives in Python, so the editor never disagrees with what is sung.
   The editor only changes the composition and draws what Python sends back.

   Mouse:  drag a note up/down = pitch (snaps to the raag's swaras; hold Alt for free cents)
           drag a note's right edge = length (snaps to quarter beats; Alt = free)
           click = select (its settings and ornaments appear below the canvas); double-click = edit the syllable
           Pencil: draw the pitch by hand over sung notes; Eraser: drag across drawn pitch to remove it
   Keys:   arrows up/down = next swara, left/right = select previous/next note, +/- = longer/shorter,
           Delete = remove note, Ctrl+Z / Ctrl+Y = undo / redo, Space = preview
*/
(function () {
  "use strict";
  if (window.KV) return;

  const PAD_L = 64, PAD_R = 52, PAD_T = 18, PAD_B = 34, H = 380;
  const COL = { grid: "#e6e6e9", gridSa: "#b9b9c4", txt: "#4a4a55", bar: "#5b8def", barSel: "#e2563b",
                rest: "#d8d8de", curve: "#1f2a44", sung: "#ef8a17", edit: "#2fa37a", beat: "#f1f1f4", sam: "#c9c9d6",
                head: "#e2563b" };

  const KV = {
    comp: null, view: null, sung: null, phrase: 0, sel: -1, mode: "select", snap: "raag",
    pxPerSec: 110, scrollT: 0, undo: [], redo: [], lastSent: "", drag: null, audio: null, playT0: 0, playing: false,
  };
  window.KV = KV;

  // ------------------------------------------------------------------ gradio bridge
  function box(id) {
    const el = document.querySelector(`#${id} textarea, #${id} input`);
    return el;
  }
  function send() {
    const el = box("kv-comp");
    if (!el || !KV.comp) return;
    const s = JSON.stringify(KV.comp);
    KV.lastSent = s;
    el.value = s;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    const b = document.querySelector("#kv-push");
    if (b) b.click();
  }
  KV.load = function (s) {
    if (!s || s === KV.lastSent) return s;
    try { KV.comp = JSON.parse(s); } catch (e) { return s; }
    KV.lastSent = s;
    if (KV.phrase >= KV.comp.phrases.length) KV.phrase = 0;
    KV.sel = -1; KV.sung = null; KV.scrollT = 0;
    build(); draw();
    return s;
  };
  KV.setView = function (s) {
    if (!s) return s;
    try { KV.view = JSON.parse(s); } catch (e) { return s; }
    build(); draw();
    return s;
  };
  KV.setSung = function (s) {
    try { KV.sung = s ? JSON.parse(s) : null; } catch (e) { KV.sung = null; }
    draw();
    return s;
  };

  // ------------------------------------------------------------------ helpers
  const phr = () => KV.comp && KV.comp.phrases[KV.phrase];
  const vph = () => KV.view && KV.view.phrases[KV.phrase];
  const notes = () => (phr() ? phr().notes : []);
  function canvasW() { return KV.canvas ? KV.canvas.width / (window.devicePixelRatio || 1) : 800; }
  function range() {
    const v = vph();
    let lo = -700, hi = 1700;
    if (v) {
      for (const n of v.notes) if (n.c !== null) { lo = Math.min(lo, n.c - 300); hi = Math.max(hi, n.c + 300); }
      for (const c of v.curve) if (c !== null) { lo = Math.min(lo, c - 100); hi = Math.max(hi, c + 100); }
    }
    return [Math.floor(lo / 100) * 100, Math.ceil(hi / 100) * 100];
  }
  function yOf(c) { const [lo, hi] = range(); return PAD_T + (hi - c) / (hi - lo) * (H - PAD_T - PAD_B); }
  function cOf(y) { const [lo, hi] = range(); return hi - (y - PAD_T) / (H - PAD_T - PAD_B) * (hi - lo); }
  function xOf(t) { return PAD_L + (t - KV.scrollT) * KV.pxPerSec; }
  function tOf(x) { return (x - PAD_L) / KV.pxPerSec + KV.scrollT; }
  function nameOf(semis) {
    const names = (KV.view && KV.view.names) || ["Sa","re(k)","Re","ga(k)","Ga","ma","Ma(t)","Pa","dha(k)","Dha","ni(k)","Ni"];
    const o = Math.floor(semis / 12), k = ((semis % 12) + 12) % 12;
    return (o < 0 ? ".".repeat(-o) : "") + names[k] + (o > 0 ? "'".repeat(o) : "");
  }
  function snapCents(c, free) {
    if (free || KV.snap === "free") return Math.round(c / 5) * 5;
    const scale = KV.snap === "raag" && KV.view ? KV.view.scale : [...Array(12).keys()];
    let best = 0, bd = 1e9;
    for (let o = -3; o <= 3; o++) for (const s of scale) {
      const v = 100 * (s + 12 * o), d = Math.abs(v - c);
      if (d < bd) { bd = d; best = v; }
    }
    return best;
  }
  function stepCents(c, dir) {
    const scale = KV.snap === "raag" && KV.view ? KV.view.scale : [...Array(12).keys()];
    const all = [];
    for (let o = -3; o <= 3; o++) for (const s of scale) all.push(100 * (s + 12 * o));
    all.sort((a, b) => a - b);
    const here = Math.round(c / 100) * 100;
    if (dir > 0) return all.find((v) => v > here) ?? here;
    return [...all].reverse().find((v) => v < here) ?? here;
  }
  function setPitch(n, cents) {
    const k = Math.round(cents / 100), rest = Math.round(cents - 100 * k);
    n.swara = nameOf(k);
    delete n.midi; delete n.hz;
    if (rest) n.cents = rest; else delete n.cents;
  }
  function tempo() { return (KV.comp && KV.comp.tempo_bpm) || 80; }
  function setLength(n, sec, free) {
    const beat = 60 / tempo();
    if (free || KV.snap === "free") { n.sec = Math.max(0.04, Math.round(sec * 50) / 50); delete n.beats; }
    else { n.beats = Math.max(0.25, Math.round(sec / beat * 4) / 4); delete n.sec; }
  }
  function snapshot() {
    KV.undo.push(JSON.stringify(KV.comp)); if (KV.undo.length > 100) KV.undo.shift();
    KV.redo = [];
  }
  function commit() { send(); build(); draw(); }

  // ------------------------------------------------------------------ DOM
  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (k === "class") e.className = v;
      else e.setAttribute(k, v);
    }
    for (const k of kids) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
    return e;
  }

  function mount(root) {
    root.innerHTML = "";
    KV.root = root;
    const bar = el("div", { class: "kv-toolbar" });
    const modes = el("span", { class: "kv-group" });
    for (const [m, label, tip] of [["select", "Notes", "drag notes: up/down = pitch, right edge = length"],
                                    ["pencil", "✎ Draw pitch", "draw the pitch curve by hand over sung notes"],
                                    ["eraser", "Erase drawing", "drag across hand-drawn pitch to remove it"]])
      modes.append(el("button", { class: "kv-btn", "data-mode": m, title: tip, onclick: () => { KV.mode = m; build(); draw(); } }, label));
    const snap = el("select", { class: "kv-sel", title: "where dragged notes land", onchange: (e) => { KV.snap = e.target.value; } },
      el("option", { value: "raag" }, "snap: raag swaras"), el("option", { value: "12" }, "snap: 12 swaras"),
      el("option", { value: "free" }, "snap: free (cents)"));
    snap.value = KV.snap;
    const edit = el("span", { class: "kv-group" },
      el("button", { class: "kv-btn", title: "add a note after the selected one", onclick: addNote }, "+ note"),
      el("button", { class: "kv-btn", title: "add a rest after the selected note", onclick: addRest }, "+ rest"),
      el("button", { class: "kv-btn", title: "split the selected note in two (the second half holds the syllable)", onclick: splitNote }, "split"),
      el("button", { class: "kv-btn", title: "delete the selected note", onclick: delNote }, "delete"),
      el("button", { class: "kv-btn", title: "undo (Ctrl+Z)", onclick: doUndo }, "↶"),
      el("button", { class: "kv-btn", title: "redo (Ctrl+Y)", onclick: doRedo }, "↷"),
      el("button", { class: "kv-btn", title: "remove all hand-drawn pitch in this phrase", onclick: clearEdits }, "clear drawing"));
    const view = el("span", { class: "kv-group" },
      el("button", { class: "kv-btn", title: "zoom out", onclick: () => zoom(0.8) }, "zoom −"),
      el("button", { class: "kv-btn", title: "zoom in", onclick: () => zoom(1.25) }, "zoom +"),
      el("button", { class: "kv-btn kv-play", title: "hear the pitch curve as a tone (Space)", onclick: () => preview(false) }, "▶ preview phrase"),
      el("button", { class: "kv-btn kv-play", title: "hear every phrase", onclick: () => preview(true) }, "▶ all"),
      el("button", { class: "kv-btn", onclick: stop }, "■"));
    bar.append(modes, snap, edit, view);
    const tabs = el("div", { class: "kv-tabs" });
    const cwrap = el("div", { class: "kv-canvas-wrap" });
    const canvas = el("canvas", { class: "kv-canvas", tabindex: "0" });
    cwrap.append(canvas);
    const scroll = el("input", { type: "range", class: "kv-scroll", min: "0", max: "100", value: "0",
      oninput: (e) => { const v = vph(); if (!v) return; KV.scrollT = Math.max(0, (v.seconds - visibleSec() + 0.5)) * e.target.value / 100; draw(); } });
    const panel = el("div", { class: "kv-panel" });
    const status = el("div", { class: "kv-status" });
    root.append(bar, tabs, cwrap, scroll, panel, status);
    Object.assign(KV, { canvas, tabs, panel, status, scrollEl: scroll, toolbar: bar });
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("dblclick", onDbl);
    canvas.addEventListener("keydown", onKey);
    canvas.addEventListener("wheel", (e) => { if (!e.shiftKey && !e.ctrlKey) return; e.preventDefault();
      if (e.ctrlKey) zoom(e.deltaY < 0 ? 1.1 : 0.9); else { KV.scrollT = Math.max(0, KV.scrollT + e.deltaY / KV.pxPerSec); draw(); } }, { passive: false });
    new ResizeObserver(() => resize()).observe(cwrap);
    resize();
    // pick up whatever Python already put in the boxes
    const c = box("kv-comp"), v = box("kv-view");
    if (c && c.value) KV.load(c.value);
    if (v && v.value) KV.setView(v.value);
  }
  function visibleSec() { return (canvasW() - PAD_L - PAD_R) / KV.pxPerSec; }
  function zoom(f) { KV.pxPerSec = Math.min(800, Math.max(20, KV.pxPerSec * f)); draw(); }
  function resize() {
    if (!KV.canvas) return;
    const dpr = window.devicePixelRatio || 1, w = KV.canvas.parentElement.clientWidth || 800;
    KV.canvas.width = w * dpr; KV.canvas.height = H * dpr;
    KV.canvas.style.width = w + "px"; KV.canvas.style.height = H + "px";
    KV.canvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  // ------------------------------------------------------------------ toolbar, phrase tabs, note panel
  function build() {
    if (!KV.root || !KV.comp) return;
    for (const b of KV.toolbar.querySelectorAll("[data-mode]")) b.classList.toggle("kv-on", b.dataset.mode === KV.mode);
    KV.canvas.style.cursor = KV.mode === "select" ? "default" : "crosshair";
    KV.tabs.innerHTML = "";
    KV.comp.phrases.forEach((p, i) => KV.tabs.append(el("button", { class: "kv-tab" + (i === KV.phrase ? " kv-on" : ""),
      onclick: () => { KV.phrase = i; KV.sel = -1; KV.scrollT = 0; build(); draw(); } }, `Phrase ${i + 1}`)));
    KV.tabs.append(el("button", { class: "kv-tab", title: "new phrase (one breath; each phrase is sung separately)", onclick: addPhrase }, "+ phrase"));
    if (KV.comp.phrases.length > 1)
      KV.tabs.append(el("button", { class: "kv-tab", title: "delete this phrase", onclick: delPhrase }, "− phrase"));
    buildPanel();
  }

  function field(label, value, onchange, attrs) {
    const inp = el("input", Object.assign({ value: value ?? "", onchange: (e) => onchange(e.target.value) }, attrs || {}));
    return el("label", { class: "kv-field" }, label, inp);
  }
  function parseVal(s) {
    s = String(s).trim();
    if (s === "") return null;
    if (/^[+-]?\d+(\.\d+)?$/.test(s)) return parseFloat(s);
    if (s.startsWith("[")) { try { return JSON.parse(s); } catch (e) { /* keep text */ } }
    return s;
  }
  function buildPanel() {
    const P = KV.panel;
    P.innerHTML = "";
    const ns = notes();
    if (KV.sel < 0 || KV.sel >= ns.length) {
      P.append(el("div", { class: "kv-hint" }, "Click a note to edit its syllable, swara, length and ornaments. " +
        "Drag notes up/down for pitch, drag the right edge for length. Pencil = draw the pitch by hand."));
      return;
    }
    const n = ns[KV.sel];
    const upd = (fn) => { snapshot(); fn(); commit(); };
    const row = el("div", { class: "kv-row" },
      el("b", {}, `Note ${KV.sel + 1}`),
      field("syllable (Hindi; - = hold, _ = rest, empty = swara name)", n.syl, (v) => upd(() => { n.syl = v.trim(); delete n.phones; if (n.syl === "_") { delete n.swara; n.orn = []; } else if (!n.swara) n.swara = "Sa"; }), { size: 8 }),
      field("swara", n.swara ?? "", (v) => upd(() => { n.swara = v.trim(); delete n.midi; delete n.hz; }), { size: 6 }),
      field("cents ±", n.cents ?? 0, (v) => upd(() => { const c = parseFloat(v) || 0; if (c) n.cents = c; else delete n.cents; }), { size: 4 }),
      field("beats", n.beats ?? "", (v) => upd(() => { const b = parseFloat(v); if (b > 0) { n.beats = b; delete n.sec; } }), { size: 4 }),
      field("or seconds", n.sec ?? "", (v) => upd(() => { const s = parseFloat(v); if (s > 0) { n.sec = s; delete n.beats; } else delete n.sec; }), { size: 4 }));
    P.append(row);
    if (n.syl === "_") return;
    const defs = (KV.view && KV.view.ornaments) || {};
    (n.orn || []).forEach((o, k) => {
      const d = defs[o.type] || { defaults: {}, doc: "" };
      const r = el("div", { class: "kv-orn" }, el("b", {}, o.type), el("span", { class: "kv-doc" }, d.doc));
      const keys = new Set([...Object.keys(d.defaults), ...Object.keys(o).filter((x) => x !== "type")]);
      for (const key of keys) {
        if (o.type === "custom" && key === "points") { r.append(el("span", { class: "kv-doc" }, `${(o.points || []).length} points (drawn from the real singer or by you)`)); continue; }
        const cur = key in o ? o[key] : d.defaults[key];
        r.append(field(key, cur === null || cur === undefined ? "" : (Array.isArray(cur) ? JSON.stringify(cur) : cur),
          (v) => upd(() => { const pv = parseVal(v); if (pv === null) delete o[key]; else o[key] = pv; }), { size: 7 }));
      }
      r.append(el("button", { class: "kv-btn", onclick: () => upd(() => n.orn.splice(k, 1)) }, "remove"));
      P.append(r);
    });
    const pick = el("select", { class: "kv-sel" }, el("option", { value: "" }, "+ add ornament …"),
      ...Object.entries(defs).map(([k, d]) => el("option", { value: k }, `${k}: ${d.doc}`)));
    pick.addEventListener("change", (e) => { const t = e.target.value; if (!t) return;
      upd(() => { n.orn = n.orn || []; n.orn.push({ type: t }); }); });
    P.append(el("div", { class: "kv-row" }, pick));
  }

  // ------------------------------------------------------------------ edits
  function addNote() {
    const ns = notes(); if (!phr()) return;
    snapshot();
    const at = KV.sel >= 0 ? KV.sel + 1 : ns.length;
    const ref = ns[Math.min(at - 1, ns.length - 1)];
    ns.splice(at, 0, { syl: "", swara: ref && ref.swara ? ref.swara : "Sa", beats: 1, orn: [] });
    KV.sel = at; commit();
  }
  function addRest() {
    if (!phr()) return; snapshot();
    const at = KV.sel >= 0 ? KV.sel + 1 : notes().length;
    notes().splice(at, 0, { syl: "_", beats: 1, orn: [] }); KV.sel = at; commit();
  }
  function splitNote() {
    const ns = notes(); if (KV.sel < 0) return;
    const n = ns[KV.sel]; snapshot();
    const half = Object.assign({}, n, { syl: n.syl === "_" ? "_" : "-", orn: [] });
    delete half.phones;
    if (n.sec) { n.sec = half.sec = Math.max(0.04, n.sec / 2); } else { n.beats = half.beats = Math.max(0.125, (n.beats || 1) / 2); }
    ns.splice(KV.sel + 1, 0, half); commit();
  }
  function delNote() {
    const ns = notes(); if (KV.sel < 0 || ns.length <= 1) return;
    snapshot(); ns.splice(KV.sel, 1);
    if (ns[0].syl === "-") ns[0].syl = "";
    KV.sel = Math.min(KV.sel, ns.length - 1); commit();
  }
  function addPhrase() {
    snapshot();
    KV.comp.phrases.splice(KV.phrase + 1, 0, { notes: [{ syl: "", swara: "Sa", beats: 1, orn: [] }], f0_edits: [] });
    KV.phrase += 1; KV.sel = 0; commit();
  }
  function delPhrase() {
    if (KV.comp.phrases.length <= 1) return;
    snapshot(); KV.comp.phrases.splice(KV.phrase, 1);
    KV.phrase = Math.max(0, KV.phrase - 1); KV.sel = -1; commit();
  }
  function clearEdits() { if (!phr()) return; snapshot(); phr().f0_edits = []; commit(); }
  function doUndo() { if (!KV.undo.length) return; KV.redo.push(JSON.stringify(KV.comp)); KV.comp = JSON.parse(KV.undo.pop()); KV.sel = -1; commit(); }
  function doRedo() { if (!KV.redo.length) return; KV.undo.push(JSON.stringify(KV.comp)); KV.comp = JSON.parse(KV.redo.pop()); KV.sel = -1; commit(); }

  // ------------------------------------------------------------------ mouse
  function pos(e) { const r = KV.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  function hit(x, y) {
    const v = vph(); if (!v) return [-1, null];
    for (let i = v.notes.length - 1; i >= 0; i--) {
      const b = v.notes[i], x0 = xOf(b.t), x1 = xOf(b.t + b.d);
      const yc = b.c === null ? H - PAD_B - 8 : yOf(b.c);
      if (x >= x0 && x <= x1 + 4 && Math.abs(y - yc) <= 9) return [i, x >= x1 - 6 ? "edge" : "body"];
    }
    for (let i = 0; i < v.notes.length; i++) {
      const b = v.notes[i];
      if (x >= xOf(b.t) && x <= xOf(b.t + b.d)) return [i, "column"];
    }
    return [-1, null];
  }
  function onDown(e) {
    if (!KV.view || !KV.comp) return;
    KV.canvas.focus();
    const [x, y] = pos(e);
    if (KV.mode === "pencil") { KV.drag = { kind: "pencil", pts: [[tOf(x), cOf(y)]] }; draw(); return; }
    if (KV.mode === "eraser") { KV.drag = { kind: "eraser", t0: tOf(x), t1: tOf(x) }; draw(); return; }
    const [i, part] = hit(x, y);
    KV.sel = i;
    if (i >= 0 && part !== "column") {
      const b = vph().notes[i];
      KV.drag = { kind: part === "edge" ? "len" : "pitch", i, x0: x, y0: y, c0: b.c, d0: b.d, t: b.t, moved: false,
                  c: b.c, d: b.d };
    }
    buildPanel(); draw();
  }
  function onMove(e) {
    const D = KV.drag; if (!D || !KV.canvas) return;
    const [x, y] = pos(e);
    if (D.kind === "pencil") { D.pts.push([tOf(x), cOf(y)]); draw(); return; }
    if (D.kind === "eraser") { D.t1 = tOf(x); draw(); return; }
    if (Math.abs(x - D.x0) + Math.abs(y - D.y0) > 3) D.moved = true;
    if (D.kind === "pitch" && D.c0 !== null) D.c = snapCents(D.c0 + (cOf(y) - cOf(D.y0)), e.altKey);
    if (D.kind === "len") {
      const sec = Math.max(0.04, D.d0 + (x - D.x0) / KV.pxPerSec);
      const beat = 60 / tempo();
      D.d = (e.altKey || KV.snap === "free") ? sec : Math.max(0.25, Math.round(sec / beat * 4) / 4) * beat;
      D.alt = e.altKey;
    }
    draw();
  }
  function onUp(e) {
    const D = KV.drag; if (!D) return;
    KV.drag = null;
    const ph = phr();
    if (D.kind === "pencil") {
      const pts = D.pts.filter((p) => p[0] >= 0).sort((a, b) => a[0] - b[0]);
      if (pts.length >= 2) { snapshot(); ph.f0_edits = ph.f0_edits || [];
        ph.f0_edits.push(pts.map((p) => [Math.round(p[0] * 1000) / 1000, Math.round(p[1])])); commit(); }
      else draw();
      return;
    }
    if (D.kind === "eraser") {
      const a = Math.min(D.t0, D.t1), b = Math.max(D.t0, D.t1);
      const out = [];
      for (const s of ph.f0_edits || []) {
        let cur = [];
        for (const p of s) { if (p[0] >= a && p[0] <= b) { if (cur.length >= 2) out.push(cur); cur = []; } else cur.push(p); }
        if (cur.length >= 2) out.push(cur);
      }
      snapshot(); ph.f0_edits = out; commit();
      return;
    }
    if (!D.moved) { draw(); return; }
    const n = ph.notes[D.i];
    snapshot();
    if (D.kind === "pitch" && D.c !== null && D.c !== D.c0) setPitch(n, D.c);
    if (D.kind === "len") setLength(n, D.d, D.alt);
    commit();
  }
  function onDbl(e) {
    const [x, y] = pos(e); const [i] = hit(x, y);
    if (i < 0 || KV.mode !== "select") return;
    const n = notes()[i];
    const s = window.prompt("Syllable for this note (Hindi; - = hold the previous one, _ = rest, empty = sing the swara name):", n.syl);
    if (s === null) return;
    snapshot(); n.syl = s.trim(); delete n.phones;
    if (n.syl === "_") { delete n.swara; n.orn = []; } else if (!n.swara) n.swara = "Sa";
    commit();
  }
  function onKey(e) {
    const ns = notes();
    if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); doUndo(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key === "y") { e.preventDefault(); doRedo(); return; }
    if (e.key === " ") { e.preventDefault(); KV.playing ? stop() : preview(false); return; }
    if (KV.sel < 0) return;
    const n = ns[KV.sel], b = vph() && vph().notes[KV.sel];
    if (e.key === "ArrowLeft") { KV.sel = Math.max(0, KV.sel - 1); }
    else if (e.key === "ArrowRight") { KV.sel = Math.min(ns.length - 1, KV.sel + 1); }
    else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && b && b.c !== null) {
      snapshot(); setPitch(n, stepCents(b.c, e.key === "ArrowUp" ? 1 : -1)); commit(); e.preventDefault(); return;
    } else if ((e.key === "+" || e.key === "=" || e.key === "-") && b) {
      snapshot(); setLength(n, b.d + (e.key === "-" ? -1 : 1) * 15 / tempo(), false); commit(); return;
    } else if (e.key === "Delete" || e.key === "Backspace") { delNote(); return; }
    else return;
    e.preventDefault(); buildPanel(); draw();
  }

  // ------------------------------------------------------------------ drawing
  function draw() {
    const cv = KV.canvas; if (!cv) return;
    const g = cv.getContext("2d"), W = canvasW();
    g.clearRect(0, 0, W, H);
    g.fillStyle = "#fff"; g.fillRect(0, 0, W, H);
    const v = vph();
    if (!v || !KV.view) {
      g.fillStyle = COL.txt; g.font = "14px sans-serif";
      g.fillText("Load something to edit: pick a starting point above and press Load.", PAD_L, H / 2);
      return;
    }
    const [lo, hi] = range(), scale = new Set(KV.view.scale), sa = KV.view.sa_hz;
    // beat grid
    const beat = 60 / tempo(), taal = KV.view.taal || 0;
    for (let k = 0; k * beat <= v.seconds + 4; k++) {
      const x = xOf(k * beat); if (x < PAD_L || x > W - PAD_R) continue;
      g.strokeStyle = taal && k % taal === 0 ? COL.sam : COL.beat; g.lineWidth = taal && k % taal === 0 ? 1.5 : 1;
      g.beginPath(); g.moveTo(x, PAD_T); g.lineTo(x, H - PAD_B); g.stroke();
    }
    // swara grid
    g.font = "11px sans-serif"; g.textBaseline = "middle";
    for (let c = lo; c <= hi; c += 100) {
      const k = Math.round(c / 100), pc = ((k % 12) + 12) % 12, y = yOf(c);
      const inRaag = scale.has(pc);
      g.strokeStyle = pc === 0 ? COL.gridSa : COL.grid; g.lineWidth = pc === 0 ? 1.5 : 1;
      g.setLineDash(inRaag ? [] : [2, 4]);
      g.beginPath(); g.moveTo(PAD_L, y); g.lineTo(W - PAD_R, y); g.stroke(); g.setLineDash([]);
      g.fillStyle = inRaag ? COL.txt : "#b0b0b8";
      g.font = (pc === 0 ? "bold " : "") + "11px sans-serif";
      g.textAlign = "right"; g.fillText(nameOf(k), PAD_L - 6, y);
      if (pc === 0 || pc === 7) { g.textAlign = "left"; g.fillStyle = "#9a9aa5";
        g.fillText(Math.round(sa * Math.pow(2, c / 1200)) + " Hz", W - PAD_R + 4, y); }
    }
    g.save(); g.beginPath(); g.rect(PAD_L, 0, W - PAD_L - PAD_R, H); g.clip();
    // note bars + lyrics
    v.notes.forEach((b, i) => {
      const D = KV.drag && KV.drag.i === i ? KV.drag : null;
      const c = D && D.kind === "pitch" ? D.c : b.c, d = D && D.kind === "len" ? D.d : b.d;
      const x0 = xOf(b.t), x1 = xOf(b.t + d);
      if (c === null) {
        g.fillStyle = COL.rest; g.fillRect(x0 + 1, H - PAD_B - 12, x1 - x0 - 2, 8);
        return;
      }
      const y = yOf(c), sel = i === KV.sel;
      g.fillStyle = sel ? COL.barSel : COL.bar; g.globalAlpha = sel ? 0.55 : 0.32;
      roundRect(g, x0 + 1, y - 7, Math.max(2, x1 - x0 - 2), 14, 4); g.fill(); g.globalAlpha = 1;
      g.fillStyle = sel ? COL.barSel : COL.bar; g.fillRect(x1 - 4, y - 7, 3, 14);
      g.fillStyle = "#111"; g.font = "15px 'Noto Sans Devanagari', sans-serif"; g.textAlign = "left"; g.textBaseline = "alphabetic";
      g.fillText(b.label, x0 + 3, y - 11);
      g.font = "10px sans-serif"; g.fillStyle = "#6a6a78";
      const sw = D && D.kind === "pitch" ? nameOf(Math.round(c / 100)) : b.swara;
      g.fillText(sw + (b.orn.length ? "  ✦ " + b.orn.join(", ") : ""), x0 + 3, y + 18);
    });
    // computed curve
    plot(g, v.curve.map((c, k) => [k / 50, c]), COL.curve, 2);
    // hand-drawn strokes
    for (const s of phr().f0_edits || []) plot(g, s, COL.edit, 1, true);
    // what the model sang
    if (KV.sung) for (const p of KV.sung) if (p.phrase === KV.phrase) plot(g, p.cents.map((c, k) => [p.t0 + k / 50, c]), COL.sung, 1.5);
    // live drag feedback
    const D = KV.drag;
    if (D && D.kind === "pencil") plot(g, D.pts, COL.edit, 2);
    if (D && D.kind === "eraser") { g.fillStyle = "rgba(226,86,59,0.12)"; g.fillRect(xOf(Math.min(D.t0, D.t1)), PAD_T, Math.abs(D.t1 - D.t0) * KV.pxPerSec, H - PAD_T - PAD_B); }
    // playhead
    if (KV.playing && KV.audio) {
      const t = KV.audio.currentTime - KV.playT0 + (KV.playOffset || 0);
      if (t >= 0 && KV.playPhrase === KV.phrase) { g.strokeStyle = COL.head; g.lineWidth = 1.5; g.beginPath(); g.moveTo(xOf(t), PAD_T); g.lineTo(xOf(t), H - PAD_B); g.stroke(); }
    }
    g.restore();
    // time axis
    g.fillStyle = COL.txt; g.font = "11px sans-serif"; g.textAlign = "center"; g.textBaseline = "top";
    const step = KV.pxPerSec > 150 ? 0.5 : KV.pxPerSec > 50 ? 1 : 2;
    for (let t = Math.ceil(KV.scrollT / step) * step; xOf(t) <= W - PAD_R; t += step) g.fillText(t.toFixed(step < 1 ? 1 : 0) + " s", xOf(t), H - PAD_B + 6);
    // legend + status
    g.textAlign = "left"; g.textBaseline = "top"; g.font = "11px sans-serif";
    let lx = PAD_L;
    for (const [c, t] of [[COL.curve, "pitch to sing"], [COL.edit, "your drawing"], [COL.sung, "what the model sang"]]) {
      g.fillStyle = c; g.fillRect(lx, 4, 14, 3); g.fillStyle = COL.txt; g.fillText(t, lx + 18, 0); lx += 30 + g.measureText(t).width;
    }
    if (KV.status) KV.status.textContent = `Sa = ${KV.view.sa_hz} Hz · tempo ${tempo()} bpm · phrase ${KV.phrase + 1}/${KV.comp.phrases.length}: ` +
      `${v.notes.length} notes, ${v.seconds.toFixed(2)} s` + (KV.sel >= 0 ? ` · note ${KV.sel + 1} selected` : "");
    if (KV.scrollEl) { const span = Math.max(0, v.seconds - visibleSec() + 0.5); KV.scrollEl.style.visibility = span > 0 ? "visible" : "hidden"; }
  }
  function plot(g, pts, color, w, dots) {
    g.strokeStyle = color; g.lineWidth = w; g.beginPath();
    let pen = false;
    for (const [t, c] of pts) {
      if (c === null || c === undefined) { pen = false; continue; }
      const x = xOf(t), y = yOf(c);
      if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
    }
    g.stroke();
    if (dots) { g.fillStyle = color; for (const [t, c] of pts) if (c !== null) { g.beginPath(); g.arc(xOf(t), yOf(c), 1.6, 0, 7); g.fill(); } }
  }
  function roundRect(g, x, y, w, h, r) {
    g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }

  // ------------------------------------------------------------------ preview (a voice-like tone in the browser)
  function preview(all) {
    if (!KV.view) return;
    stop();
    const A = KV.audio || (KV.audio = new (window.AudioContext || window.webkitAudioContext)());
    if (A.state === "suspended") A.resume();
    const sa = KV.view.sa_hz, list = all ? KV.view.phrases.map((p, i) => [i, p]) : [[KV.phrase, vph()]];
    let t = A.currentTime + 0.1;
    KV.nodes = [];
    const start = t;
    KV.playStarts = [];
    for (const [pi, p] of list) {
      KV.playStarts.push([pi, t]);
      const curve = p.curve;
      let k = 0;
      while (k < curve.length) {
        if (curve[k] === null) { k++; continue; }
        let j = k; while (j < curve.length && curve[j] !== null) j++;
        const hz = new Float32Array(curve.slice(k, j).map((c) => sa * Math.pow(2, c / 1200)));
        const t0 = t + k / 50, dur = (j - k) / 50;
        if (dur >= 0.04) {
          const o = A.createOscillator(); o.type = "sawtooth";
          const f1 = A.createBiquadFilter(); f1.type = "bandpass"; f1.frequency.value = 750; f1.Q.value = 3;
          const f2 = A.createBiquadFilter(); f2.type = "bandpass"; f2.frequency.value = 1250; f2.Q.value = 4;
          const lp = A.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 3000;
          const gn = A.createGain(); gn.gain.setValueAtTime(0, t0);
          gn.gain.linearRampToValueAtTime(0.25, t0 + 0.03); gn.gain.setValueAtTime(0.25, t0 + dur - 0.04);
          gn.gain.linearRampToValueAtTime(0, t0 + dur);
          o.frequency.setValueCurveAtTime(hz, t0, dur);
          o.connect(f1); o.connect(f2); f1.connect(lp); f2.connect(lp); lp.connect(gn); gn.connect(A.destination);
          o.start(t0); o.stop(t0 + dur + 0.02);
          KV.nodes.push(o);
        }
        k = j;
      }
      t += p.seconds + 0.3;
    }
    KV.playing = true; KV.playEnd = t;
    const tick = () => {
      if (!KV.playing) return;
      const now = A.currentTime;
      let cur = null;
      for (const [pi, st] of KV.playStarts) if (now >= st) cur = [pi, st];
      if (cur) { KV.playPhrase = cur[0]; KV.playT0 = cur[1]; KV.playOffset = 0;
        if (cur[0] !== KV.phrase && all) { KV.phrase = cur[0]; build(); } }
      if (now > KV.playEnd) { KV.playing = false; }
      draw();
      if (KV.playing) requestAnimationFrame(tick);
    };
    KV.playPhrase = list[0][0]; KV.playT0 = start; KV.playOffset = 0;
    requestAnimationFrame(tick);
  }
  function stop() {
    for (const o of KV.nodes || []) { try { o.stop(); } catch (e) { /* already stopped */ } }
    KV.nodes = []; KV.playing = false; draw();
  }

  // ------------------------------------------------------------------ mount when Gradio renders the div
  function tryMount() {
    const root = document.getElementById("kv-editor");
    if (root && root !== KV.root && root.isConnected) mount(root);
  }
  new MutationObserver(tryMount).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", tryMount);
  tryMount();
})();

// Web worker: spectrogram, pitch (McLeod NSDF), intensity, LPC formants.
'use strict';
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

function spectrogram(x, sr, win, hop, fmax) {
  const N = Math.max(256, 1 << Math.ceil(Math.log2(win)));
  const nb = Math.min(N / 2, Math.ceil(fmax / (sr / N)));
  const nf = Math.floor(x.length / hop);
  const out = new Float32Array(nf * nb), w = new Float32Array(win);
  for (let i = 0; i < win; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win - 1));
  const re = new Float32Array(N), im = new Float32Array(N);
  let mx = -1e9;
  for (let f = 0; f < nf; f++) {
    re.fill(0); im.fill(0);
    const c = Math.round(f * hop + hop / 2 - win / 2);
    for (let i = 0; i < win; i++) { const k = c + i; if (k > 0 && k < x.length) re[i] = (x[k] - 0.95 * x[k - 1]) * w[i]; }
    fft(re, im);
    for (let b = 0; b < nb; b++) { const v = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-12); out[f * nb + b] = v; if (v > mx) mx = v; }
  }
  return { data: out, nf, nb, binHz: sr / N, hop, max: mx };
}

function pitchAndIntensity(x, sr, hop, fmin, fmax) {
  const W = Math.round(sr * Math.max(0.03, 2.5 / fmin));
  const N = 1 << Math.ceil(Math.log2(2 * W));
  const tmin = Math.floor(sr / fmax), tmax = Math.min(W - 2, Math.ceil(sr / fmin));
  const nf = Math.floor(x.length / hop);
  const f0 = new Float32Array(nf), clar = new Float32Array(nf), rms = new Float32Array(nf), db = new Float32Array(nf);
  const re = new Float32Array(N), im = new Float32Array(N), seg = new Float32Array(W), nsdf = new Float32Array(tmax + 2);
  for (let f = 0; f < nf; f++) {
    const s0 = Math.round(f * hop + hop / 2 - W / 2);
    let e = 0;
    for (let i = 0; i < W; i++) { const k = s0 + i; const v = (k >= 0 && k < x.length) ? x[k] : 0; seg[i] = v; e += v * v; }
    rms[f] = Math.sqrt(e / W);
    db[f] = 20 * Math.log10(rms[f] + 1e-9) + 94; // ~dB SPL if full scale = 94 dB (relative)
    if (e < 1e-7) continue;
    re.fill(0); im.fill(0); re.set(seg); fft(re, im);
    for (let i = 0; i < N; i++) { re[i] = re[i] * re[i] + im[i] * im[i]; im[i] = 0; }
    fft(re, im);
    let m = 2 * e;
    for (let t = 0; t <= tmax + 1; t++) {
      if (t > 0) m -= seg[t - 1] * seg[t - 1] + seg[W - t] * seg[W - t];
      nsdf[t] = m > 1e-9 ? 2 * (re[t] / N) / m : 0;
    }
    let t = 1; while (t < tmax && nsdf[t] > 0) t++;
    let best = -1, bestT = 0; const peaks = [];
    while (t < tmax) {
      while (t < tmax && nsdf[t] <= 0) t++;
      let pk = t, pv = -1;
      while (t < tmax && nsdf[t] > 0) { if (nsdf[t] > pv) { pv = nsdf[t]; pk = t; } t++; }
      if (pv > 0) peaks.push([pk, pv]);
      if (pv > best) best = pv;
    }
    if (best <= 0) continue;
    for (const [pk, pv] of peaks) if (pv >= 0.9 * best && pk >= tmin) { bestT = pk; break; }
    if (!bestT) continue;
    const a = nsdf[bestT - 1], b = nsdf[bestT], c = nsdf[bestT + 1], d = a - 2 * b + c;
    f0[f] = sr / (bestT + (d !== 0 ? 0.5 * (a - c) / d : 0)); clar[f] = b;
  }
  return { f0, clar, rms, db, hop, nf };
}

// LPC formants (autocorrelation + Levinson, peaks of the LPC envelope)
function formants(x, sr, hop, maxF) {
  const dec = Math.max(1, Math.floor(sr / (2 * maxF)));
  const sr2 = sr / dec, n2 = Math.floor(x.length / dec), y = new Float32Array(n2);
  for (let i = 0; i < n2; i++) { let s = 0; for (let j = 0; j < dec; j++) s += x[i * dec + j]; y[i] = s / dec; }
  const pre = Math.exp(-2 * Math.PI * 50 / sr2);
  for (let i = n2 - 1; i > 0; i--) y[i] -= pre * y[i - 1];
  const W = Math.round(sr2 * 0.025), p = 12, hop2 = hop / dec, nf = Math.floor(x.length / hop);
  const out = new Float32Array(nf * 4), w = new Float32Array(W), seg = new Float32Array(W);
  for (let i = 0; i < W; i++) w[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (W - 1));
  const NE = 256, cosT = [], sinT = [];
  for (let k = 0; k < NE; k++) { const om = Math.PI * (k / NE) * (Math.min(maxF, sr2 / 2) / (sr2 / 2)); cosT.push([]); sinT.push([]); for (let j = 0; j <= p; j++) { cosT[k].push(Math.cos(om * j)); sinT[k].push(Math.sin(om * j)); } }
  const r = new Float64Array(p + 1), A = new Float64Array(p + 1), tmp = new Float64Array(p + 1), env = new Float32Array(NE);
  for (let f = 0; f < nf; f++) {
    const s0 = Math.round(f * hop2 + hop2 / 2 - W / 2);
    let e = 0;
    for (let i = 0; i < W; i++) { const k = s0 + i; const v = k >= 0 && k < n2 ? y[k] * w[i] : 0; seg[i] = v; e += v * v; }
    if (e < 1e-8) continue;
    for (let l = 0; l <= p; l++) { let s = 0; for (let i = l; i < W; i++) s += seg[i] * seg[i - l]; r[l] = s; }
    r[0] *= 1.0001;
    A.fill(0); A[0] = 1; let E = r[0];
    for (let i = 1; i <= p; i++) {
      let acc = r[i]; for (let j = 1; j < i; j++) acc += A[j] * r[i - j];
      const k = -acc / E;
      tmp.set(A); for (let j = 1; j < i; j++) A[j] = tmp[j] + k * tmp[i - j];
      A[i] = k; E *= (1 - k * k); if (E <= 0) break;
    }
    for (let k = 0; k < NE; k++) { let re = 0, im = 0; for (let j = 0; j <= p; j++) { re += A[j] * cosT[k][j]; im -= A[j] * sinT[k][j]; } env[k] = -Math.log(re * re + im * im + 1e-12); }
    let nfm = 0;
    for (let k = 1; k < NE - 1 && nfm < 4; k++) if (env[k] > env[k - 1] && env[k] >= env[k + 1]) {
      const a = env[k - 1], b = env[k], c = env[k + 1], d = a - 2 * b + c, off = d !== 0 ? 0.5 * (a - c) / d : 0;
      const hz = (k + off) / NE * Math.min(maxF, sr2 / 2);
      if (hz > 90) out[f * 4 + nfm++] = hz;
    }
  }
  return { data: out, nf, hop };
}

onmessage = e => {
  const { x, sr, set } = e.data;
  const spec = spectrogram(x, sr, Math.round(sr * set.specWin), Math.round(sr * 0.005), 8000);
  const hop = Math.round(sr * 0.01);
  const pitch = pitchAndIntensity(x, sr, hop, set.pitchFloor, set.pitchCeil);
  const form = formants(x, sr, hop, set.formantMax);
  postMessage({ spec, pitch, form }, [spec.data.buffer, pitch.f0.buffer, pitch.clar.buffer, pitch.rms.buffer, pitch.db.buffer, form.data.buffer]);
};

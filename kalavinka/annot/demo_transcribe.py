"""Turn passages of real songs into Compose JSON (the app's composition format), and check hand-written compositions
against what the singer actually sang.

Why: for the demo we want Kalavinka to sing well-known passages (a sargam climax, a mukhda) from compositions the
user writes in the Compose tab. Writing them by ear gets the shape right but often misses the tonic, a komal/shuddh
swara, the octave, or the rhythm. The labelling pipeline already has everything needed to measure this: the vocal
separator, the 3-tracker F0 ensemble and the tonic estimator. This module wraps them for single songs outside the
dataset (no manifest, no dataset work dir).

Subcommands
-----------
``analyse``     separate the lead vocal (Mel-RoFormer + de-reverb, then the karaoke model), F0 ensemble at 50 fps
                (RMVPE + FCPE + YIN), tonic (pitch-class histogram), and an overview PDF of the pitch in swaras over
                the whole song; ``--asr`` adds Whisper-large-v3 transcripts with timestamps (to find passages).
                Writes ``<out>/{lead.flac, vocals.flac, f0.npz, analysis.json, overview.pdf, asr.json}``.
``transcribe``  one passage (``--start``/``--end`` s) -> a composition: notes from stable pitch segments snapped to
                the raag's swaras, split at syllable onsets, durations on a beat grid at ``--tempo``; held notes get
                vibrato, glides between notes a meend, short touches before a note a kan. ``--syl`` gives one lyric
                syllable per note (space-separated); otherwise each note sings its swara name.
``compare``     a composition against the real singing: each phrase is aligned to the song (``--starts``, or found by
                searching near the previous phrase), and every note's designed swara is compared with the median sung
                pitch over its span. Prints a per-phrase table and writes ``--fixed`` (the composition with wrong
                swaras replaced and the tonic and tempo set) when asked.

Usage (annotation venv; the GPU is used by ``analyse`` only)::

    python -m kalavinka.annot.demo_transcribe analyse --audio song.opus --out <dir> [--asr]
    python -m kalavinka.annot.demo_transcribe transcribe --analysis <dir> --start 245.0 --end 262.5 --tempo 112 \\
        --raag Desh --title "..." --out passage.json
    python -m kalavinka.annot.demo_transcribe compare --analysis <dir> --comp mine.json --start 245.0 \\
        [--fixed mine_fixed.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FPS, SR = 50, 24000


# ------------------------------------------------------------------------------------------------ analyse (GPU)
def analyse(audio: Path, out: Path, device: str = "cuda", asr: bool = False) -> Dict:
    """Separate, track F0, estimate the tonic; write the files listed in the module docstring."""
    import torch
    from saregama.audio import resample
    from saregama.config import F0Cfg, SeparationCfg
    from saregama.p1.config import DEFAULTS
    from saregama.p1.gpu_stages import _mono
    from saregama.p1.state import atomic_flac, atomic_npz
    from saregama.stages.f0 import F0Estimator
    from saregama.stages.separate import Separator
    from saregama.stages.tonic import estimate_tonic
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "lead.flac").exists():
        full = Separator(SeparationCfg(sep_model="vocals", dereverb=True, device=device))._load_soulx()
        lead = Separator(SeparationCfg(sep_model=DEFAULTS["separation"]["lead_model"], dereverb=False,
                                       device=device))._load_soulx()
        res = full.process(str(audio), verbose=False)
        sr = int(res.sample_rate)
        voc = np.asarray(res.vocals_dereverbed if res.vocals_dereverbed is not None else res.vocals, np.float32)
        voc_st = voc if voc.ndim == 2 else np.stack([voc, voc])
        from preprocess.tools.vocal_separation.model import process as sep_process  # type: ignore
        lead_arr = sep_process(voc_st, lead.sep_model, lead.args, lead.sep_config, torch.device(device))
        v24 = resample(_mono(voc), sr, SR)
        l24 = resample(_mono(lead_arr), sr, SR)[: len(v24)]
        l24 = np.pad(l24, (0, len(v24) - len(l24)))
        gain = min(10 ** (-20 / 20) / (float(np.sqrt(np.mean(v24 ** 2))) or 1e-9),
                   0.99 / max(float(np.abs(v24).max()), 1e-9))
        atomic_flac(out / "vocals.flac", v24 * gain, SR)
        atomic_flac(out / "lead.flac", l24 * gain, SR)
        del full, lead
        torch.cuda.empty_cache()
    from saregama.audio import load_audio
    lead_w = load_audio(out / "lead.flac", SR)
    c = DEFAULTS["f0"]
    est = F0Estimator(F0Cfg(backends=list(c["backends"]), fmin=c["fmin"], fmax=c["fmax"],
                            agree_cents=c["agree_cents"], device=device), fps=FPS)
    r = est(lead_w, SR)
    hz, agree = np.asarray(r["f0"], np.float32), np.asarray(r["f0_agree"], np.float32)
    tonic, conf = estimate_tonic(hz)
    atomic_npz(out / "f0.npz", f0=hz, agree=agree, fps=FPS)
    info = {"audio": str(audio), "seconds": round(len(lead_w) / SR, 2), "tonic_hz": round(float(tonic), 2),
            "tonic_conf": round(float(conf), 3), "tonic_note": hz_name(tonic), "voiced": round(float((hz > 0).mean()), 3)}
    if asr:
        info["asr_segments"] = run_asr(lead_w, out / "asr.json", device)
    (out / "analysis.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    overview_pdf(hz, tonic, out / "overview.pdf", title=audio.stem[:80],
                 asr=json.loads((out / "asr.json").read_text()) if (out / "asr.json").exists() else None)
    return info


def run_asr(wav24: np.ndarray, path: Path, device: str, win: float = 6.0, hop: float = 5.0) -> int:
    """Whisper-large-v3 (Hindi) over the lead vocal in short windows (6 s every 5 s) -> ``asr.json``. Whole-song
    long-form transcription loops on repetitive singing (sargam, refrains) and loses its timestamps, so each window
    is transcribed on its own, with repeated 6-grams blocked; silent windows are skipped."""
    import librosa
    import torch
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", dtype=torch.float16,
                   device=device if device != "cuda" else "cuda:0")
    x = librosa.resample(np.asarray(wav24, np.float32), orig_sr=SR, target_sr=16000)
    segs = []
    for t in np.arange(0, max(len(x) / 16000 - 1, 0), hop):
        seg = x[int(t * 16000):int((t + win) * 16000)]
        if np.sqrt(np.mean(seg ** 2)) < 1e-3:
            continue
        txt = asr({"raw": seg, "sampling_rate": 16000},
                  generate_kwargs={"language": "hi", "task": "transcribe", "no_repeat_ngram_size": 6})["text"].strip()
        if txt:
            segs.append({"start": float(t), "end": float(t + win), "text": txt})
    path.write_text(json.dumps(segs, ensure_ascii=False, indent=1))
    return len(segs)


SWARA_NAMES = ["Sa", "re(k)", "Re", "ga(k)", "Ga", "ma", "Ma(t)", "Pa", "dha(k)", "Dha", "ni(k)", "Ni"]


def hz_name(hz: float) -> str:
    """Western note name of a frequency (for the tonic)."""
    if hz <= 0:
        return "?"
    m = 69 + 12 * np.log2(hz / 440.0)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    k = int(round(m))
    return f"{names[k % 12]}{k // 12 - 1}{'+' if m - k > 0 else '-'}{abs(m - k) * 100:.0f}c"


def overview_pdf(hz: np.ndarray, tonic: float, path: Path, title: str, asr: Optional[List[Dict]] = None,
                 panel_s: float = 30.0) -> None:
    """Pitch of the lead vocal in cents above Sa, one 30 s panel per row, swara grid lines, ASR text on top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    c = np.where(hz > 0, 1200 * np.log2(np.maximum(hz, 1e-9) / tonic), np.nan)
    t = np.arange(len(hz)) / FPS
    lo = np.nanpercentile(c, 1) - 150 if np.isfinite(c).any() else -1200
    hi = np.nanpercentile(c, 99) + 150 if np.isfinite(c).any() else 2400
    n_pan = int(np.ceil(t[-1] / panel_s)) if len(t) else 1
    with PdfPages(path) as pdf:
        for p0 in range(0, n_pan, 5):
            fig, axs = plt.subplots(5, 1, figsize=(11.7, 8.3))
            for k, ax in enumerate(axs):
                i = p0 + k
                a, b = i * panel_s, (i + 1) * panel_s
                m = (t >= a) & (t < b)
                for s in range(int(np.floor(lo / 100)), int(np.ceil(hi / 100)) + 1):
                    name = SWARA_NAMES[s % 12] + ("'" * (s // 12) if s >= 12 else "." * (-(s // 12)) if s < 0 else "")
                    ax.axhline(s * 100, color="#e0e0e0" if s % 12 not in (0, 7) else "#b0b0b0", lw=0.5)
                    if (s % 12) in (0, 4, 7, 9, 11, 2, 5) and lo <= s * 100 <= hi:
                        ax.text(a - 0.2, s * 100, name, fontsize=5, ha="right", va="center", color="#777")
                ax.plot(t[m], c[m], color="#1F4E79", lw=0.8)
                for seg in asr or []:
                    if seg["start"] is not None and a <= seg["start"] < b:
                        ax.text(seg["start"], hi - 60, seg["text"][:60], fontsize=5, color="#9E1B32", va="top")
                ax.set_xlim(a, b)
                ax.set_ylim(lo, hi)
                ax.tick_params(labelsize=6)
                ax.set_ylabel("cents above Sa", fontsize=6)
            axs[0].set_title(f"{title}  (Sa = {tonic:.1f} Hz, {hz_name(tonic)})", fontsize=8)
            axs[-1].set_xlabel("time (s)", fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ------------------------------------------------------------------------------------------- shared helpers (CPU)
def load(analysis: Path) -> Tuple[np.ndarray, float, Dict]:
    """(F0 in Hz at 50 fps, tonic Hz, analysis.json) of an analysed song; ``--tonic`` overrides are applied later."""
    z = np.load(analysis / "f0.npz")
    info = json.loads((analysis / "analysis.json").read_text())
    return np.asarray(z["f0"], float), float(info["tonic_hz"]), info


def cents_of(hz: np.ndarray, tonic: float) -> np.ndarray:
    """Cents above Sa (NaN where unvoiced)."""
    return np.where(hz > 0, 1200 * np.log2(np.maximum(hz, 1e-9) / tonic), np.nan)


def swara_text(semi: int) -> str:
    """Semitones above Sa -> the app's swara notation (``Sa``, ``ni(k)``, ``Sa'``, ``.Dha``)."""
    name = SWARA_NAMES[semi % 12]
    octv = semi // 12
    return ("." * -octv + name) if octv < 0 else (name + "'" * octv)


def onsets(wav24: np.ndarray, start: float, end: float) -> np.ndarray:
    """Syllable onsets (s, absolute) in a passage of the lead vocal, from the spectral-flux onset detector."""
    import librosa
    a, b = int(start * SR), int(end * SR)
    x = np.asarray(wav24[a:b], np.float32)
    env = librosa.onset.onset_strength(y=x, sr=SR, hop_length=SR // 100)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=SR // 100, units="time", backtrack=False,
                                    delta=0.07, wait=6)
    return start + np.asarray(on, float)


def segment_notes(c: np.ndarray, scale: Sequence[int], on_frames: Sequence[int], min_frames: int = 3,
                  snap_cents: float = 45.0) -> List[Dict]:
    """Notes of a passage (frames relative to its start): each frame is snapped to the nearest allowed semitone
    (within ``snap_cents``; else it is 'moving'); a note is a run of the same snapped pitch of at least
    ``min_frames``, split again at every onset. Returns dicts with start/end frames, semitone, median cents,
    and the frames before it that were moving (a glide or a grace note)."""
    allowed = sorted({s + 12 * o for s in scale for o in range(-2, 4)})
    snapped = np.full(len(c), np.nan)
    for i, v in enumerate(c):
        if np.isfinite(v):
            s = min(allowed, key=lambda a: abs(a * 100 - v))
            if abs(s * 100 - v) <= snap_cents:
                snapped[i] = s
    cuts = set(int(f) for f in on_frames)
    notes, i = [], 0
    while i < len(c):
        if np.isnan(snapped[i]):
            i += 1
            continue
        j = i + 1
        while j < len(c) and snapped[j] == snapped[i] and j not in cuts:
            j += 1
        if j - i >= min_frames:
            notes.append({"a": i, "b": j, "semi": int(snapped[i]), "cents": float(np.nanmedian(c[i:j]))})
        i = j
    # merge a note into the previous one when they are the same pitch with only a tiny gap and no onset between
    merged: List[Dict] = []
    for n in notes:
        if merged and merged[-1]["semi"] == n["semi"] and n["a"] - merged[-1]["b"] <= 2 and n["a"] not in cuts:
            merged[-1]["b"] = n["b"]
        else:
            merged.append(n)
    return merged


def vibrato_of(c: np.ndarray) -> Optional[Dict]:
    """Rate (Hz) and depth (cents, half peak-to-peak) of a held note's wobble, or None if it has none."""
    v = c[np.isfinite(c)]
    if len(v) < 25:
        return None
    x = v - np.convolve(v, np.ones(15) / 15, mode="same")
    x = x[7:-7]
    if len(x) < 12:
        return None
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), 1 / FPS)
    band = (fr >= 3) & (fr <= 9)
    if not band.any():
        return None
    k = np.argmax(np.where(band, sp, 0))
    depth = float(np.sqrt(2) * np.std(x))
    if depth < 12:
        return None
    return {"type": "vibrato", "rate": round(float(fr[k]), 1), "depth": int(round(min(depth, 80))), "delay": 0.15,
            "fade": 0.2}


# -------------------------------------------------------------------------------------------- transcribe (CPU)
def transcribe(analysis: Path, start: float, end: float, tempo: float, raag: str = "", title: str = "",
               syls: Sequence[str] = (), tonic_override: Optional[float] = None, grid: float = 0.25,
               phrase_gap: float = 0.35, lyrics: str = "") -> Dict:
    """One passage -> a composition dict (see the module docstring). ``lyrics`` (a Devanagari line) is split with
    the dataset G2P and its syllables put on the notes that start at a syllable onset; the other notes hold the
    previous syllable (``-``). Without lyrics or ``syls`` every note sings its swara name."""
    from saregama.audio import load_audio
    from kalavinka.deploy.compose import raag_scale
    from kalavinka.deploy.sargam import DEV
    hz, tonic, info = load(analysis)
    tonic = tonic_override or tonic
    c_all = cents_of(hz, tonic)
    a, b = int(start * FPS), int(end * FPS)
    c = c_all[a:b]
    wav = load_audio(analysis / "lead.flac", SR)
    on = ((onsets(wav, start, end) - start) * FPS).round().astype(int)
    notes = segment_notes(c, raag_scale(raag), on)
    beat = 60.0 / tempo
    phrases, cur, prev_end = [], [], None
    syl_iter = iter(syls)
    if lyrics:                                        # a lyric line: syllables go on notes that start at an onset
        from saregama.g2p import line_to_syllables
        sylls = [s.text for s in line_to_syllables(lyrics, "hi")]
        on_set = np.asarray(on)
        starts = [n["a"] for n in notes]
        is_new = [k == 0 or (len(on_set) and np.min(np.abs(on_set - a)) <= 3) for k, a in enumerate(starts)]
        out, it = [], iter(sylls)
        for new in is_new:
            nxt = next(it, None) if new else None
            out.append(nxt if nxt is not None else "-")
        left = list(it)
        if left:                                       # more syllables than onsets: put the rest on the last notes
            for k in range(len(out) - 1, -1, -1):
                if not left:
                    break
                if out[k] == "-":
                    out[k] = left.pop()
        syl_iter = iter(out)
    for k, n in enumerate(notes):
        t0, t1 = n["a"] / FPS, n["b"] / FPS
        if prev_end is not None and t0 - prev_end > phrase_gap and cur:
            phrases.append(cur)
            cur = []
        elif prev_end is not None and t0 - prev_end > 0.12 and cur:
            cur.append({"syl": "_", "beats": max(grid, round((t0 - prev_end) / beat / grid) * grid)})
        nxt = notes[k + 1]["a"] / FPS if k + 1 < len(notes) else t1
        dur = (min(nxt, t1 + 0.12) - t0) if nxt - t1 < 0.12 else (t1 - t0)
        beats = max(grid, round(dur / beat / grid) * grid)
        name = swara_text(n["semi"])
        syl = next(syl_iter, None)
        note = {"syl": syl if syl is not None else DEV.get(name.strip(".'").split("(")[0].lower(), ""),
                "swara": name, "beats": beats}
        orn = []
        if k and notes[k - 1]["b"] < n["a"] - 3:                  # a glide or a touch before this note
            gap = c[notes[k - 1]["b"]:n["a"]]
            if np.isfinite(gap).sum() >= 3:
                orn.append({"type": "glide_in", "from": int(round(np.nanmean(gap[:2]) - n["semi"] * 100)),
                            "time": round(min(0.2, len(gap) / FPS), 2)})
        vib = vibrato_of(c[n["a"]:n["b"]]) if dur >= 0.45 else None
        if vib:
            orn.append(vib)
        if abs(n["cents"] - n["semi"] * 100) >= 15:
            note["cents"] = int(round(n["cents"] - n["semi"] * 100))
        if orn:
            note["orn"] = orn
        cur.append(note)
        prev_end = t1
    if cur:
        phrases.append(cur)
    return {"title": title or f"{Path(info['audio']).stem[:40]} {start:.1f}-{end:.1f} s", "tonic": "auto",
            "tempo_bpm": tempo, "raag": raag or "Chromatic (all 12)", "taal": "none",
            "source": {"audio": Path(info["audio"]).name, "start_s": start, "end_s": end,
                       "sung_tonic_hz": round(tonic, 2), "sung_tonic": hz_name(tonic)},
            "phrases": [{"notes": p} for p in phrases]}


def beat_grid(mix_path: Path, start: float, end: float, sub: int = 4) -> Tuple[float, np.ndarray]:
    """(tempo bpm, slot start times) for a passage: beats tracked on the original mix (librosa), each beat cut
    into ``sub`` equal slots."""
    import librosa
    import soundfile as sf
    info = sf.info(str(mix_path))
    a = max(0.0, start - 4.0)
    y, sr = sf.read(str(mix_path), start=int(a * info.samplerate), stop=int((end + 4.0) * info.samplerate))
    y = (y.mean(1) if y.ndim == 2 else y).astype(np.float32)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=100)
    bt = a + librosa.frames_to_time(beats, sr=sr)
    slots = []
    for x, y2 in zip(bt[:-1], bt[1:]):
        slots += list(x + (y2 - x) * np.arange(sub) / sub)
    slots = np.array([s for s in slots if start - 1e-6 <= s < end])
    return float(np.atleast_1d(tempo)[0]), slots


def transcribe_grid(analysis: Path, start: float, end: float, raag: str = "", title: str = "", sub: int = 4,
                    tonic_override: Optional[float] = None, fold_octave: bool = True, rest_beats: float = 1.0) -> Dict:
    """A rhythmic passage (sargam, taans) -> a composition on the song's own beat grid: one slot = 1/``sub`` beat;
    each slot's pitch is the median over its middle half, snapped to the raag; a slot with the same swara and no
    syllable onset continues the previous note (a held note), otherwise it starts a new one (syllable = the swara's
    name). With ``fold_octave``, passages sung an octave below by another singer are lifted so the median sits in
    the middle octave (one voice sings the whole composition). Rests of at least ``rest_beats`` split phrases."""
    from saregama.audio import load_audio
    from kalavinka.deploy.compose import raag_scale
    from kalavinka.deploy.sargam import DEV
    hz, tonic, info = load(analysis)
    tonic = tonic_override or tonic
    c = cents_of(hz, tonic)
    tempo, slots = beat_grid(Path(info["audio"]), start, end, sub)
    wav = load_audio(analysis / "lead.flac", SR)
    ons = onsets(wav, start, end)
    allowed = sorted({s + 12 * o for s in raag_scale(raag) for o in range(-2, 4)})
    step = 60.0 / tempo / sub
    cells: List[Optional[Tuple[int, bool]]] = []
    for s in slots:
        a, b = int((s + 0.25 * step) * FPS), max(int((s + 0.75 * step) * FPS), int((s + 0.25 * step) * FPS) + 1)
        w = c[a:b][np.isfinite(c[a:b])]
        if len(w) == 0:
            cells.append(None)
            continue
        v = float(np.median(w))
        semi = min(allowed, key=lambda x: abs(x * 100 - v))
        onset = bool(np.any(np.abs(ons - s) <= 0.45 * step))
        cells.append((semi, onset))
    if fold_octave:                                   # lift lines sung an octave lower (the other singer)
        voiced = [x[0] for x in cells if x]
        mid = float(np.median(voiced)) if voiced else 0.0
        win = 2 * sub                                 # judged per 2-beat window: singers trade lines without gaps
        for w0 in range(0, len(cells), win):
            idx = [j for j in range(w0, min(w0 + win, len(cells))) if cells[j] is not None]
            if not idx:
                continue
            shift = 12 * int(round((mid - np.median([cells[j][0] for j in idx])) / 12))
            if shift:
                for j in idx:
                    cells[j] = (cells[j][0] + shift, cells[j][1])
    q = 1.0 / sub
    phrases, cur, rest = [], [], 0
    for x in cells:
        if x is None:
            rest += 1
            continue
        if rest:
            if rest * q >= rest_beats and cur:
                phrases.append(cur)
                cur = []
            elif cur:
                cur.append({"syl": "_", "beats": rest * q})
            rest = 0
        semi, onset = x
        name = swara_text(semi)
        if cur and cur[-1].get("swara") == name and not onset:
            cur[-1]["beats"] += q                     # held over the slot
        else:
            cur.append({"syl": DEV.get(name.strip(".'").split("(")[0].lower(), ""), "swara": name, "beats": q})
    if cur:
        phrases.append(cur)
    for ph in phrases:                                # trailing rests are not needed inside a phrase
        while ph and ph[-1].get("syl") == "_":
            ph.pop()
    return {"title": title or f"{Path(info['audio']).stem[:40]} {start:.1f}-{end:.1f} s", "tonic": "auto",
            "tempo_bpm": round(tempo, 1), "raag": raag or "Chromatic (all 12)", "taal": "none",
            "source": {"audio": Path(info["audio"]).name, "start_s": start, "end_s": end,
                       "sung_tonic_hz": round(tonic, 2), "sung_tonic": hz_name(tonic), "grid": f"1/{sub} beat"},
            "phrases": [{"notes": p} for p in phrases if p]}


def syllable_starts(wav24: np.ndarray, voiced: np.ndarray, start: float, end: float, k: int,
                    min_gap: float = 0.14) -> List[float]:
    """``k`` syllable start times (s, absolute) in a sung line: the first voiced frame, plus the ``k - 1`` strongest
    onset-strength peaks at least ``min_gap`` apart (an approximation of forced alignment, good enough to place a
    lyric line whose syllables are then checked by ear)."""
    import librosa
    from scipy.signal import find_peaks
    a, b = int(start * SR), int(end * SR)
    env = librosa.onset.onset_strength(y=np.asarray(wav24[a:b], np.float32), sr=SR, hop_length=SR // 100)
    # the voice resuming after a pause of >= 0.3 s always starts a syllable
    must, prev_v = [], -10 ** 9
    for i in np.flatnonzero(voiced):
        if i - prev_v > 15:
            must.append(start + i / FPS)
        prev_v = i
    must = must[:k] or [start]
    peaks, _ = find_peaks(env, distance=max(1, int(min_gap * 100)))
    def sung_after(t: float) -> bool:
        """The voice is on within 60 ms after ``t`` (an onset in silence is a click, not a syllable)."""
        i = int(round((t - start) * FPS))
        return bool(voiced[max(i, 0):i + 4].any())
    cand = [(env[p], start + p / 100.0) for p in peaks
            if all(abs(start + p / 100.0 - m) > min_gap for m in must) and start + p / 100.0 > must[0]
            and sung_after(start + p / 100.0)]
    chosen = [t for _, t in sorted(cand, reverse=True)[: max(0, k - len(must))]]
    return sorted(must + chosen)


def transcribe_line(analysis: Path, start: float, end: float, lyrics: str, tonic_override: Optional[float] = None,
                    min_slur_s: float = 0.15, curve_step: int = 2) -> Dict:
    """One sung lyric line -> one phrase: a note per syllable (``sec`` = its sung length; ``swara`` = its most-held
    pitch), a slur note (``-``) where the syllable settles on another swara for at least ``min_slur_s``, rests where
    the voice stops, and the singer's real pitch curve as hand-drawn strokes (``f0_edits``, every ``curve_step``
    frames), so the model sings the actual ornaments. Returns the phrase dict."""
    from saregama.audio import load_audio
    from saregama.g2p import line_to_syllables
    hz, tonic, _ = load(analysis)
    tonic = tonic_override or tonic
    c_all = cents_of(hz, tonic)
    a0, b0 = int(start * FPS), int(end * FPS)
    c = c_all[a0:b0]
    voiced = np.isfinite(c)
    sylls = [s.text for s in line_to_syllables(lyrics, "hi")]
    wav = load_audio(analysis / "lead.flac", SR)
    st = syllable_starts(wav, voiced, start, end, len(sylls))
    bounds = [int(round((t - start) * FPS)) for t in st] + [int(np.flatnonzero(voiced).max()) + 1 if voiced.any() else len(c)]
    notes: List[Dict] = []
    t_prev = 0
    for k, syl in enumerate(sylls):
        x, y = bounds[k], max(bounds[k + 1], bounds[k] + 1)
        if x > t_prev:                                 # silence before this syllable (or leading gap)
            notes.append({"syl": "_", "sec": round((x - t_prev) / FPS, 3)})
        vz = np.flatnonzero(voiced[x:y])
        y_end = x + int(vz.max()) + 1 if len(vz) else y        # trailing silence of >= 0.2 s becomes a rest
        if y - y_end >= 10:
            y_sung = y_end
        else:
            y_sung = y
        seg = c[x:y_sung]
        parts = segment_notes(seg, list(range(12)), [], min_frames=max(2, int(min_slur_s * FPS)))
        if not parts:                                  # no stable pitch: one note on the median
            med = float(np.nanmedian(seg)) if np.isfinite(seg).any() else 0.0
            parts = [{"a": 0, "b": y_sung - x, "semi": int(round(med / 100))}]
        parts[0]["a"], parts[-1]["b"] = 0, y_sung - x  # the syllable owns its whole sung span
        for i in range(1, len(parts)):
            parts[i]["a"] = parts[i - 1]["b"] = max(parts[i - 1]["a"] + 1, parts[i]["a"])
        for i, p in enumerate(parts):
            notes.append({"syl": syl if i == 0 else "-", "swara": swara_text(p["semi"]),
                          "sec": round((p["b"] - p["a"]) / FPS, 3)})
        t_prev = y_sung
    strokes, cur = [], []
    for i in range(0, len(c), curve_step):
        if np.isfinite(c[i]):
            cur.append([round(i / FPS, 3), int(round(c[i]))])
        elif cur:
            if len(cur) > 1:
                strokes.append(cur)
            cur = []
    if len(cur) > 1:
        strokes.append(cur)
    return {"notes": notes, "f0_edits": strokes,
            "source": {"start_s": start, "end_s": end, "lyrics": lyrics, "sung_tonic_hz": round(tonic, 2)}}


# ----------------------------------------------------------------------------------------------- compare (CPU)
def designed_semis(comp: Dict) -> List[List[Tuple[Optional[int], float]]]:
    """Per phrase, per note: (semitones above Sa or None for a rest, seconds)."""
    from kalavinka.deploy.compose import normalise, note_seconds, parse_swara
    comp = normalise(comp)
    out = []
    for ph in comp["phrases"]:
        row = []
        for n in ph["notes"]:
            rest = n.get("syl") == "_" or str(n.get("swara", "")).lower() in ("rest", "_")
            row.append((None if rest else parse_swara(n["swara"]), note_seconds(n, comp["tempo_bpm"])))
        out.append(row)
    return out


def fit_phrase(c: np.ndarray, design: Sequence[Tuple[Optional[int], float]], t0: int,
               search: Tuple[int, int] = (-150, 150), stretch: Sequence[float] = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
               octave_free: bool = True) -> Dict:
    """Best placement of one designed phrase on the sung cents ``c`` (frames): start offset within ``search`` of
    ``t0``, a uniform tempo stretch, and an octave shift; score = mean |sung - designed| over voiced frames,
    capped at 300 c per frame. Returns the start frame, stretch, octave and per-note sung medians."""
    durs = np.array([d for _, d in design])
    semis = [s for s, _ in design]
    best = None
    for st in stretch:
        fr = np.round(np.concatenate([[0], np.cumsum(durs * st)]) * FPS).astype(int)
        target = np.full(fr[-1], np.nan)
        for (s, _), x, y in zip(design, fr[:-1], fr[1:]):
            if s is not None:
                target[x:y] = s * 100
        for off in range(search[0], search[1] + 1, 2):
            a = t0 + off
            if a < 0 or a + len(target) > len(c):
                continue
            seg = c[a:a + len(target)]
            for octv in ((-1200, 0, 1200) if octave_free else (0,)):
                m = np.isfinite(seg) & np.isfinite(target)
                if m.sum() < 0.3 * np.isfinite(target).sum():
                    continue
                err = float(np.mean(np.minimum(np.abs(seg[m] - target[m] - octv), 300)))
                if best is None or err < best["err"]:
                    best = {"err": err, "start": a, "stretch": st, "octave": octv // 1200, "bounds": fr}
    if best is None:
        return {"err": None}
    seg_notes = []
    for (s, _), x, y in zip(design, best["bounds"][:-1], best["bounds"][1:]):
        w = c[best["start"] + x: best["start"] + y] - best["octave"] * 1200
        k0, k1 = int(len(w) * 0.25), max(int(len(w) * 0.75), int(len(w) * 0.25) + 1)
        core = w[k0:k1][np.isfinite(w[k0:k1])]
        seg_notes.append({"designed": s, "sung_cents": float(np.median(core)) if len(core) else None})
    best["notes"] = seg_notes
    return best


def compare(analysis: Path, comp: Dict, start: float, starts: Optional[Sequence[float]] = None,
            tonic_override: Optional[float] = None, max_gap_s: float = 4.0) -> Dict:
    """Align every phrase of ``comp`` to the song and compare swaras note by note (see the module docstring)."""
    hz, tonic, _ = load(analysis)
    tonic = tonic_override or tonic
    c = cents_of(hz, tonic)
    design = designed_semis(comp)
    rows, t = [], int(start * FPS)
    for k, ph in enumerate(design):
        if starts and k < len(starts):
            fit = fit_phrase(c, ph, int(starts[k] * FPS), search=(-25, 25))
        else:
            fit = fit_phrase(c, ph, t, search=(-50, int(max_gap_s * FPS)))
        if fit.get("err") is None:
            rows.append({"phrase": k + 1, "found": False})
            continue
        dur = sum(d for _, d in ph) * fit["stretch"]
        t = fit["start"] + int(round(dur * FPS))
        notes = []
        for i, (n, (s, _)) in enumerate(zip(fit["notes"], ph)):
            if s is None or n["sung_cents"] is None:
                notes.append({"i": i + 1, "designed": None if s is None else swara_text(s), "sung": None})
                continue
            sung = int(round(n["sung_cents"] / 100))
            notes.append({"i": i + 1, "designed": swara_text(s), "sung": swara_text(sung),
                          "off_cents": int(round(n["sung_cents"] - s * 100)), "ok": sung == s})
        rows.append({"phrase": k + 1, "found": True, "start_s": round(fit["start"] / FPS, 2),
                     "tempo_factor": fit["stretch"], "octave_shift": fit["octave"],
                     "mean_err_cents": round(fit["err"], 1), "notes": notes,
                     "wrong": sum(1 for n in notes if n.get("ok") is False)})
    return {"tonic_hz": round(tonic, 2), "tonic": hz_name(tonic), "phrases": rows}


def apply_fixes(comp: Dict, report: Dict, min_wrong_share: float = 0.0) -> Dict:
    """The composition with every note whose sung swara differs replaced by the sung one, tempo scaled by the
    median fitted stretch, and the octave moved when a whole phrase was sung an octave away."""
    import copy
    out = copy.deepcopy(comp)
    stretches = [r["tempo_factor"] for r in report["phrases"] if r.get("found")]
    if stretches:
        out["tempo_bpm"] = round(float(out.get("tempo_bpm", 80)) / float(np.median(stretches)), 1)
    log = []
    for r, ph in zip(report["phrases"], out["phrases"]):
        if not r.get("found"):
            continue
        for n, note in zip(r["notes"], ph["notes"]):
            if n.get("ok") is False:
                log.append(f"phrase {r['phrase']} note {n['i']}: {n['designed']} -> {n['sung']} ({n['off_cents']:+d} c)")
                note["swara"] = n["sung"]
                if note.get("syl") not in ("-", "_", None, ""):
                    from kalavinka.deploy.sargam import DEV
                    base = n["sung"].strip(".'").split("(")[0].lower()
                    if note["syl"] in DEV.values():               # it sings the swara name: rename it too
                        note["syl"] = DEV.get(base, note["syl"])
    out.setdefault("notes_for_editor", [])
    out["notes_for_editor"] = log
    return out


# --------------------------------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyse")
    pa.add_argument("--audio", type=Path, required=True)
    pa.add_argument("--out", type=Path, required=True)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--asr", action="store_true")
    pt = sub.add_parser("transcribe")
    pt.add_argument("--analysis", type=Path, required=True)
    pt.add_argument("--start", type=float, required=True)
    pt.add_argument("--end", type=float, required=True)
    pt.add_argument("--tempo", type=float, default=0.0, help="bpm (required without --grid)")
    pt.add_argument("--raag", default="")
    pt.add_argument("--title", default="")
    pt.add_argument("--syl", default="", help="one lyric syllable per note, space-separated")
    pt.add_argument("--lyrics", default="", help="a Devanagari lyric line, put on the notes that start at onsets")
    pt.add_argument("--grid", action="store_true",
                    help="rhythmic passage (sargam, taan): read the pitch on the song's own beat grid (tempo from the mix)")
    pt.add_argument("--tonic", type=float, default=None, help="Sa in Hz (default: the estimated tonic)")
    pt.add_argument("--out", type=Path, required=True)
    pc = sub.add_parser("compare")
    pc.add_argument("--analysis", type=Path, required=True)
    pc.add_argument("--comp", type=Path, required=True)
    pc.add_argument("--start", type=float, required=True, help="about where the first phrase starts (s)")
    pc.add_argument("--starts", default="", help="comma-separated start (s) of every phrase, if known")
    pc.add_argument("--tonic", type=float, default=None)
    pc.add_argument("--fixed", type=Path, default=None)
    pc.add_argument("--report", type=Path, default=None)
    a = p.parse_args(argv)
    if a.cmd == "analyse":
        print(json.dumps(analyse(a.audio, a.out, a.device, a.asr), ensure_ascii=False, indent=1))
    elif a.cmd == "transcribe":
        if a.grid:
            comp = transcribe_grid(a.analysis, a.start, a.end, a.raag, a.title, tonic_override=a.tonic)
        else:
            comp = transcribe(a.analysis, a.start, a.end, a.tempo, a.raag, a.title, a.syl.split(), a.tonic,
                              lyrics=a.lyrics)
        a.out.write_text(json.dumps(comp, ensure_ascii=False, indent=1))
        print(f"{sum(len(p['notes']) for p in comp['phrases'])} notes in {len(comp['phrases'])} phrases -> {a.out}")
    else:
        comp = json.loads(a.comp.read_text())
        rep = compare(a.analysis, comp, a.start, [float(x) for x in a.starts.split(",") if x], a.tonic)
        print(f"tonic {rep['tonic_hz']} Hz ({rep['tonic']})")
        for r in rep["phrases"]:
            if not r.get("found"):
                print(f"phrase {r['phrase']}: not found")
                continue
            bad = [f"{n['i']}:{n['designed']}->{n['sung']}" for n in r["notes"] if n.get("ok") is False]
            print(f"phrase {r['phrase']}: at {r['start_s']} s, tempo x{r['tempo_factor']}, octave {r['octave_shift']:+d}, "
                  f"mean err {r['mean_err_cents']} c, wrong {r['wrong']}/{len(r['notes'])}  {' '.join(bad)}")
        if a.report:
            a.report.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        if a.fixed:
            a.fixed.write_text(json.dumps(apply_fixes(comp, rep), ensure_ascii=False, indent=1))
            print("fixed ->", a.fixed)


if __name__ == "__main__":
    main()

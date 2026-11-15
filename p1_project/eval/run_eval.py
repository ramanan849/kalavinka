"""Benchmark: ASR models x timing methods on Hindi / Hinglish film songs, scored
against LRCLIB synced lyrics.

For every song in songs.csv (with a matching reference in REFS):
  1. the full pipeline (separation, F0, tonic, Qwen3-ASR lyrics, MMS_FA
     phrase-window alignment, notes, QC) -> work/songs/<id>/, plus the
     pitch-contour sargam PDF;
  2. every ASR system on the separated vocal (and Qwen3 on the mix):
     loose-phonetic WER / CER against the reference text;
  3. timing of each transcript three ways - ASR alone (Whisper word timestamps;
     for Qwen3 the phrase start, words spread over the phrase), MMS_FA global,
     MMS_FA phrase windows - plus an oracle: the *reference* text aligned with
     MMS_FA, which isolates the aligner from ASR errors;
  4. results.json + summary.md in the output folder.

Usage (GPU node):  python run_eval.py --songs songs.csv --work work --out results
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from saregama.audio import FPS, SR, load_audio
from saregama.config import AsrCfg, load_config
from saregama.g2p import split_words
from saregama.lyrics_eval import error_rates, line_onset_errors, parse_lrc
from saregama.pipeline import Pipeline, g2p_opts, read_songs, safe_syllables
from saregama.schema import SongAnnotation
from saregama.stages.asr import Qwen3Asr, WhisperAsr, transcribe_song
from saregama.viz import song_contour_pdf

HERE = Path(__file__).resolve().parent
REFS = {  # song_id -> LRCLIB record (same release length as our file)
    "hi_film_2013_tum_hi_ho": "reference/lrclib_30373668.json",
    "hi_film_2022_kesariya": "reference/lrclib_32496065.json",
    # the dataset file labelled "Dilliwaali Girlfriend" actually holds "Subhanallah"
    # (same film); ASR revealed it, and the LRC of Subhanallah matches its length
    "hi_film_2013_dilliwaali_girlfriend": "reference/lrclib_25992830.json",
    "hi_film_2014_abhi_toh_party_shuru_hui_hai": "reference/lrclib_16827692.json",  # Hinglish
}
# references that do not match our recording's edit (scored for the record, but
# left out of averages): Kesariya's LRC is the 4:19 album song, our file a 2:52
# video edit with film dialogue and a verse / chorus cut
MISMATCHED = {"hi_film_2022_kesariya"}
SYSTEMS = [  # name, backend, model, language, input, word timestamps
    ("qwen3-asr-1.7b", "qwen3", "Qwen/Qwen3-ASR-1.7B", "hi", "vocals", False),
    ("qwen3-asr-1.7b (mix)", "qwen3", "Qwen/Qwen3-ASR-1.7B", "hi", "mix", False),
    ("whisper-large-v3", "whisper", "openai/whisper-large-v3", "hi", "vocals", True),
    ("whisper-hinglish-prime", "whisper", "Oriserve/Whisper-Hindi2Hinglish-Prime", "en", "vocals", True),
]


def timing(lrc, words):
    """Line-onset errors with an offset fit and with an offset + drift fit."""
    return {"offset": line_onset_errors(lrc, words, fit="offset"),
            "linear": line_onset_errors(lrc, words, fit="linear")}


def word_onsets(aligned, sylls, lines):
    """(word, onset s) in sung order from aligned syllables."""
    out, seen = [], set()
    for a in aligned:
        if a.kind != "lyric":
            continue
        s = sylls[a.line][a.index]
        key = (a.line, s.word_index)
        if key in seen:
            continue
        seen.add(key)
        out.append((split_words(lines[a.line])[s.word_index], a.start / FPS))
    return out


def phrase_spread(phrases):
    """ASR-alone timing without word times: words spread evenly over each phrase."""
    out = []
    for p in phrases:
        ws = p.text.split()
        for k, w in enumerate(ws):
            out.append((w, p.start + (p.end - p.start) * k / max(len(ws), 1)))
    return out


def align_lines(pipe, lines, windows, ann, emission, lang):
    """MMS_FA alignment of ``lines`` (windows=None: global) -> (word onsets, info)."""
    sylls, kept, win, _ = safe_syllables(lines, windows, lang, pipe.lexicon, **g2p_opts(pipe.cfg))
    aligned, info = pipe.aligner.align(None, sylls, ann.f0, ann.energy_db, emission=emission,
                                       windows=win, window_pad_s=pipe.cfg.asr.pad_s)
    return word_onsets(aligned, sylls, kept), info


def main():
    """Run the benchmark (see the module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", default=str(HERE / "songs.csv"))
    ap.add_argument("--work", default=str(HERE / "work"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--systems", default="", help="comma-separated system names to run (default all)")
    ap.add_argument("--reuse", action="store_true", help="keep existing pipeline annotations")
    ap.add_argument("--sep-model", default="vocals", help="karaoke (lead only) | vocals (all voices)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config, work_dir=a.work)
    cfg.asr.backend, cfg.asr.model, cfg.asr.align_mode = "qwen3", "Qwen/Qwen3-ASR-1.7B", "phrase"
    cfg.f0.backends = ["rmvpe", "fcpe", "yin"]
    cfg.separation.sep_model = a.sep_model
    pipe = Pipeline(cfg)
    rows = [r for r in read_songs(a.songs) if r["song_id"] in REFS]
    old = out / "results.json"
    results = json.loads(old.read_text(encoding="utf-8")) if a.reuse and old.exists() else {}

    # 1. full pipeline (Qwen3 lyrics) ------------------------------------
    for row in rows:
        t0 = time.time()
        ann = pipe.run_song(row, force=not a.reuse)
        sid = row["song_id"]
        prev = results.get(sid, {})
        results[sid] = {"pipeline": {"status": ann.status, "flags": ann.qc.get("flags"),
                                     "lyrics": ann.qc.get("lyrics"), "tonic_hz": ann.tonic_hz,
                                     "syllables": len(ann.syllables), "seconds": round(time.time() - t0, 1),
                                     "metrics": ann.qc.get("metrics")},
                        "asr": prev.get("asr", {}), "timing": prev.get("timing", {})}
        song_contour_pdf(pipe.song_dir(sid), out / f"{sid}_pitch_contour_sargam.pdf", title=sid)
        print(sid, "pipeline", ann.status, f"{time.time() - t0:.0f}s", flush=True)
    pipe._asr = pipe._sep = pipe._f0 = None          # free GPU memory for the ASR comparison
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    # cached per-song inputs
    data = {}
    for row in rows:
        sid = row["song_id"]
        d = pipe.song_dir(sid)
        ann = SongAnnotation.load(d)
        ref = json.loads((HERE / REFS[sid]).read_text(encoding="utf-8"))
        lrc = parse_lrc(ref["syncedLyrics"])
        vocals = load_audio(d / "vocals.wav", SR)
        data[sid] = dict(row=row, ann=ann, lrc=lrc, ref_text=" ".join(t for _, t in lrc),
                         vocals=vocals, emission=pipe.aligner.emitter(vocals, ann.n_frames))
        # oracle: reference lyrics + MMS_FA global
        ow, oinfo = align_lines(pipe, [t for _, t in lrc], None, ann, data[sid]["emission"], "hi")
        results[sid]["timing"]["reference text + MMS_FA (oracle)"] = timing(lrc, ow)

    # 2-3. ASR systems ----------------------------------------------------
    only = set(a.systems.split(",")) if a.systems else None
    for name, backend, model, lang, inp, words in SYSTEMS:
        if only and name not in only:
            continue
        try:
            run_system(results, data, pipe, cfg, name, backend, model, lang, inp, words)
        except Exception as e:                      # keep the other systems' results
            import traceback
            traceback.print_exc()
            for sid in data:
                results[sid]["asr"].setdefault(name, {"error": repr(e)})
        save(results, out)

    save(results, out)
    print((out / "summary.md").read_text(encoding="utf-8"))


def save(results, out):
    """Write results.json and summary.md (called after every system)."""
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_summary(results, out / "summary.md")


def run_system(results, data, pipe, cfg, name, backend, model, lang, inp, words):
    """Transcribe every song with one ASR system and score words and timing."""
    acfg = AsrCfg(backend=backend, model=model, language=lang, input=inp, device=cfg.asr.device)
    t0 = time.time()
    asr = Qwen3Asr(acfg) if backend == "qwen3" else WhisperAsr(acfg)
    words = words and asr.has_word_times              # Hinglish fine-tunes have no alignment heads
    print(name, "loaded", f"{time.time() - t0:.0f}s", "(word times)" if words else "", flush=True)
    for sid, D in data.items():
        ann, lrc = D["ann"], D["lrc"]
        audio = D["vocals"] if inp == "vocals" else load_audio(D["row"]["audio"], SR)
        t0 = time.time()
        phrases = transcribe_song(audio, SR, ann.f0 > 0, FPS, asr, acfg, words=words)
        secs = time.time() - t0
        kept = [p for p in phrases if p.text]
        hyp = " ".join(p.text for p in kept)
        er = error_rates(D["ref_text"], hyp)
        results[sid]["asr"][name] = {**er, "seconds": round(secs, 1), "phrases": len(phrases),
                                     "kept": len(kept), "text": [[p.start, p.end, p.text] for p in kept],
                                     "dropped": [[p.start, p.end, p.raw, p.dropped] for p in phrases if p.dropped]}
        print(sid, name, f"WER {er['wer']:.3f} CER {er['cer']:.3f} {secs:.0f}s", flush=True)
        if inp != "vocals":
            continue
        tm = results[sid]["timing"]
        if words:
            wt = [(w["text"], w["start"]) for p in kept for w in p.words if w["text"].strip()]
            tm[f"{name}: ASR word timestamps"] = timing(lrc, wt)
        else:
            tm[f"{name}: ASR phrase times"] = timing(lrc, phrase_spread(kept))
        lines = [p.text for p in kept]
        gw, _ = align_lines(pipe, lines, None, ann, D["emission"], "hi")
        tm[f"{name} + MMS_FA global"] = timing(lrc, gw)
        pw, _ = align_lines(pipe, lines, [(p.start, p.end) for p in kept], ann, D["emission"], "hi")
        tm[f"{name} + MMS_FA phrase"] = timing(lrc, pw)
        if len(gw) == len(pw):                       # how different the two alignments are
            d = np.abs(np.array([t for _, t in gw]) - np.array([t for _, t in pw]))
            tm[f"{name} + MMS_FA phrase"]["vs_global"] = {
                "words": len(d), "mean_abs_s": round(float(d.mean()), 3),
                "share_moved_over_0.2s": round(float((d > 0.2).mean()), 3)}
    del asr
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def write_summary(results, path):
    """Markdown tables: WER / CER per system and song; line-onset timing per method
    (offset fit and offset + drift fit).  Songs in MISMATCHED are shown but left
    out of the mean."""
    sids = list(results)
    head = [s + (" (ref. mismatch)" if s in MISMATCHED else "") for s in sids]
    good = [s for s in sids if s not in MISMATCHED]
    L = ["## Lyrics transcription (loose-phonetic WER / CER vs LRCLIB)", "",
         "| system | " + " | ".join(head) + " | mean WER (matched refs) |", "|---" * (len(sids) + 2) + "|"]
    names = [x[0] for x in SYSTEMS if any(x[0] in results[s]["asr"] for s in sids)]
    for n in names:
        if any("wer" not in results[s]["asr"].get(n, {}) for s in sids):
            L.append(f"| {n} | failed (see log) |")
            continue
        cells = [f"{results[s]['asr'][n]['wer']:.2f} / {results[s]['asr'][n]['cer']:.2f}" for s in sids]
        mean = np.mean([results[s]["asr"][n]["wer"] for s in good])
        L.append(f"| {n} | " + " | ".join(cells) + f" | **{mean:.2f}** |")
    L += ["", "## Line-onset timing vs LRCLIB", "",
          "Median |error| after removing an offset / after removing offset + tempo drift; "
          "share of lines within 0.5 s (drift fit); lines scored.", "",
          "| method | " + " | ".join(head) + " |", "|---" * (len(sids) + 1) + "|"]
    methods = []
    for s in sids:
        for m in results[s]["timing"]:
            if m not in methods:
                methods.append(m)
    for m in methods:
        cells = []
        for s in sids:
            r = results[s]["timing"].get(m)
            if not r or not r["offset"].get("scored"):
                cells.append("-")
                continue
            o, l = r["offset"], r["linear"]
            cells.append(f"{o['median_abs_s']:.2f} / {l.get('median_abs_s', o['median_abs_s']):.2f} s; "
                         f"{100 * l.get('within_0.5s', o['within_0.5s']):.0f}%; {o['scored']}/{o['lines']}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

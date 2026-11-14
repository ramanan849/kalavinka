"""Runs the stages for each song in songs.csv and writes one SongAnnotation per song.

songs.csv columns (header required; only the first four are mandatory):

    song_id     unique, filesystem-safe id, e.g. mr_bhavgeet_0007
    audio       path to the mp3/wav/flac
    language    hi | mr | ta
    lyrics      path to the lyric sheet (.txt); empty -> transcribed by the [asr]
                backend (stages.asr) when one is set, else the song stops there
    style       film | bhajan | hindustani | carnatic | bhavgeet | natyasangeet | ...
    singer_id   stable id per singer (needed for singer-disjoint splits)
    gender      f | m | other
    vocal_only  1 if the file is already a dry solo vocal (skip separation)
    tonic_hz    known Sa in Hz (any octave), if available
    source      where the recording came from
    license     rights note
"""
from __future__ import annotations

import csv
import json
from collections import Counter
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio import FPS, SR, load_audio, n_frames, save_audio
from .config import Config
from .g2p import Lexicon, line_to_syllables, split_words
from .schema import SongAnnotation, SyllableAnn
from .stages.align import Aligner
from .stages.f0 import F0Estimator
from .stages.lyrics import read_lyrics
from .stages.notes import build_notes
from .stages.qc import song_qc
from .stages.quality import quality_curves
from .stages.separate import Separator, leakage_score
from .stages.tonic import estimate_tonic, snap_to_tonic_octave

REQUIRED = ("song_id", "audio", "language")


def read_songs(csv_path: str | Path) -> List[Dict[str, str]]:
    """Rows of songs.csv as dicts (whitespace stripped); relative ``audio`` and
    ``lyrics`` paths are resolved against the CSV's folder.  Raises on a missing
    required column value or a duplicate ``song_id``.
    """
    base = Path(csv_path).parent
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            for k in REQUIRED:
                if not row.get(k):
                    raise ValueError(f"songs.csv row {row} is missing {k!r}")
            for k in ("audio", "lyrics"):
                if row.get(k) and not Path(row[k]).is_absolute():
                    row[k] = str((base / row[k]).resolve())
            rows.append(row)
    ids = [r["song_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("song_id values in songs.csv must be unique")
    return rows


def g2p_opts(cfg: Config) -> Dict:
    """Keyword options for the G2P functions, from the ``[g2p]`` config section."""
    g = cfg.g2p
    return {"keep_final_schwa": g.keep_final_schwa, "delete_medial_schwa": g.delete_medial_schwa,
            "marathi_affricates": g.marathi_affricates, "initial_c_as_s": g.tamil_initial_c_as_s,
            "intervocalic_k": g.tamil_intervocalic_k}


def safe_syllables(lines: List[str], windows: Optional[List[Tuple[float, float]]], lang: str,
                   lexicon: Lexicon, **opts):
    """``lyrics_to_syllables`` that drops words the G2P cannot read instead of
    failing (ASR output can contain stray symbols).  Lines left without
    syllables are dropped together with their window.

    Returns (syllables per line, kept lines, kept windows or None, dropped words).
    """
    out, kept, kwin, bad = [], [], [] if windows is not None else None, []
    for k, line in enumerate(lines):
        words = []
        for w in split_words(line):
            try:
                line_to_syllables(w, lang, lexicon, **opts)
                words.append(w)
            except (ValueError, KeyError):
                bad.append(w)
        if not words:
            continue
        text = " ".join(words)
        out.append(line_to_syllables(text, lang, lexicon, **opts))
        kept.append(text)
        if kwin is not None:
            kwin.append(windows[k])
    return out, kept, kwin, bad


class Pipeline:
    """Legacy film-song pipeline: separation, F0, tonic, alignment, notes,
    quality curves and QC for each song of songs.csv, written as a
    ``SongAnnotation`` under ``<work_dir>/songs/<song_id>/``.
    """
    def __init__(self, cfg: Config, annotator=None):
        """Keep the config and lexicon; models are built lazily on first use."""
        self.cfg = cfg
        self.work = Path(cfg.work_dir)
        self.annotator = annotator
        self._sep: Optional[Separator] = None
        self._f0: Optional[F0Estimator] = None
        self._aligner: Optional[Aligner] = None
        self._asr = None
        self.lexicon = Lexicon.load(*cfg.g2p.lexicon)

    # lazily built so that `--stages` subsets don't load unneeded models
    @property
    def sep(self):
        """The vocal separator (built on first access)."""
        if self._sep is None:
            self._sep = Separator(self.cfg.separation)
        return self._sep

    @property
    def f0est(self):
        """The F0 ensemble (built on first access)."""
        if self._f0 is None:
            self._f0 = F0Estimator(self.cfg.f0, self.cfg.separation.soulx_repo, self.annotator)
        return self._f0

    @property
    def aligner(self):
        """The CTC aligner (built on first access)."""
        if self._aligner is None:
            self._aligner = Aligner(self.cfg.align, self.annotator)
        return self._aligner

    @property
    def asr(self):
        """The ASR backend of ``[asr]`` (built on first access)."""
        if self._asr is None:
            from .stages.asr import load_backend
            self._asr = load_backend(self.cfg.asr)
        return self._asr

    def song_dir(self, song_id: str) -> Path:
        """Folder of one song's annotation."""
        return self.work / "songs" / song_id

    def transcribe(self, row: Dict[str, str], d: Path, vocals: np.ndarray, f0: np.ndarray,
                   force: bool = False):
        """ASR transcript of one song: reuse ``asr.json`` unless ``force`` (or the
        model changed), else transcribe the vocal (or the mix, ``[asr] input``).
        Returns (phrases, model id).
        """
        from .stages.asr import load_transcript, transcribe_song, write_transcript
        a = self.cfg.asr
        old = load_transcript(d)
        if old is not None and not force and (not a.model or old[0] == a.model):
            return old[1], old[0]
        audio = vocals if a.input == "vocals" else load_audio(row["audio"], SR)
        phrases = transcribe_song(audio, SR, f0 > 0, FPS, self.asr, a)
        model = getattr(self.asr, "model_id", a.backend)
        write_transcript(phrases, d, model)
        return phrases, model

    # ------------------------------------------------------------------
    def run_song(self, row: Dict[str, str], force: bool = False,
                 emission: Optional[np.ndarray] = None) -> SongAnnotation:
        """Annotate one song and return its annotation.

        Reviewed and gold songs are never overwritten; other existing annotations
        are returned unless ``force``.  ``emission`` injects precomputed CTC
        log-probabilities (tests).
        """
        cfg = self.cfg
        sid = row["song_id"]
        d = self.song_dir(sid)
        d.mkdir(parents=True, exist_ok=True)
        if (d / "annotation.json").exists():
            existing = SongAnnotation.load(d)
            if existing.status in ("reviewed", "gold") or not force:
                return existing                 # never overwrite human work
        lang = row["language"]

        # 1. separation ------------------------------------------------
        vpath = d / "vocals.wav"
        if vpath.exists() and not force:
            vocals = load_audio(vpath, SR)
            accomp = load_audio(d / "accomp.wav", SR) if (d / "accomp.wav").exists() else None
        elif row.get("vocal_only", "") in ("1", "true", "yes"):
            from .audio import loudness_normalize
            vocals = loudness_normalize(load_audio(row["audio"], SR))
            save_audio(vpath, vocals)
            accomp = None
        else:
            vocals, accomp = self.sep.process(row["audio"], str(vpath), str(d / "accomp.wav"))
        n = n_frames(len(vocals))

        # 2. F0 ----------------------------------------------------------
        f0info = self.f0est(vocals)
        f0, agree = f0info["f0"][:n], f0info["f0_agree"][:n]

        # 3. voice-quality curves ---------------------------------------
        qual = quality_curves(vocals, f0)

        # 4. tonic -------------------------------------------------------
        if row.get("tonic_hz"):
            tonic, tconf, tsrc = snap_to_tonic_octave(float(row["tonic_hz"]), f0), 1.0, "given"
        else:
            tonic, tconf = estimate_tonic(f0, cfg.tonic.pa_weight, cfg.tonic.ma_weight)
            tsrc = "auto"

        ann = SongAnnotation(
            song_id=sid, language=lang, style=row.get("style", "other") or "other",
            singer_id=row.get("singer_id", ""), gender=row.get("gender", ""),
            source=row.get("source", ""), license=row.get("license", ""),
            audio_path=row["audio"], vocals_path=str(vpath), n_frames=n,
            tonic_hz=round(tonic, 3), tonic_source=tsrc, tonic_conf=round(tconf, 3),
            f0=f0, f0_agree=agree, energy_db=qual["energy_db"], hnr_db=qual["hnr_db"],
            h1h2_db=qual["h1h2_db"])

        # 5. lyrics -> syllables -----------------------------------------
        windows = None
        lyrics_info: Dict = {"source": "sheet", "path": row.get("lyrics", "")}
        if row.get("lyrics"):
            lines = read_lyrics(row["lyrics"])
        elif cfg.asr.backend != "none":
            # 5a. no lyric sheet: transcribe the sung phrases (stages.asr)
            phrases, model = self.transcribe(row, d, vocals, f0, force=force)
            kept = [p for p in phrases if p.text]
            lines = [p.text for p in kept]
            if cfg.asr.align_mode == "phrase":
                windows = [(p.start, p.end) for p in kept]
            lyrics_info = {"source": "asr", "model": model, "align_mode": cfg.asr.align_mode,
                           "phrases": len(phrases), "kept": len(kept),
                           "dropped": dict(Counter(p.dropped for p in phrases if p.dropped))}
        else:
            ann.status = "rejected"
            ann.qc = {"flags": ["no_lyrics"], "metrics": {},
                      "note": "set [asr] backend, or run `saregama draft-lyrics` and correct the draft"}
            ann.save(d)
            return ann
        sylls, lines, windows, bad_words = safe_syllables(lines, windows, lang, self.lexicon,
                                                         **g2p_opts(cfg))
        ann.lines = lines
        if bad_words:
            lyrics_info["g2p_dropped_words"] = bad_words
        if not lines:
            ann.status = "rejected"
            ann.qc = {"flags": ["no_lyrics"], "metrics": {}, "lyrics": lyrics_info,
                      "note": "ASR found no usable lyrics (see asr.json)"}
            ann.save(d)
            return ann

        # 6. alignment ---------------------------------------------------
        aligned, ainfo = self.aligner.align(vocals, sylls, f0, qual["energy_db"], emission=emission,
                                            windows=windows, window_pad_s=cfg.asr.pad_s)
        ann.syllables = []
        for a in aligned:
            if a.kind == "vocalise":
                txt, phones, flags = "<VOC>", ["aa"], ["vocalise"]
                if cfg.align.vocalise_policy == "akaar":
                    txt, flags = ("ஆ" if lang == "ta" else "आ"), ["vocalise", "akaar"]
                ann.syllables.append(SyllableAnn(txt, phones, a.start, a.end, a.line, -1, 0.0, flags))
            else:
                s = sylls[a.line][a.index]
                ann.syllables.append(SyllableAnn(s.text, list(s.phones), a.start, a.end, a.line,
                                                 s.word_index, round(float(a.conf), 3)
                                                 if np.isfinite(a.conf) else float("nan"), []))

        # 7. notes -------------------------------------------------------
        ann.notes = build_notes(ann.syllables, f0, ann.tonic_hz, n, cfg.notes)

        # 8. QC ----------------------------------------------------------
        leak = leakage_score(vocals, accomp, f0, qual["energy_db"]) if accomp is not None else None
        metrics, flags, status = song_qc(ann, cfg.qc, ainfo, f0info, leak)
        if lyrics_info["source"] == "asr":
            flags = list(flags) + ["lyrics_asr_unverified"]
            if not cfg.asr.accept_unreviewed and status == "auto":
                status = "flagged"            # ASR words are a draft until a person checks them
        ann.qc = {"metrics": metrics, "flags": flags, "lyrics": lyrics_info,
                  "align": {k: v for k, v in ainfo.items() if not isinstance(v, np.ndarray)}}
        ann.status = status
        errs = ann.validate()
        if errs:
            ann.qc["validation_errors"] = errs
            ann.status = "flagged"
        src = f":asr={lyrics_info['model']}" if lyrics_info["source"] == "asr" else ""
        ann.history.append(f"auto:{cfg.align.backend}:{'+'.join(cfg.f0.backends)}{src}")
        ann.save(d)
        return ann

    def run(self, songs: List[Dict[str, str]], force: bool = False) -> Dict[str, str]:
        """``run_song`` for every row; errors are recorded per song (``error.txt``)
        instead of stopping the batch.  Returns {song_id: status or error}.
        """
        results = {}
        for row in songs:
            try:
                ann = self.run_song(row, force=force)
                results[row["song_id"]] = ann.status
            except Exception as e:                         # keep going, record the error
                results[row["song_id"]] = f"error: {e}"
                d = self.song_dir(row["song_id"])
                d.mkdir(parents=True, exist_ok=True)
                (d / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (self.work / "run_status.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        return results

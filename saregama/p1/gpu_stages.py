"""GPU stages (PLAN.md §4, S3–S6 and S9's emission cache).

``separate``   full-vocals Mel-RoFormer + de-reverb on the mix -> ``vocals.flac``;
               the karaoke model on that vocal -> ``lead.flac``.  Both 24 kHz mono
               FLAC at the same gain, so ``vocals - lead`` is the other voices
               (duet partner, chorus, harmonies) with reverb out of the picture.
``f0``         RMVPE + FCPE + YIN ensemble at 50 fps on the vocal (``f0.npz``),
               plus overlap features (``poly.npz``): RMVPE's second salience peak
               away from the main pitch and its octaves, and the residual level.
``voices``     single-voice mask; ECAPA embedding per single-voice phrase;
               agglomerative clustering into singers within the song (``voices.json``).
``asr``        Qwen3-ASR and Whisper per sung phrase (``asr.json``)
               (singer / overlap per phrase is added later, in reconcile, so tuning
               the voices stage never re-runs ASR).
``emissions``  MMS_FA log-probabilities at 50 fps (``mms.npz``, float16): every later
               alignment decision is CPU-only and re-runnable from this cache.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..audio import FPS, SR, load_audio, resample
from .manifest import Song
from .state import atomic_flac, atomic_json, atomic_npz, read_json
from .worker import Skip, Stage

ASR_SR = 16000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _dup_check(ctx, song: Song) -> None:
    """Raise Skip for a duplicate upload (``dupes.json``)."""
    dup = (read_json(ctx.work / "dupes.json", {}) or {}).get("duplicate_of", {})
    if song.song_id in dup:
        raise Skip(f"duplicate_of:{dup[song.song_id]}")


def _mono(x: np.ndarray) -> np.ndarray:
    """Channel-mean of a (channels, n) or (n, channels) or (n,) array, float32."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0) if x.shape[0] <= 2 else x.mean(axis=1)
    return x


def _separators(ctx):
    """(full-vocals separator with de-reverb, karaoke separator) - SoulX wrappers."""
    from ..config import SeparationCfg
    from ..stages.separate import Separator
    dev = ctx.cfg["device"]["device"]

    def build():
        """Load both Mel-RoFormer separators."""
        full = Separator(SeparationCfg(sep_model="vocals", dereverb=True, device=dev))._load_soulx()
        lead = Separator(SeparationCfg(sep_model=ctx.cfg["separation"]["lead_model"], dereverb=False,
                                       device=dev))._load_soulx()
        return full, lead
    return ctx.model("separators", build)


def _load(sdir: Path, name: str) -> np.ndarray:
    """A 24 kHz mono stem of the song."""
    return load_audio(sdir / name, SR)


# ---------------------------------------------------------------------------
# S3 separate
# ---------------------------------------------------------------------------
def separate(ctx, song: Song, sdir: Path) -> Dict:
    """Full vocals (de-reverbed) and the karaoke lead, 24 kHz FLAC, same gain."""
    _dup_check(ctx, song)
    import torch
    full, lead = _separators(ctx)
    res = full.process(song.audio, verbose=False)
    sr = int(res.sample_rate)
    voc = res.vocals_dereverbed if res.vocals_dereverbed is not None else res.vocals
    voc = np.asarray(voc, dtype=np.float32)
    voc_st = voc if voc.ndim == 2 else np.stack([voc, voc])
    # the SoulX repo is on sys.path once the separators are loaded
    from preprocess.tools.vocal_separation.model import process as sep_process  # type: ignore
    lead_arr = sep_process(voc_st, lead.sep_model, lead.args, lead.sep_config,
                           torch.device(lead.device) if isinstance(lead.device, str) else lead.device)
    v24 = resample(_mono(voc), sr, SR)
    l24 = resample(_mono(lead_arr), sr, SR)[: len(v24)]
    if len(l24) < len(v24):
        l24 = np.pad(l24, (0, len(v24) - len(l24)))
    rms_v = float(np.sqrt(np.mean(v24 ** 2))) or 1e-9
    gain = 10 ** (-20 / 20) / rms_v                   # vocal at -20 dBFS RMS (as the legacy pipeline)
    peak = max(np.abs(v24).max() * gain, 1e-9)
    if peak > 0.99:                                   # keep both inside [-1, 1] at the same gain
        gain *= 0.99 / peak
    atomic_flac(sdir / "vocals.flac", v24 * gain, SR)
    atomic_flac(sdir / "lead.flac", l24 * gain, SR)
    rms = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)   # noqa: E731
    return {"vocal_db": round(float(rms(v24 * gain)), 1), "lead_db": round(float(rms(l24 * gain)), 1),
            "seconds_audio": round(len(v24) / SR, 1)}


# ---------------------------------------------------------------------------
# S4 f0 + overlap features
# ---------------------------------------------------------------------------
def _frame_db(x: np.ndarray, n: int) -> np.ndarray:
    """RMS level (dB) per 20 ms frame, ``n`` frames."""
    hop = SR // FPS
    x = np.pad(x, (0, max(0, n * hop - len(x))))[: n * hop]
    return 20 * np.log10(np.sqrt((x.reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)


def second_peak(sal: np.ndarray) -> np.ndarray:
    """Per frame: the highest salience at least 150 cents from the main peak and
    from its octaves (RMVPE bins are 20 cents) - a second simultaneous pitch."""
    n, b = sal.shape
    top = sal.argmax(axis=1)
    bins = np.arange(b)[None, :]
    d = np.abs(bins - top[:, None])
    far = (d >= 8) & (np.abs(d - 60) >= 8) & (np.abs(d - 120) >= 8)    # 60 bins = one octave
    return np.where(far, sal, 0).max(axis=1)


def f0(ctx, song: Song, sdir: Path) -> Dict:
    """F0 ensemble (50 fps) -> ``f0.npz``; overlap features -> ``poly.npz``."""
    _dup_check(ctx, song)
    from ..config import F0Cfg
    from ..stages.f0 import F0Estimator
    c = ctx.cfg["f0"]
    est = ctx.model("f0", lambda: F0Estimator(F0Cfg(backends=list(c["backends"]), fmin=c["fmin"],
                                                    fmax=c["fmax"], agree_cents=c["agree_cents"],
                                                    device=ctx.cfg["device"]["device"]), fps=FPS))
    voc, lead = _load(sdir, "vocals.flac"), _load(sdir, "lead.flac")
    out = est(voc, SR)
    hz, agree = out["f0"], out["f0_agree"]
    n = len(hz)
    sal = est.backends["rmvpe"].salience(resample(voc, SR, 16000))      # 100 fps
    sp = second_peak(sal)[::2]
    sp = np.pad(sp, (0, max(0, n - len(sp))))[:n]
    vdb, rdb = _frame_db(voc, n), _frame_db(voc - lead[: len(voc)], n)
    atomic_npz(sdir / "f0.npz", f0=hz.astype(np.float32), agree=np.asarray(agree, np.float32), fps=FPS,
               **{f"track_{k}": np.asarray(v, np.float32) for k, v in out["tracks"].items()})
    atomic_npz(sdir / "poly.npz", second_peak=sp.astype(np.float32), vocal_db=vdb.astype(np.float32),
               resid_db=rdb.astype(np.float32), fps=FPS)
    v = hz > 0
    return {"voiced": round(float(v.mean()), 3),
            "agree_voiced": round(float(np.mean(agree[v])) if v.any() else 0.0, 3)}


# ---------------------------------------------------------------------------
# S5 voices
# ---------------------------------------------------------------------------
def single_voice_mask(hz: np.ndarray, agree: np.ndarray, poly: Dict, cfg: Dict) -> np.ndarray:
    """Voiced frames with one voice: little residual beside the lead, no strong
    second pitch, and the F0 ensemble in agreement."""
    resid_rel = poly["resid_db"] - poly["vocal_db"]
    return (hz > 0) & (resid_rel < cfg["residual_db"]) & (poly["second_peak"] < cfg["second_peak"]) \
        & (agree >= 0.6)


def runs(mask: np.ndarray, min_len: int, max_gap: int = 0) -> List[Tuple[int, int]]:
    """[start, end) runs of ``mask`` (gaps up to ``max_gap`` bridged), at least ``min_len`` long."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    rs = [[a, b] for a, b in zip(idx[::2], idx[1::2])]
    merged: List[List[int]] = []
    for r in rs:
        if merged and r[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(int(a), int(b)) for a, b in merged if b - a >= min_len]


def cluster_embeddings(emb: np.ndarray, threshold: float, median_hz: np.ndarray | None = None,
                       register_weight: float = 0.5) -> np.ndarray:
    """Average-linkage agglomerative clustering; labels ordered by first appearance.

    Distance = cosine distance of the speaker embeddings + ``register_weight``
    per octave between the segments' median F0.  ECAPA is a speech model; on
    singing it can confuse a male and a female voice, the pitch register can't.
    """
    if len(emb) == 0:
        return np.zeros(0, dtype=int)
    if len(emb) == 1:
        return np.zeros(1, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    d = pdist(e, metric="cosine")
    if median_hz is not None:
        oct_ = np.log2(np.maximum(np.asarray(median_hz, float), 1.0))[:, None]
        d = d + register_weight * pdist(oct_, metric="cityblock")
    lab = fcluster(linkage(d, method="average"), t=threshold, criterion="distance")
    order = {l: k for k, l in enumerate(dict.fromkeys(lab))}
    return np.array([order[l] for l in lab])


def voices(ctx, song: Song, sdir: Path) -> Dict:
    """Single-voice segments with singer clusters -> ``voices.json`` (+ ``voices_emb.npy``)."""
    _dup_check(ctx, song)
    c = ctx.cfg["voices"]
    fz = np.load(sdir / "f0.npz")
    poly = dict(np.load(sdir / "poly.npz"))
    mask = single_voice_mask(fz["f0"], fz["agree"], poly, c)
    segs = runs(mask, int(c["min_single_s"] * FPS), max_gap=int(0.15 * FPS))

    def build():
        """ECAPA speaker encoder (SpeechBrain), from the offline HF cache."""
        from speechbrain.inference.speaker import EncoderClassifier
        return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                              savedir=str(ctx.work / "models" / "ecapa"),
                                              run_opts={"device": ctx.cfg["device"]["device"]})
    enc = ctx.model("ecapa", build)
    import torch
    voc16 = resample(_load(sdir, "vocals.flac"), SR, ASR_SR)
    embs = []
    with torch.inference_mode():
        for a, b in segs:
            w = torch.from_numpy(voc16[a * ASR_SR // FPS: b * ASR_SR // FPS]).float()[None]
            embs.append(enc.encode_batch(w.to(enc.device)).squeeze().float().cpu().numpy())
    emb = np.array(embs, dtype=np.float32).reshape(len(segs), -1) if embs else np.zeros((0, 192), np.float32)
    med = np.array([np.median(fz["f0"][a:b][fz["f0"][a:b] > 0]) if (fz["f0"][a:b] > 0).any() else 0.0
                    for a, b in segs])
    lab = cluster_embeddings(emb, c["cluster_threshold"], med)
    clusters = {}
    for k in sorted(set(lab.tolist())):
        sel = lab == k
        clusters[str(k)] = {"seconds": round(float(sum(b - a for (a, b), s in zip(segs, sel) if s)) / FPS, 1),
                            "segments": int(sel.sum())}
    np.save(sdir / "voices_emb.npy", emb)
    voiced = fz["f0"] > 0
    out = {"fps": FPS, "single_frames": int(mask.sum()), "voiced_frames": int(voiced.sum()),
           "segments": [{"start": a / FPS, "end": b / FPS, "cluster": int(l), "median_hz": round(float(m), 1)}
                        for (a, b), l, m in zip(segs, lab, med)],
           "clusters": clusters}
    atomic_json(sdir / "voices.json", out)
    np.save(sdir / "single_mask.npy", mask)
    return {"single_share": round(float(mask.sum() / max(voiced.sum(), 1)), 3), "clusters": len(clusters)}


# ---------------------------------------------------------------------------
# S6 asr (two systems)
# ---------------------------------------------------------------------------
def _asr_backends(ctx):
    """(Qwen3-ASR, Whisper) backends, loaded once per worker."""
    from ..config import AsrCfg
    from ..stages.asr import Qwen3Asr, WhisperAsr
    a, dev = ctx.cfg["asr"], ctx.cfg["device"]["device"]
    q = ctx.model("qwen3", lambda: Qwen3Asr(AsrCfg(backend="qwen3", model=a["primary"], language=a["language"],
                                                   batch_size=a["batch_size"], device=dev)))
    w = ctx.model("whisper", lambda: WhisperAsr(AsrCfg(backend="whisper", model=a["secondary"],
                                                       language=a["language"], batch_size=a["batch_size"],
                                                       beams=a.get("whisper_beams", 1), device=dev)))
    return q, w


def asr(ctx, song: Song, sdir: Path) -> Dict:
    """Phrases transcribed by both systems -> ``asr.json``."""
    _dup_check(ctx, song)
    from ..config import AsrCfg
    from ..stages.asr import clean_text, plan_phrases
    a = ctx.cfg["asr"]
    acfg = AsrCfg(min_gap_s=a["min_gap_s"], min_phrase_s=a["min_phrase_s"], max_phrase_s=a["max_phrase_s"],
                  pad_s=a["pad_s"], max_chars_per_s=a["max_chars_per_s"])
    fz = np.load(sdir / "f0.npz")
    spans = plan_phrases(fz["f0"] > 0, FPS, acfg)      # singer / overlap info is added in reconcile
    voc16 = resample(_load(sdir, "vocals.flac"), SR, ASR_SR)
    clips = [voc16[max(0, int((s - a["pad_s"]) * ASR_SR)): int((e + a["pad_s"]) * ASR_SR)].astype(np.float32)
             for s, e in spans]
    q, w = _asr_backends(ctx)
    import time
    t = time.time()
    tq = q(clips, a["language"]) if clips else []
    tq_s, t = time.time() - t, time.time()
    tw = w(clips, a["language"]) if clips else []
    tw_s = time.time() - t
    phrases = []
    for (s, e), (rq, _), (rw, _) in zip(spans, tq, tw):
        qt, qd = clean_text(rq, e - s, a["max_chars_per_s"])
        wt, wd = clean_text(rw, e - s, a["max_chars_per_s"])
        phrases.append({"start": round(s, 3), "end": round(e, 3), "qwen": qt, "qwen_raw": (rq or "").strip(), "qwen_dropped": qd,
                        "whisper": wt, "whisper_raw": (rw or "").strip(), "whisper_dropped": wd})
    atomic_json(sdir / "asr.json", {"models": {"qwen": a["primary"], "whisper": a["secondary"]},
                                    "phrases": phrases})
    return {"qwen_s": round(tq_s, 1), "whisper_s": round(tw_s, 1), "phrases": len(phrases), "qwen_kept": sum(1 for p in phrases if p["qwen"]),
            "whisper_kept": sum(1 for p in phrases if p["whisper"])}


# ---------------------------------------------------------------------------
# S9a emissions
# ---------------------------------------------------------------------------
def emissions(ctx, song: Song, sdir: Path) -> Dict:
    """MMS_FA log-probabilities (with star token) at 50 fps -> ``mms.npz``."""
    _dup_check(ctx, song)
    from ..config import AlignCfg
    from ..stages.align import MmsFaEmitter
    em = ctx.model("mms", lambda: MmsFaEmitter(AlignCfg(backend="mms_fa", use_star=True,
                                                        device=ctx.cfg["device"]["device"])))
    voc = _load(sdir, "vocals.flac")
    n = len(np.load(sdir / "f0.npz")["f0"])
    logp = em(voc, n)
    atomic_npz(sdir / "mms.npz", logp=logp.astype(np.float16), vocab=json.dumps(em.vocab),
               blank=em.blank, star=-1 if em.star_id is None else em.star_id)
    return {"frames": int(n)}


STAGES = [
    Stage("separate", 1, separate, sections=("separation",), depends=("inventory",), gpu=True),
    Stage("f0", 1, f0, sections=("f0",), depends=("separate",), gpu=True),
    Stage("voices", 2, voices, sections=("voices",), depends=("f0",), gpu=True),
    Stage("asr", 1, asr, sections=("asr",), depends=("f0",), gpu=True),
    Stage("emissions", 1, emissions, sections=(), depends=("f0",), gpu=True),
]

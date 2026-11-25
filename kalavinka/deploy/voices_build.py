"""Build the voice profiles: about 24 held-out voices balanced over eras and (estimated) gender.

Why: the app had 10 voices called "Voice N: singer of <song>", with no gender, a register label from the median *note*
(voice 1 was "low" with Sa at 335 Hz), and no way to filter. The dataset has no gender or singer names (the gender
field is empty on every clip), so this script estimates them and writes a small, editable spec,
``kalavinka/deploy/voice_profiles.json``, that ``build_bundle`` turns into the bundle's voices.

Steps
-----
1. **Candidates**: every singer cluster (``singer_id``) of the held-out val + test songs, so no voice was heard in
   training. Its best clip: band ``accept``, 3-8 s, at least 60 % voiced frames, highest mean confidence.
2. **Gender (estimated, when there are no credits)**: a pretrained wav2vec2 gender classifier (``GENDER_MODEL``, trained on speech) on the
   clip's vocals, averaged over up to 3 clips, combined with the median sung F0. If the classifier is unsure
   (p < 0.7) the pitch decides when it is clear (< 200 Hz male, > 280 Hz female); if they disagree, "unsure".
3. **Register**: low / mid / high = terciles of the median sung F0 *within* the estimated gender, so "low" means low
   for that voice type. The profile also stores the Sa (tonic) and the 5-95 % sung range in Hz.
4. **Pick**: voices 1-10 (the existing bundle voices) keep their numbers, so earlier takes and cache keys stay
   valid. Extra voices are added round-robin over decade x gender cells that have the fewest voices so far, best
   clip first, at most one voice per song, until ``--n`` voices. Cells with no held-out singer are listed as gaps.
5. **Credits** (``--credits voice_credits.json``: per song id, the credited singers ``[{"name", "gender"}]`` from a
   web lookup, with the source URL). The speech-trained classifier turned out unreliable on singing (it called
   several high male film voices female with 99 % confidence), so when credits exist they decide the gender
   (``gender_and_name``) and give the name: ``singer: {"name", "source": "credited singer of the source song",
   "confirmed": false}``. Everything can be edited later in the app (``outputs/voice_profiles.local.json``).

Usage (GPU optional, about a minute)::

    python -m kalavinka.deploy.voices_build --export <PROJECT>/export/v1 --bundle-voices <bundle>/data/voices.jsonl \\
        --out kalavinka/deploy/voice_profiles.json [--credits credits.json] [--n 24] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
DECADES = ("1980s", "1990s", "2000s", "2010s", "2020s")


def pretty_song(song_id: str) -> str:
    """``1991_tu_pyaar_hai_kisi_aur_ka__98MGT9JStKk`` -> ``Tu Pyaar Hai Kisi Aur Ka``."""
    name = song_id.split("__")[0].split("_", 1)[1] if "_" in song_id else song_id
    return " ".join(w.capitalize() for w in name.split("_") if w)


def clip_stats(it: Dict) -> Dict:
    """Seconds, voiced share, median and 5/95 % F0 of a clip, and its mean alignment confidence."""
    f0 = np.array(it["f0"].split(), float)
    v = f0[f0 > 0]
    conf = np.array(str(it.get("conf", "0")).split(), float) if it.get("conf") not in (None, "") else np.zeros(1)
    return {"seconds": len(f0) / 50, "voiced": float(len(v) / max(len(f0), 1)),
            "f0_med": float(np.median(v)) if len(v) else 0.0,
            "f0_lo": float(np.percentile(v, 5)) if len(v) else 0.0, "f0_hi": float(np.percentile(v, 95)) if len(v) else 0.0,
            "conf": float(np.mean(conf)) if conf.size else 0.0}


def candidates(items: Sequence[Dict]) -> Dict[str, Dict]:
    """Per singer: up to 3 good clips (best first) and pooled pitch statistics."""
    by: Dict[str, List[Dict]] = {}
    for it in items:
        by.setdefault(it["singer_id"], []).append(it)
    out = {}
    for sid, its in by.items():
        good = []
        for it in its:
            st = clip_stats(it)
            if it.get("band") == "accept" and 3 <= st["seconds"] <= 8 and st["voiced"] >= 0.6:
                good.append((st["conf"], it, st))
        if not good:
            continue
        good.sort(key=lambda x: -x[0])
        allf0 = np.concatenate([np.array(it["f0"].split(), float) for it in its])
        v = allf0[allf0 > 0]
        song = its[0]["index"].rsplit("_c", 1)[0]
        out[sid] = {"singer_id": sid, "clips": [g[1] for g in good[:3]], "best": good[0][1], "song_id": song,
                    "year": int(song[:4]), "f0_med": float(np.median(v)), "range_hz": [round(float(np.percentile(v, 5)), 1),
                    round(float(np.percentile(v, 95)), 1)], "tonic_hz": float(good[0][1].get("tonic_hz") or 0)}
    return out


def classify_gender(paths: Sequence[Path], device: str = "cpu") -> List[float]:
    """P(female) for each audio file with ``GENDER_MODEL`` (16 kHz mono)."""
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    fe = AutoFeatureExtractor.from_pretrained(GENDER_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(GENDER_MODEL).to(device).eval()
    labels = {i: l.lower() for i, l in model.config.id2label.items()}
    fem = next(i for i, l in labels.items() if l.startswith("f"))
    out = []
    for p in paths:
        y, _ = librosa.load(str(p), sr=16000, mono=True)
        x = fe(y, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            pr = torch.softmax(model(**x).logits, -1)[0]
        out.append(float(pr[fem]))
    return out


def decide_gender(p_female: float, f0_med: float) -> Dict:
    """Combine the classifier and the sung pitch (see the module doc)."""
    pitch = "male" if f0_med < 200 else ("female" if f0_med > 280 else None)
    clf = "female" if p_female >= 0.7 else ("male" if p_female <= 0.3 else None)
    if clf and pitch and clf != pitch:
        g = "unsure"
    else:
        g = clf or pitch or "unsure"
    conf = max(p_female, 1 - p_female)
    return {"gender": g, "gender_p_female": round(p_female, 3), "gender_conf": round(conf, 2),
            "gender_from": "classifier+pitch" if clf and pitch else ("classifier" if clf else ("pitch" if pitch else "-"))}


def gender_and_name(f0_med: float, p_female: float, credit: Optional[Dict]) -> Dict:
    """Gender, its confidence and the credited name for one singer cluster.

    The speech-trained classifier is unreliable on singing (it calls many high male film voices female), so the
    **credits** of the source song decide when they exist: a solo song's singer gender is taken as is (unless the
    sung pitch flatly contradicts it); in a duet the cluster's median pitch picks the side (>= 255 Hz female,
    <= 235 Hz male, in between the classifier). Without credits: ``decide_gender`` (classifier + pitch)."""
    if not credit or not credit.get("singers"):
        return {**decide_gender(p_female, f0_med), "name": None}
    sing = credit["singers"]
    genders = {c["gender"] for c in sing}
    sure = 0.95 if credit.get("confident", True) else 0.8
    if genders == {"male"} and f0_med > 330 or genders == {"female"} and f0_med < 180:
        g = "female" if f0_med > 330 else "male"            # the pitch flatly contradicts the credits
        return {"gender": g, "gender_p_female": round(p_female, 3), "gender_conf": 0.7, "gender_from": "pitch (credits disagree)",
                "name": None}
    if len(genders) == 1:
        g, how, conf = next(iter(genders)), "credits", sure
    else:
        if f0_med >= 255:
            g, conf = "female", sure - 0.1
        elif f0_med <= 235:
            g, conf = "male", sure - 0.1
        else:
            g, conf = ("female" if p_female >= 0.5 else "male"), 0.6
        how = "credits (duet) + pitch"
    names = [c["name"] for c in sing if c["gender"] == g]
    return {"gender": g, "gender_p_female": round(p_female, 3), "gender_conf": round(conf, 2), "gender_from": how,
            "name": names[0] if len(names) == 1 else None, "name_options": names if len(names) > 1 else []}


def registers(singers: Dict[str, Dict]) -> None:
    """low / mid / high = terciles of median F0 within each gender (in place)."""
    for g in ("male", "female", "unsure"):
        meds = sorted(s["f0_med"] for s in singers.values() if s["gender"] == g)
        if not meds:
            continue
        lo, hi = np.percentile(meds, [100 / 3, 200 / 3])
        for s in singers.values():
            if s["gender"] == g:
                s["register"] = "low" if s["f0_med"] < lo else ("mid" if s["f0_med"] <= hi else "high")


def decade(year: int) -> str:
    """1987 -> '1980s'."""
    return f"{year // 10 * 10}s"


def pick(singers: Dict[str, Dict], keep: Sequence[str], n: int) -> List[Dict]:
    """The kept singers (in order) then extra ones, balancing decade x gender cells (see the module doc)."""
    chosen = [singers[s] for s in keep if s in singers]
    used_songs = {s["song_id"] for s in chosen}
    count: Dict[tuple, int] = {}
    for s in chosen:
        count[(decade(s["year"]), s["gender"])] = count.get((decade(s["year"]), s["gender"]), 0) + 1
    pool = sorted((s for s in singers.values() if s["singer_id"] not in keep and s["gender"] != "unsure"
                   and s.get("clips")),
                  key=lambda s: -clip_stats(s["best"])["conf"])
    while len(chosen) < n:
        cells = sorted({(decade(s["year"]), s["gender"]) for s in pool if s["song_id"] not in used_songs},
                       key=lambda c: (count.get(c, 0), c))
        if not cells:
            break
        cell = cells[0]
        s = next(s for s in pool if (decade(s["year"]), s["gender"]) == cell and s["song_id"] not in used_songs)
        chosen.append(s)
        pool.remove(s)
        used_songs.add(s["song_id"])
        count[cell] = count.get(cell, 0) + 1
    return chosen


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--bundle-voices", type=Path, required=True, help="the current bundle's data/voices.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--credits", type=Path, default=None)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    items = []
    for split in ("val", "test"):
        items += [json.loads(l) for l in (a.export / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    singers = candidates(items)
    old = [json.loads(l) for l in a.bundle_voices.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = [v["item"]["singer_id"] for v in old]
    for v in old:                                   # the kept voices use their existing prompt clip
        s = singers.setdefault(v["item"]["singer_id"], {"singer_id": v["item"]["singer_id"], "song_id":
                               v["item"]["index"].rsplit("_c", 1)[0], "year": int(v["year"]), "clips": []})
        full = next((it for it in items if it["index"] == v["item"]["index"]), None)
        if full is None:
            raise SystemExit(f"voice {v['no']} clip {v['item']['index']} not in val/test")
        st = clip_stats(full)
        s["best_for_keep"] = True
        s.update({"best": full, "clips": [full] + [c for c in s["clips"] if c["index"] != full["index"]][:2]})
        s.setdefault("f0_med", st["f0_med"]); s.setdefault("range_hz", [round(st["f0_lo"], 1), round(st["f0_hi"], 1)])
        s.setdefault("tonic_hz", float(full.get("tonic_hz") or 0))
    paths, owners = [], []
    for sid, s in singers.items():
        for c in s["clips"]:
            paths.append(a.export / c["wav"]); owners.append(sid)
    probs = classify_gender(paths, a.device)
    credits = json.loads(a.credits.read_text(encoding="utf-8")).get("songs", {}) if a.credits else {}
    for sid, s in singers.items():
        ps = [p for p, o in zip(probs, owners) if o == sid]
        s.update(gender_and_name(s["f0_med"], float(np.mean(ps)) if ps else 0.5, credits.get(s["song_id"])))
    registers(singers)
    chosen = pick(singers, keep, a.n)
    out = []
    for k, s in enumerate(chosen, 1):
        name = s.get("name")
        out.append({"no": k, "index": s["best"]["index"], "singer_id": s["singer_id"], "song_id": s["song_id"],
                    "song": pretty_song(s["song_id"]), "year": s["year"], "decade": decade(s["year"]),
                    "gender": s["gender"], "gender_p_female": s["gender_p_female"], "gender_conf": s["gender_conf"],
                    "gender_from": s["gender_from"], "register": s["register"], "f0_median_hz": round(s["f0_med"], 1),
                    "range_hz": s["range_hz"], "sa_hz": round(s["tonic_hz"], 1),
                    "singer": ({"name": name, "source": "credited singer of the source song",
                                "confirmed": False} if name else None),
                    "name_options": s.get("name_options", []),
                    "credit_source": (credits.get(s["song_id"]) or {}).get("source")})
    cells = {(d, g): 0 for d in DECADES for g in ("female", "male")}
    for v in out:
        if (v["decade"], v["gender"]) in cells:
            cells[(v["decade"], v["gender"])] += 1
    spec = {"built_from": str(a.export), "gender_model": GENDER_MODEL, "voices": out,
            "gaps": [f"{d} {g}" for (d, g), c in cells.items() if c == 0],
            "all_singers": [{k: s[k] for k in ("singer_id", "song_id", "year", "gender", "gender_p_female", "f0_med",
                                               "register")} for s in singers.values()]}
    a.out.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out)} voices -> {a.out}; per decade x gender: "
          + ", ".join(f"{d} {g}: {c}" for (d, g), c in cells.items()) + (f"; gaps: {spec['gaps']}" if spec["gaps"] else ""))


if __name__ == "__main__":
    main()

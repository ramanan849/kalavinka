"""Voice profiles: who each bundled voice is (estimated gender, register, era, source song, credited singer), filters,
and your own edits.

Sources, merged in this order (later wins):

1. ``data/voices.jsonl``: the voices the model can sing with (number, prompt clip, source song, year).
2. ``data/voice_profiles.json``: profiles written by ``voices_build`` / ``build_bundle`` (estimated gender with its
   confidence, register within that gender, Sa, sung range, decade, credited singer of the source song).
3. ``outputs/voice_profiles.local.json``: **your edits** in the app or with ``./sing voice-edit`` (a singer's name,
   a corrected gender, notes). Kept in ``outputs/`` so they survive bundle updates; every edit is also logged.

Old bundles without (2) still work: gender is then "unsure" and the register comes from ``voices.jsonl``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .sargam import fold_tonic, voice_f0_median

EDITABLE = ("name", "gender", "register", "notes", "singer_confirmed")
GENDERS = ("female", "male", "unsure")
ERAS = ("1980s", "1990s", "2000s", "2010s", "2020s")
REGISTERS = ("low", "mid", "high")


def _read_json(path: Path, default):
    """JSON file or ``default`` (missing or unreadable)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load(bundle: Path, outputs: Optional[Path] = None) -> List[Dict]:
    """All voice profiles, sorted by number (see the module doc for the sources)."""
    bundle = Path(bundle)
    voices = [json.loads(l) for l in (bundle / "data" / "voices.jsonl").read_text(encoding="utf-8").split("\n")
              if l.strip()]
    spec = {v["no"]: v for v in _read_json(bundle / "data" / "voice_profiles.json", {}).get("voices", [])}
    local = _read_json(Path(outputs) / "voice_profiles.local.json", {}) if outputs else {}
    out = []
    for v in voices:
        s = spec.get(v["no"], {})
        year = int(v.get("year") or s.get("year") or 0)
        singer = s.get("singer") or {}
        p = {"no": v["no"], "song": v.get("song") or s.get("song", ""), "year": year,
             "decade": s.get("decade") or (f"{year // 10 * 10}s" if year else ""),
             "gender": s.get("gender", "unsure"), "gender_conf": s.get("gender_conf"),
             "gender_from": s.get("gender_from", ""), "register": s.get("register") or v.get("register", ""),
             "sa_hz": round(fold_tonic(float(v["item"].get("tonic_hz") or 0), voice_f0_median(v["item"])), 1),
             "range_hz": s.get("range_hz"), "f0_median_hz": s.get("f0_median_hz"),
             "name": singer.get("name") or "", "name_source": singer.get("source", ""),
             "singer_confirmed": bool(singer.get("confirmed")), "notes": "", "sample": v["item"]["wav"],
             "edited": False}
        edit = local.get(str(v["no"]), {})
        for k in EDITABLE:
            if k in edit:
                p[k] = edit[k]
                p["edited"] = True
        if edit.get("name"):
            p["name_source"] = "your edit"
        p["label"] = label(p)
        out.append(p)
    return out


def label(p: Dict) -> str:
    """Short dropdown label: 'Voice 7 · female · mid · 2010s · <name>'."""
    parts = [f"Voice {p['no']}"]
    if p.get("gender") and p["gender"] != "unsure":
        parts.append(p["gender"])
    parts += [x for x in (p.get("register"), p.get("decade")) if x]
    if p.get("name"):
        parts.append(p["name"])
    return " · ".join(parts)


def card(p: Dict) -> str:
    """Markdown profile card."""
    g = p["gender"] if p["gender"] != "unsure" else "not sure"
    est = "" if p.get("edited") and p.get("gender") else (
        f" (estimated, {int(round((p.get('gender_conf') or 0) * 100))}% sure)" if p.get("gender_conf") else " (estimated)")
    rng = (f"{p['range_hz'][0]:.0f}-{p['range_hz'][1]:.0f} Hz sung" if p.get("range_hz") else "")
    who = (f"**{p['name']}**" + (f" ({p['name_source']}{', confirmed' if p.get('singer_confirmed') else ''})"
                                 if p.get("name_source") else "")) if p.get("name") else "name not set (add it below)"
    lines = [f"### Voice {p['no']}", f"- Singer: {who}", f"- Gender: {g}{est}",
             f"- Register: {p.get('register') or '?'}" + (f", Sa = {p['sa_hz']:.0f} Hz" if p.get("sa_hz") else "")
             + (f", {rng}" if rng else ""),
             f"- From: *{p['song']}* ({p['year']}), {p.get('decade', '')}"]
    if p.get("notes"):
        lines.append(f"- Notes: {p['notes']}")
    return "\n".join(lines)


def filter_profiles(profiles: List[Dict], gender: str = "any", era: str = "any", register: str = "any",
                    text: str = "") -> List[Dict]:
    """Profiles matching every filter ('any' = no filter; ``text`` searches name, song and notes)."""
    t = (text or "").strip().lower()
    return [p for p in profiles
            if (gender in ("any", "", None) or p["gender"] == gender)
            and (era in ("any", "", None) or p["decade"] == era)
            and (register in ("any", "", None) or p["register"] == register)
            and (not t or t in f"{p['name']} {p['song']} {p['notes']}".lower())]


def resolve(profiles: List[Dict], spec) -> Optional[int]:
    """Voice number from a number, 'Voice 7 ...' label, or a name / song fragment (None for empty / auto)."""
    if spec in (None, "", "auto"):
        return None
    s = str(spec).strip()
    if s.isdigit():
        return int(s)
    if s.lower().startswith("voice "):
        head = s.split()[1].strip(":·,")
        if head.isdigit():
            return int(head)
    hits = [p["no"] for p in profiles if s.lower() in f"{p['name']} {p['song']}".lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no voice matches {spec!r} (see: ./sing voices)")
    raise ValueError(f"{spec!r} matches voices {hits}; be more specific or give the number")


def save_edit(outputs: Path, no: int, **fields) -> Dict:
    """Store your edits of one voice in ``outputs/voice_profiles.local.json`` (only the given fields change)."""
    bad = set(fields) - set(EDITABLE)
    if bad:
        raise ValueError(f"cannot edit {sorted(bad)}; editable: {EDITABLE}")
    if "gender" in fields and fields["gender"] not in GENDERS:
        raise ValueError(f"gender must be one of {GENDERS}")
    path = Path(outputs) / "voice_profiles.local.json"
    data = _read_json(path, {})
    cur = data.setdefault(str(int(no)), {})
    cur.update({k: v for k, v in fields.items()})
    cur["edited_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(outputs).mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return cur

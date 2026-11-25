"""Build the two shareable zips: ``kalavinka_colab.zip`` (notebook) and ``kalavinka_mac.zip`` (local pipeline).

Both hold the same core, laid out so that the unchanged project code runs as-is::

    kalavinka_app/                  the singer, command line and web UI (this package's singer/cli/app)
    vendor/kalavinka/...            copies of the project modules inference needs (sx, eval, ir subsets)
    vendor/saregama/...             G2P + phone set (pure Python) and the path helper
    vendor/SoulX-Singer/soulxsinger SoulX-Singer's code (no weights; model.pt is downloaded on first use)
    weights/kalavinka_v4l/best.pt   our final model (LoRA + Hindi phone rows), run L4_v2
    weights/contour_cflow/best.pt   stage-1 contour model C-flow, run I2_v1
    data/phone_set_indic.json       SoulX phone table + 62 Hindi phones
    data/tunes.jsonl, voices.jsonl  the tune and voice library (held-out test clips) + data/wavs/
    data/novelty.bin                hashes of 4-syllable phrases of the training lyrics (not the lyrics)
    requirements.txt  README.md  NOTICE.md

The Colab zip adds ``Kalavinka_Colab.ipynb``; the Mac zip adds ``setup_mac.sh``, ``sing`` and ``MAC.md``.

Tune choice: accepted **test** clips (songs never seen in training) that are pleasant to sing new words to:
6-20 syllables, 0.3-0.7 s per syllable, at most 2 slur notes, no leading rest, at most 8 s, and another clip of the
same singer to act as the voice prompt; at most 2 tunes per song, up to ``--n-tunes``. Each distinct singer's prompt
clip becomes a voice.

Usage::

    .venv-sx/bin/python -m kalavinka.deploy.build_bundle --out <PROJECT>/bundles [--n-tunes 20]
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
HERE = Path(__file__).resolve().parent

VENDOR_FILES = [
    "kalavinka/__init__.py", "kalavinka/sx/__init__.py", "kalavinka/sx/adapter.py", "kalavinka/sx/phones.py",
    "kalavinka/sx/train.py", "kalavinka/sx/data.py", "kalavinka/eval/__init__.py", "kalavinka/eval/render.py",
    "kalavinka/eval/sing_new.py", "kalavinka/ir/__init__.py", "kalavinka/ir/features.py", "kalavinka/ir/model.py",
    "saregama/g2p.py", "saregama/phoneset.py", "saregama/paths.py",
]
APP_FILES = ["singer.py", "cli.py", "app.py", "sargam.py", "history.py"]
DROP_FIELDS = ("conf", "energy_db", "hnr_db", "h1h2_db", "f0_agree", "features", "swara", "note_cents")


def pretty_song(song_id: str) -> str:
    """``1991_tu_pyaar_hai_kisi_aur_ka__98MGT9JStKk`` -> ``Tu Pyaar Hai Kisi Aur Ka``."""
    name = song_id.split("__")[0].split("_", 1)[1] if "_" in song_id else song_id
    return " ".join(w.capitalize() for w in name.split("_") if w)


def register(item: Dict) -> str:
    """low / mid / high from the median note (a rough male / female hint)."""
    ps = sorted(int(p) for p in item["note_pitch"].split() if int(p) > 0)
    med = ps[len(ps) // 2] if ps else 60
    return "low" if med < 55 else ("mid" if med < 64 else "high")


def choose_tunes(items: List[Dict], n: int) -> List[Dict]:
    """Singable accepted test clips with a same-singer prompt (see the module doc); returns tune dicts."""
    from kalavinka.eval.sing_new import n_syllables
    from kalavinka.sx.data import prompt_map
    pm = prompt_map([{"band": "accept", "singer_id": it["singer_id"], "time": it["time"]} for it in items])
    per_song: Dict[str, int] = {}
    out = []
    for i, it in sorted(enumerate(items), key=lambda x: x[1]["index"]):
        types = it["note_type"].split()
        ns = n_syllables(it)
        sung = sum(float(d) for d, t in zip(it["duration"].split(), types) if t != "1")
        secs = int(it["n_frames"]) / 50
        if pm[i] is None or not (6 <= ns <= 20) or not (0.3 <= sung / ns <= 0.7) or types.count("3") > 2 \
                or types[0] == "1" or secs > 8:
            continue
        song = it["index"].rsplit("_c", 1)[0]
        if per_song.get(song, 0) >= 2:
            continue
        per_song[song] = per_song.get(song, 0) + 1
        out.append({"item": it, "prompt": items[pm[i]], "syllables": ns, "seconds": round(secs, 1), "song_id": song})
    # spread over songs: round-robin by song so a short list still covers many songs
    by_song: Dict[str, List[Dict]] = {}
    for t in out:
        by_song.setdefault(t["song_id"], []).append(t)
    order, k = [], 0
    while len(order) < min(n, len(out)):
        for s in sorted(by_song):
            if k < len(by_song[s]) and len(order) < n:
                order.append(by_song[s][k])
        k += 1
    return sorted(order, key=lambda t: (t["song_id"][:4], t["item"]["index"]))


def clean_checkpoint(src: Path, dst: Path) -> None:
    """Copy a run's best.pt with plain-string ``args`` (no Path objects), so it loads with PyTorch's safe
    ``weights_only=True`` default (PyTorch >= 2.6); only the weights and args are kept."""
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    out = {k: v for k, v in ck.items() if k in ("model", "trainable")}
    out["args"] = {k: (v if isinstance(v, (int, float, bool)) or v is None else str(v)) for k, v in ck["args"].items()}
    torch.save(out, dst)


def slim(item: Dict) -> Dict:
    """Export row without the fields inference never reads."""
    return {k: v for k, v in item.items() if k not in DROP_FIELDS}


def novelty_hashes(train_jsonl: Path) -> bytes:
    """8-byte hashes of every 4-syllable phrase of the training lyrics (sorted, deduplicated)."""
    from kalavinka.deploy.singer import SPAN, span_hash
    seen = set()
    for line in train_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        syl = [s for s in json.loads(line)["text"].split() if s != "<SP>"]
        for i in range(max(1, len(syl) - SPAN + 1)):
            seen.add(span_hash(syl[i:i + SPAN]))
    return b"".join(sorted(seen))


def build_core(dst: Path, export: Path, n_tunes: int) -> Dict:
    """Write the shared core into ``dst``; returns a summary."""
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "kalavinka_app").mkdir(parents=True)
    for f in APP_FILES:
        shutil.copy2(HERE / f, dst / "kalavinka_app" / f)
    (dst / "kalavinka_app" / "__init__.py").write_text('"""Kalavinka: Hindi film-song singing (see README.md)."""\n')
    shutil.copy2(HERE / "bundle_main.py", dst / "kalavinka_app" / "__main__.py")
    for rel in VENDOR_FILES:
        (dst / "vendor" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst / "vendor" / rel)
    (dst / "vendor" / "saregama" / "__init__.py").write_text('"""Vendored subset of saregama: G2P and phone set."""\n')
    shutil.copytree(PROJECT / "SoulX-Singer" / "soulxsinger", dst / "vendor" / "SoulX-Singer" / "soulxsinger",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(PROJECT / "SoulX-Singer" / "LICENSE", dst / "vendor" / "SoulX-Singer" / "LICENSE")
    (dst / "vendor" / "SoulX-Singer" / "pretrained_models" / "SoulX-Singer").mkdir(parents=True)
    for name, src in (("kalavinka_v4l", PROJECT / "runs" / "L4_v2" / "best.pt"),
                      ("contour_cflow", PROJECT / "runs" / "I2_v1" / "best.pt")):
        (dst / "weights" / name).mkdir(parents=True)
        clean_checkpoint(src, dst / "weights" / name / "best.pt")
    data = dst / "data"
    (data / "wavs").mkdir(parents=True)
    shutil.copy2(export.parent / "soulx" / "phone_set_indic.json", data / "phone_set_indic.json")

    from kalavinka.eval.render import load_items
    tunes = choose_tunes(load_items(export / "test.jsonl"), n_tunes)
    voices: Dict[str, Dict] = {}
    with open(data / "tunes.jsonl", "w", encoding="utf-8") as ft:
        for k, t in enumerate(tunes, 1):
            pr = t["prompt"]
            v = voices.setdefault(pr["singer_id"], {"no": len(voices) + 1, "item": slim(pr), "song": pretty_song(t["song_id"]),
                                                    "year": int(t["song_id"][:4]), "register": register(pr)})
            v["name"] = f"Voice {v['no']}: singer of {v['song']} ({v['year']}), {v['register']} register"
            for it in (t["item"], pr):
                shutil.copy2(export / it["wav"], data / it["wav"])
            ft.write(json.dumps({"no": k, "song": pretty_song(t["song_id"]), "year": int(t["song_id"][:4]),
                                 "syllables": t["syllables"], "seconds": t["seconds"],
                                 "lyric": " ".join(s for s in t["item"]["text"].split() if s != "<SP>"),
                                 "voice": v["no"], "item": slim(t["item"])}, ensure_ascii=False) + "\n")
    with open(data / "voices.jsonl", "w", encoding="utf-8") as fv:
        for v in sorted(voices.values(), key=lambda v: v["no"]):
            fv.write(json.dumps(v, ensure_ascii=False) + "\n")
    (data / "novelty.bin").write_bytes(novelty_hashes(export / "train.jsonl"))
    for f in ("requirements.txt", "README.md", "NOTICE.md"):
        shutil.copy2(HERE / "bundle_files" / f, dst / f)
    return {"tunes": len(tunes), "voices": len(voices)}


def zip_dir(src: Path, zpath: Path, top: str) -> Path:
    """Zip ``src`` as folder ``top`` inside the archive (keeps the executable bit of scripts)."""
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                zi = zipfile.ZipInfo.from_file(f, f"{top}/{f.relative_to(src)}")
                zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, f.read_bytes())
    return zpath


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Build the core once, then the Colab and Mac folders and zips."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, default=PROJECT / "export" / "v1")
    p.add_argument("--out", type=Path, default=PROJECT / "bundles")
    p.add_argument("--n-tunes", type=int, default=20)
    a = p.parse_args(argv)
    extra = HERE / "bundle_files"
    for name, files in (("kalavinka_colab", ["Kalavinka_Colab.ipynb"]), ("kalavinka_mac", ["setup_mac.sh", "sing", "MAC.md"])):
        d = a.out / name
        summary = build_core(d, a.export, a.n_tunes)
        for f in files:
            shutil.copy2(extra / f, d / f)
        z = zip_dir(d, a.out / f"{name}.zip", name)
        shutil.rmtree(d)                                   # keep only the zip
        print(f"{z}  ({z.stat().st_size / 1e6:.0f} MB, {summary['tunes']} tunes, {summary['voices']} voices)")


if __name__ == "__main__":
    main()

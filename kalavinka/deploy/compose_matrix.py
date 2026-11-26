"""Sing a set of compositions with many voices and both models, resumably, with a progress log and an ETA.

Why: for the demo we want every demo-song composition sung by a female and a male voice, with Kalavinka and with
base SoulX-Singer, and then by every voice profile. That is hundreds of takes; this loads the models once per worker,
skips takes that already exist (so it can be stopped and restarted), and logs progress and an ETA to one file that
several workers (one per GPU) share.

What it writes (``--out``)::

    <out>/<composition>/<model>/v17_female_Anuradha_Paudwal.wav   the take (24 kHz)
    <out>/<composition>/<model>/v17_female_Anuradha_Paudwal.json  voice, model, seconds, how closely it followed
    <out>/progress.jsonl     one line per finished take (all workers)
    <out>/progress.log       human-readable: "[done/total] take ... | elapsed | ETA"
    <out>/plan.json          the full job list (voices in order of priority)

Voices (``--voices``): a comma list of numbers and the words ``female``, ``male``, ``all`` (every remaining voice
of that kind), in priority order. ``17,9,all`` sings voices 17 and 9 for every composition first, then everyone
else. ``--pair 17,9`` is a shortcut for the first phase. Models: ``--models kalavinka,base``.

Several GPUs: start one worker per GPU with ``--shard k/n`` (jobs are dealt round-robin, so the first voices finish
first on all workers)::

    for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$k python -m kalavinka_app.compose_matrix --bundle . \\
        --comps compositions/*.json --voices 17,9,all --out renders --shard $k/3 & done
    tail -f renders/progress.log

Inside the Mac/Colab bundle the module is ``kalavinka_app.compose_matrix``; in the repo ``kalavinka.deploy.
compose_matrix`` (then ``--bundle`` must point at a built bundle, which holds the weights).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def voice_table(bundle: Path) -> List[Dict]:
    """The bundle's voice profiles: number, gender, register, Sa, credited singer."""
    d = json.loads((bundle / "data" / "voice_profiles.json").read_text(encoding="utf-8"))
    out = []
    for v in d["voices"]:
        s = v.get("singer") or {}
        name = s.get("name") if isinstance(s, dict) else s
        out.append({"no": int(v["no"]), "gender": v.get("gender") or "unsure", "register": v.get("register", ""),
                    "sa_hz": v.get("sa_hz"), "singer": name or "unknown", "song": v.get("song", "")})
    return out


def pick_voices(spec: str, table: Sequence[Dict]) -> List[Dict]:
    """``17,9,female,all`` -> voice dicts in that priority order, each voice once."""
    by_no = {v["no"]: v for v in table}
    out: List[Dict] = []
    for tok in [t.strip().lower() for t in spec.split(",") if t.strip()]:
        if tok.isdigit():
            if int(tok) not in by_no:
                raise SystemExit(f"there is no voice {tok}")
            cand = [by_no[int(tok)]]
        elif tok in ("female", "male"):
            cand = [v for v in table if v["gender"] == tok]
        elif tok == "all":
            cand = list(table)
        else:
            raise SystemExit(f"--voices: cannot read {tok!r} (numbers, female, male, all)")
        out += [v for v in cand if v not in out]
    return out


def take_name(v: Dict) -> str:
    """File stem of a voice's take: ``v17_female_Anuradha_Paudwal``."""
    return f"v{v['no']:02d}_{v['gender']}_{re.sub(r'[^A-Za-z0-9]+', '_', v['singer']).strip('_')}"


def plan_jobs(comps: Sequence[Path], voices: Sequence[Dict], models: Sequence[str], out: Path) -> List[Dict]:
    """All (composition, voice, model) takes, ordered voice first (priority), then composition, then model."""
    jobs = []
    for v in voices:
        for c in comps:
            for m in models:
                jobs.append({"comp": str(c), "name": c.stem, "voice": v, "model": m,
                             "wav": str(out / c.stem / m / f"{take_name(v)}.wav")})
    return jobs


def fmt(sec: float) -> str:
    """Seconds -> ``1h 02m`` / ``3m 10s``."""
    sec = int(max(sec, 0))
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"


def log_line(out: Path, text: str) -> None:
    """Append one line to the shared log (and print it)."""
    stamp = time.strftime("%H:%M:%S")
    with open(out / "progress.log", "a", encoding="utf-8") as f:
        f.write(f"{stamp} {text}\n")
    print(f"{stamp} {text}", flush=True)


def progress(out: Path, total: int) -> str:
    """``[done/total] elapsed | ETA``, from every worker's finished takes since the run started."""
    done_rows = [json.loads(l) for l in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] \
        if (out / "progress.jsonl").exists() else []
    start = float((out / "started").read_text()) if (out / "started").exists() else time.time()
    now = time.time()
    fresh = [r for r in done_rows if r.get("finished", 0) >= start and not r.get("skipped")]
    done = sum(1 for p in json.loads((out / "plan.json").read_text())["jobs"] if Path(p["wav"]).exists())
    rate = len(fresh) / max(now - start, 1e-6)
    eta = (total - done) / rate if rate > 0 else float("nan")
    return f"[{done}/{total}] elapsed {fmt(now - start)} | ETA {fmt(eta) if np.isfinite(eta) else '?'}"


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    import soundfile as sf
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--comps", nargs="+", type=Path, required=True, help="composition JSON files")
    p.add_argument("--voices", default="17,9,all")
    p.add_argument("--models", default="kalavinka,base")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--shard", default="0/1", help="k/n: this worker takes every n-th job starting at k")
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args(argv)
    try:                                               # inside a bundle
        from kalavinka_app.compose import ComposeSinger, normalise
        from kalavinka_app.singer import KalavinkaSinger, default_steps
    except ImportError:                                # in the repo
        from kalavinka.deploy.compose import ComposeSinger, normalise
        from kalavinka.deploy.singer import KalavinkaSinger, default_steps
    a.out.mkdir(parents=True, exist_ok=True)
    table = voice_table(a.bundle)
    voices = pick_voices(a.voices, table)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    comps = sorted(set(Path(c) for c in a.comps))
    jobs = plan_jobs(comps, voices, models, a.out)
    k, n = (int(x) for x in a.shard.split("/"))
    if k == 0:
        (a.out / "plan.json").write_text(json.dumps({"voices": voices, "models": models,
                                                     "comps": [str(c) for c in comps], "jobs": jobs},
                                                    ensure_ascii=False, indent=1))
        if not (a.out / "started").exists():
            (a.out / "started").write_text(str(time.time()))
    else:
        for _ in range(60):                            # worker 0 writes the plan
            if (a.out / "plan.json").exists():
                break
            time.sleep(1)
    mine = [j for i, j in enumerate(jobs) if i % n == k]
    log_line(a.out, f"worker {k}/{n}: {len(mine)} of {len(jobs)} takes ({len(comps)} compositions x "
                    f"{len(voices)} voices x {len(models)} models)")
    if a.plan_only:
        return
    singer = KalavinkaSinger(a.bundle, device=a.device, verbose=False)
    cs = ComposeSinger(singer)
    steps = int(a.steps or default_steps(singer.device))
    for j in mine:
        wav_path = Path(j["wav"])
        if wav_path.exists():
            continue
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        v = j["voice"]
        t0 = time.time()
        row = {"comp": j["name"], "voice": v["no"], "gender": v["gender"], "singer": v["singer"], "model": j["model"],
               "worker": k}
        try:
            comp = normalise(json.loads(Path(j["comp"]).read_text(encoding="utf-8")))
            comp["voice"] = v["no"]
            wav, sr, info = cs.sing(comp, j["model"], v["no"], steps, a.seed)
            tmp = wav_path.with_suffix(".tmp.wav")
            sf.write(tmp, np.asarray(wav, np.float32), sr)
            os.replace(tmp, wav_path)
            f = info.get("follow", {})
            row.update({"seconds": round(len(wav) / sr, 2), "render_s": round(time.time() - t0, 1),
                        "within_50c": f.get("within_50c"), "mean_abs_cents": f.get("mean_abs_cents"),
                        "file": str(wav_path)})
            wav_path.with_suffix(".json").write_text(json.dumps({**row, "voice_profile": v, "steps": steps,
                                                                 "seed": a.seed, "info": info}, ensure_ascii=False,
                                                                default=str, indent=1))
            status = (f"{j['name']} | v{v['no']} {v['singer']} ({v['gender']}) | {j['model']} | "
                      f"{row['seconds']} s audio in {row['render_s']} s | follow {100 * (f.get('within_50c') or 0):.0f}%")
        except Exception as e:                         # one bad take must not stop the run
            row.update({"error": f"{type(e).__name__}: {e}", "render_s": round(time.time() - t0, 1)})
            status = f"{j['name']} | v{v['no']} | {j['model']} | FAILED: {row['error'][:160]}"
        row["finished"] = time.time()
        with open(a.out / "progress.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_line(a.out, f"{progress(a.out, len(jobs))} | {status}")
    log_line(a.out, f"worker {k}/{n} finished. {progress(a.out, len(jobs))}")


if __name__ == "__main__":
    main()

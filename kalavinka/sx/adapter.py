"""SoulX-Singer <-> P1 export adapter: Indic phone table, ``in_`` note strings, checks.

Why
---
P1 exports notes as ``in_t-u-m`` (one ``in_`` prefix, phones joined by ``-``;
``saregama/phoneset.py``).  Upstream ``DataProcessor.preprocess`` only expands
the ``en_`` prefix that way; any other string is looked up as one token, so an
``in_`` note raises ``KeyError``.  Instead of patching upstream (the repo ships
the patch text as ``saregama.export.DATA_PROCESSOR_PATCH``), this module

* builds the extended phone table with the repo's own
  ``saregama.phoneset.extend_soulx_phoneset`` (62 ``in_*`` tokens appended at
  rows 2820+, which the 3000-row embedding already has -- no resize);
* subclasses ``DataProcessor`` as ``IndicDataProcessor`` whose ``preprocess``
  expands ``in_`` exactly like ``en_`` (a ``<SEP>`` after the phones);
* normalises export items (durations rebuilt from integer frames so
  ``int(sum * 50)`` can never round down; optional nearest-English remap for
  the zero-shot baseline R0);
* checks a whole jsonl: every item passes the processor and
  ``len(mel2note) == n_frames == len(f0)``.

Usage::

    python -m kalavinka.sx.adapter --phones-out <dir>/phone_set_indic.json \
        --check <export>/train.jsonl [--check ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from saregama.paths import SOULX_REPO
from saregama.phoneset import BY_NAME, PREFIX, embedding_init_map, extend_soulx_phoneset
from soulxsinger.utils.data_processor import DataProcessor

FPS = 50
UPSTREAM_PHONES = SOULX_REPO / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
MULTI_PREFIXES = ("en_", PREFIX)


def build_phone_table(out: Path, upstream: Path = UPSTREAM_PHONES) -> Dict[str, int]:
    """Write upstream phone_set.json + the Indic tokens to ``out``; returns token -> row of the added ones."""
    out.parent.mkdir(parents=True, exist_ok=True)
    return extend_soulx_phoneset(upstream, out)


class IndicDataProcessor(DataProcessor):
    """Upstream DataProcessor whose multi-phone note branch accepts ``in_`` as well as ``en_``."""

    def preprocess(self, note_duration: List[float], phonemes: List[str], note_pitch: List[int],
                   note_type: List[int]):
        """Upstream ``preprocess`` with ``MULTI_PREFIXES`` instead of the hard-coded ``en_``
        (same <BOW>/<EOW>/<SEP> layout and mel2note filling, line for line)."""
        sample_rate, hop_size = self.sample_rate, self.hop_size
        duration = sum(note_duration) * sample_rate / hop_size
        mel2note = torch.zeros(int(duration), dtype=torch.long)
        ph_locations, new_phonemes, note2origin = [], [], []
        dur_sum = 0
        for ph_idx in range(len(phonemes)):
            dur = int(np.round(dur_sum * sample_rate / hop_size))
            dur = min(dur, len(mel2note) - 1)
            new_phonemes.append("<BOW>")
            note2origin.append(ph_idx)
            pre = phonemes[ph_idx][:3]
            if pre in MULTI_PREFIXES:
                phs = [pre + x for x in phonemes[ph_idx][3:].split("-")] + ["<SEP>"]
                ph_locations.append([dur, max(1, len(phs))])
                new_phonemes.extend(phs)
                note2origin.extend([ph_idx] * len(phs))
            else:
                ph_locations.append([dur, 1])
                new_phonemes.append(phonemes[ph_idx])
                note2origin.append(ph_idx)
            new_phonemes.append("<EOW>")
            note2origin.append(ph_idx)
            dur_sum += note_duration[ph_idx]
        ph_idx = 1
        for idx, (i, j) in enumerate(ph_locations):
            nxt = ph_locations[idx + 1][0] if idx < len(ph_locations) - 1 else len(mel2note)
            if i >= len(mel2note) or i + j > len(mel2note):
                break
            if i < len(mel2note) and mel2note[i] > 0:
                while i < len(mel2note) and mel2note[i] > 0:
                    i += 1
            mel2note[i] = ph_idx
            k = i + 1
            while k + j < nxt:
                mel2note[k: k + j] = torch.arange(ph_idx, ph_idx + j) + 1
                k += j
            mel2note[nxt - 1] = ph_idx + j + 1
            ph_idx += j + 2
        new_phonemes = ["<PAD>"] + new_phonemes
        return {
            "phoneme": torch.tensor([self.phone2idx[x] for x in new_phonemes], device=self.device).unsqueeze(0),
            "note_pitch": torch.tensor([0] + [note_pitch[k] for k in note2origin], device=self.device).unsqueeze(0),
            "note_type": torch.tensor([1] + [note_type[k] for k in note2origin], device=self.device).unsqueeze(0),
            "mel2note": mel2note.clone().detach().to(self.device).unsqueeze(0),
        }


def exact_durations(item: Dict) -> str:
    """Duration string rebuilt from integer frames per note (sum == n_frames exactly,
    with the 0.1 ms nudge on the last note that keeps ``int(sum*50)`` from rounding down)."""
    frames = [int(round(float(d) * FPS)) for d in item["duration"].split()]
    frames[-1] += int(item["n_frames"]) - sum(frames)
    durs = [f / FPS for f in frames]
    return " ".join([f"{d:.2f}" for d in durs[:-1]] + [f"{durs[-1] + 1e-4:.4f}"])


def nearest_english(note: str) -> str:
    """``in_k-aa-m`` -> ``en_K-AA1-M`` via each phone's first ``init`` token (zero-shot R0)."""
    if not note.startswith(PREFIX):
        return note
    m = embedding_init_map()
    out = [m[PREFIX + p][0][3:] for p in note[len(PREFIX):].split("-")]
    return "en_" + "-".join(out)


def normalise_item(item: Dict, remap_english: bool = False) -> Dict:
    """Copy of an export item ready for the processor (exact durations; optional R0 remap)."""
    it = dict(item)
    it["duration"] = exact_durations(item)
    if remap_english:
        it["phoneme"] = " ".join(nearest_english(p) for p in item["phoneme"].split())
    return it


def check_jsonl(path: Path, proc: DataProcessor) -> Dict:
    """Run every item of an export jsonl through ``proc``; count passes and list failures."""
    ok, bad = 0, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        try:
            out = proc.process(normalise_item(item))
            n = out["mel2note"].shape[1]
            nf = len(item["f0"].split())
            if not (n == int(item["n_frames"]) == nf):
                raise ValueError(f"frames: mel2note {n}, n_frames {item['n_frames']}, f0 {nf}")
            ok += 1
        except Exception as e:  # noqa: BLE001 - report every failure
            bad.append({"index": item.get("index"), "error": f"{type(e).__name__}: {e}"})
    return {"file": str(path), "ok": ok, "failed": len(bad), "failures": bad[:20]}


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: build the Indic phone table and check export jsonl files against it."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phones-out", type=Path, required=True)
    p.add_argument("--check", type=Path, action="append", default=[])
    a = p.parse_args(argv)
    added = build_phone_table(a.phones_out)
    print(f"{len(added)} Indic tokens -> {a.phones_out} (rows {min(added.values())}-{max(added.values())})")
    proc = IndicDataProcessor(hop_size=480, sample_rate=24000, phoneset_path=str(a.phones_out), device="cpu")
    for f in a.check:
        print(json.dumps(check_jsonl(f, proc), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

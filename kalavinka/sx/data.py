"""Training data for the SoulX fine-tune: a per-split cache and [prompt | target] batches.

Cache (``build_cache``)
-----------------------
One ``<split>.pt`` per export split holding, per clip: note tokens / pitch /
type and ``mel2note`` from ``kalavinka.sx.adapter.IndicDataProcessor`` (the
upstream frame arithmetic), the 50 fps F0, the per-frame confidence (export
field ``conf``: MMS log-probability per syllable frame, 0 on rests, -99 on
voiced frames no label explains; all 0 when the export predates the field),
the mel target computed with SoulX's own ``MelSpectrogramEncoder`` (so the
targets match what the model was trained on), and the clip's metadata (song,
singer, band, time, features).  ``--remap-english`` builds the zero-shot
variant (Hindi phones mapped to the nearest English tokens, R0).

Batches (``PairDataset`` + ``collate``)
---------------------------------------
Like inference, every sample is ``[prompt | target]``: the prompt is the
accepted clip of the same singer id (same song + cluster) that ends closest
before the target (else the nearest one), its last ``prompt_max_s`` seconds;
when the singer has no other clip, the first 25-40 % of the target itself is
the prompt (upstream's own training default).  Note tokens are concatenated
and the target's ``mel2note`` offset, exactly as ``SoulXSinger.infer`` does.

**Control-gate dropout** re-creates the paper's MIDI/F0 gate (absent from the
released code): per sample, with ``p_score`` only note pitches are kept (F0 ->
0, i.e. score mode), with ``p_melody`` only F0 (note pitch -> 0, melody mode),
else both.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
import torch

from .adapter import IndicDataProcessor, normalise_item

FPS, HOP, SR = 50, 480, 24000
REST_CONF, UNLABELLED_CONF = 0.0, -99.0


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def _mel_encoder(device: str = "cpu"):
    """SoulX's MelSpectrogramEncoder with the released config (no model weights needed)."""
    from soulxsinger.models.modules.mel_transform import MelSpectrogramEncoder
    from soulxsinger.utils.file_utils import load_config
    from saregama.paths import SOULX_REPO
    cfg = load_config(str(SOULX_REPO / "soulxsinger" / "config" / "soulxsinger.yaml"))
    return MelSpectrogramEncoder(cfg.audio).to(device).eval()


def _floats(s, n: int, fill: float) -> np.ndarray:
    """Space-separated floats (or a list) cut/padded to ``n`` with ``fill``."""
    if s is None:
        return np.full(n, fill, np.float32)
    v = np.array([float(x) for x in (s.split() if isinstance(s, str) else s)], np.float32)
    return np.pad(v[:n], (0, max(0, n - len(v))), constant_values=fill)


@torch.no_grad()
def build_cache(jsonl: Path, wav_root: Path, phone_table: Path, out: Path, remap_english: bool = False,
                device: str = "cpu") -> Dict:
    """Cache one export split (see the module doc); returns counts."""
    proc = IndicDataProcessor(hop_size=HOP, sample_rate=SR, phoneset_path=str(phone_table), device="cpu")
    mel = _mel_encoder(device)
    clips, bad = [], []
    for line in Path(jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        n = int(item["n_frames"])
        try:
            p = proc.process(normalise_item(item, remap_english=remap_english))
            if p["mel2note"].shape[1] != n:
                raise ValueError(f"mel2note {p['mel2note'].shape[1]} != {n}")
            wav, sr = sf.read(wav_root / item["wav"], dtype="float32")
            assert sr == SR, f"{item['wav']}: {sr} Hz"
            wav = np.pad(wav[: n * HOP], (0, max(0, n * HOP - len(wav))))
            m = mel(torch.from_numpy(wav)[None].to(device))[0].cpu()            # (T, 128)
            if m.shape[0] != n:
                raise ValueError(f"mel frames {m.shape[0]} != {n}")
        except Exception as e:  # noqa: BLE001
            bad.append({"index": item.get("index"), "error": f"{type(e).__name__}: {e}"})
            continue
        clips.append({
            "index": item["index"], "song_id": item.get("song_id", ""), "singer_id": item.get("singer_id", ""),
            "band": item.get("band", "accept"), "time": item.get("time", [0, 0]),
            "features": item.get("features", {}), "text": item.get("text", ""),
            "phoneme_str": item["phoneme"], "tonic_hz": item.get("tonic_hz"), "year": item.get("song_id", "0000")[:4],
            "phoneme": p["phoneme"][0].short(), "note_pitch": p["note_pitch"][0].short(),
            "note_type": p["note_type"][0].byte(), "mel2note": p["mel2note"][0].short(),
            "f0": torch.from_numpy(_floats(item["f0"], n, 0.0)),
            "conf": torch.from_numpy(_floats(item.get("conf"), n, REST_CONF)),
            "f0_agree": torch.from_numpy(_floats(item.get("f0_agree"), n, 1.0)),
            "mel": m.half(),
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"clips": clips, "source": str(jsonl), "remap_english": remap_english}, out)
    return {"clips": len(clips), "failed": len(bad), "failures": bad[:10],
            "hours": round(sum(c["mel"].shape[0] for c in clips) / FPS / 3600, 3)}


# ---------------------------------------------------------------------------
# pairing and batching
# ---------------------------------------------------------------------------
@dataclass
class GateProbs:
    """Probabilities of the control modes during training."""
    p_score: float = 0.4
    p_melody: float = 0.4


def prompt_map(clips: Sequence[Dict], prompt_bands=("accept",)) -> List[Optional[int]]:
    """For each clip, the index of its prompt clip (same singer id, accepted, closest
    before it in time, else nearest after), or None."""
    by: Dict[str, List[int]] = {}
    for i, c in enumerate(clips):
        if c["band"] in prompt_bands and c["singer_id"]:
            by.setdefault(c["singer_id"], []).append(i)
    out: List[Optional[int]] = []
    for i, c in enumerate(clips):
        cands = [j for j in by.get(c["singer_id"], []) if j != i]
        if not cands:
            out.append(None)
            continue
        t0 = c["time"][0]
        before = [j for j in cands if clips[j]["time"][1] <= t0]
        pool = before or cands
        out.append(min(pool, key=lambda j: abs(t0 - clips[j]["time"][1])))
    return out


class PairDataset(torch.utils.data.Dataset):
    """[prompt | target] samples over the cached clips of chosen bands."""

    def __init__(self, cache: Path, bands=("accept",), prompt_max_s: float = 6.0, max_total_s: float = 22.0,
                 gate: Optional[GateProbs] = GateProbs(), seed: int = 0):
        """Load the cache and the prompt map; ``gate=None`` keeps both controls (eval)."""
        data = torch.load(cache)
        self.all = data["clips"]
        self.idx = [i for i, c in enumerate(self.all) if c["band"] in bands]
        self.prompt = prompt_map(self.all)
        self.prompt_max = int(prompt_max_s * FPS)
        self.max_total = int(max_total_s * FPS)
        self.gate = gate
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        """Number of targets."""
        return len(self.idx)

    def frames(self, k: int) -> int:
        """Target length in frames (for length-bucketed batching)."""
        return int(self.all[self.idx[k]]["mel"].shape[0])

    def __getitem__(self, k: int) -> Dict:
        """One sample: concatenated tokens / frames, prompt mask and per-frame info."""
        t = self.all[self.idx[k]]
        j = self.prompt[self.idx[k]]
        T = t["mel"].shape[0]
        if j is not None:
            p = self.all[j]
            L = min(p["mel"].shape[0], self.prompt_max, max(self.max_total - T, FPS))
            a = p["mel"].shape[0] - L
            pm2n = p["mel2note"][a:].long()
            parts = [(p, slice(a, None), pm2n), (t, slice(0, None), t["mel2note"].long() + len(p["phoneme"]))]
            tokens = torch.cat([p["phoneme"], t["phoneme"]]).long()
            pitch = torch.cat([p["note_pitch"], t["note_pitch"]]).long()
            ntype = torch.cat([p["note_type"], t["note_type"]]).long()
            n_prompt = L
        else:                                   # self-prompt: the first 25-40 % of the target
            n_prompt = int(T * self.rng.uniform(0.25, 0.4))
            parts = [(t, slice(0, None), t["mel2note"].long())]
            tokens, pitch, ntype = t["phoneme"].long(), t["note_pitch"].long(), t["note_type"].long()
        cat = lambda key: torch.cat([c[key][s] for c, s, _ in parts])          # noqa: E731
        mel2note = torch.cat([m for _, _, m in parts])
        f0 = cat("f0").float()
        mode = "both"
        if self.gate is not None:
            r = self.rng.random()
            mode = "score" if r < self.gate.p_score else ("melody" if r < self.gate.p_score + self.gate.p_melody else "both")
        if mode == "score":
            f0 = torch.zeros_like(f0)
        elif mode == "melody":
            pitch = torch.zeros_like(pitch)
        n = mel2note.shape[0]
        is_prompt = torch.zeros(n)
        is_prompt[:n_prompt] = 1
        return {"tokens": tokens, "note_pitch": pitch, "note_type": ntype, "mel2note": mel2note, "f0": f0,
                "mel": cat("mel").float(), "conf": cat("conf").float(), "f0_agree": cat("f0_agree").float(),
                "is_prompt": is_prompt, "band": t["band"], "mode": mode, "index": t["index"]}


def collate(batch: Sequence[Dict]) -> Dict:
    """Pad a list of samples (tokens with <PAD>=0 / type 1; frames with zeros, x_mask 0)."""
    B = len(batch)
    nt = max(len(b["tokens"]) for b in batch)
    nf = max(len(b["mel2note"]) for b in batch)
    out = {"tokens": torch.zeros(B, nt, dtype=torch.long), "note_pitch": torch.zeros(B, nt, dtype=torch.long),
           "note_type": torch.ones(B, nt, dtype=torch.long), "mel2note": torch.zeros(B, nf, dtype=torch.long),
           "f0": torch.zeros(B, nf), "mel": torch.zeros(B, nf, 128), "conf": torch.zeros(B, nf),
           "f0_agree": torch.ones(B, nf), "is_prompt": torch.zeros(B, nf), "x_mask": torch.zeros(B, nf)}
    for i, b in enumerate(batch):
        k, n = len(b["tokens"]), len(b["mel2note"])
        for key in ("tokens", "note_pitch", "note_type"):
            out[key][i, :k] = b[key]
        for key in ("mel2note", "f0", "mel", "conf", "f0_agree", "is_prompt"):
            out[key][i, :n] = b[key]
        out["x_mask"][i, :n] = 1
    out["band"] = [b["band"] for b in batch]
    out["mode"] = [b["mode"] for b in batch]
    out["index"] = [b["index"] for b in batch]
    return out


class FrameBudgetSampler(torch.utils.data.Sampler):
    """Batches of similar-length targets whose padded size stays under ``max_frames`` (shuffled)."""

    def __init__(self, ds: PairDataset, max_frames: int, seed: int = 0, shuffle: bool = True):
        """Pre-sort by length; batches are rebuilt every epoch."""
        self.ds, self.max_frames, self.seed, self.shuffle, self.epoch = ds, max_frames, seed, shuffle, 0
        self.lengths = [ds.frames(k) + ds.prompt_max for k in range(len(ds))]

    def _batches(self) -> List[List[int]]:
        """Length-sorted chunks under the frame budget, in a (seeded) random order."""
        rng = random.Random(self.seed + self.epoch)
        order = sorted(range(len(self.ds)), key=lambda k: self.lengths[k] + rng.random())
        out, cur, longest = [], [], 0
        for k in order:
            longest2 = max(longest, self.lengths[k])
            if cur and longest2 * (len(cur) + 1) > self.max_frames:
                out.append(cur)
                cur, longest2 = [], self.lengths[k]
            cur.append(k)
            longest = longest2
        if cur:
            out.append(cur)
        if self.shuffle:
            rng.shuffle(out)
        return out

    def __iter__(self):
        """Yield the batches of the current epoch."""
        bs = self._batches()
        self.epoch += 1
        return iter(bs)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return len(self._batches())

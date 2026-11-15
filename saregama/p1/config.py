"""P1 configuration: defaults here, overrides from ``<work>/p1.toml``.

The config is a nested dict (sections below).  Each stage declares which
sections it reads; its config hash covers only those, so tuning, say, the QC
thresholds re-runs QC but not the GPU stages.
"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, Dict

from ..paths import PROJECT

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "paths": {
        "audio_dir": str(PROJECT / "music_dataset"),
        "work_dir": str(PROJECT / "work"),
        "export_dir": str(PROJECT / "export"),
    },
    "separation": {"backend": "soulx", "sep_model": "vocals", "dereverb": True, "lead_model": "karaoke"},
    "f0": {"backends": ["rmvpe", "fcpe", "yin"], "fmin": 60.0, "fmax": 1400.0, "agree_cents": 50.0},
    "voices": {"min_single_s": 1.0, "residual_db": -12.0, "second_peak": 0.35, "cluster_threshold": 0.55},
    "asr": {"primary": "Qwen/Qwen3-ASR-1.7B", "secondary": "openai/whisper-large-v3", "language": "hi",
            "min_gap_s": 0.4, "min_phrase_s": 1.0, "max_phrase_s": 25.0, "pad_s": 0.3,
            "max_chars_per_s": 22.0, "batch_size": 16, "whisper_beams": 1},
    "lyrics": {"min_version_coverage": 0.4, "match_threshold": 0.75},
    "align": {"window_pad_s": 0.3},
    "qc": {},
    "device": {"device": "cuda"},
}


def load_p1_config(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Defaults updated (per section) from the TOML at ``path`` if it exists;
    default path: ``<work_dir>/p1.toml``."""
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else Path(cfg["paths"]["work_dir"]) / "p1.toml"
    if p.exists():
        with open(p, "rb") as f:
            data = tomllib.load(f)
        for sec, vals in data.items():
            if sec not in cfg:
                raise KeyError(f"unknown P1 config section {sec!r} in {p}")
            for k in vals:
                if k not in cfg[sec]:
                    raise KeyError(f"unknown P1 config key {sec}.{k} in {p}")
            cfg[sec].update(vals)
    return cfg

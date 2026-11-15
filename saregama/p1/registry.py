"""All P1 stages in pipeline order (PLAN.md §4) and the named stage groups the
workers run.  Bump a stage's ``version`` whenever its code changes its output:
that re-runs it (and its dependants) everywhere and nothing else.
"""
from __future__ import annotations

from typing import Dict, List

from .worker import Stage


def all_stages() -> Dict[str, Stage]:
    """name -> Stage, in pipeline order."""
    from . import clean
    stages: List[Stage] = [
        Stage("inventory", 1, clean.inventory, sections=(), depends=()),
        Stage("fingerprint", 1, clean.fingerprint, sections=(), depends=("inventory",)),
    ]
    try:                                    # stages added as they are implemented
        from . import gpu_stages
        stages += gpu_stages.STAGES
    except ImportError:
        pass
    try:
        from . import text_stages
        stages += text_stages.STAGES
    except ImportError:
        pass
    return {s.name: s for s in stages}


GROUPS = {
    "clean": ["inventory", "fingerprint"],
    "gpu": ["separate", "f0", "voices", "asr", "emissions"],
    "text": ["singers", "identify", "reconcile", "align", "vocables", "notes_qc"],
}


def resolve(spec: str) -> List[str]:
    """Stage names from a comma list of stage or group names (order kept)."""
    names: List[str] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        for n in GROUPS.get(tok, [tok]):
            if n not in names:
                names.append(n)
    return names

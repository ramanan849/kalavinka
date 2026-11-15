"""P1: large-scale annotation of Hindi / Hinglish film songs for SoulX-Singer.

Resumable, per-song, per-stage pipeline run by Slurm workers (see the project's
PLAN.md): ``state`` (atomic files, stage markers), ``manifest`` (the song list),
``worker`` (claim loop) and ``stages`` (the stage implementations).
"""

"""Kalavinka: Hindi film-song singing voice synthesis on SoulX-Singer.

Built on the ``saregama`` annotation package (never modified; wrapped):

* ``kalavinka.annot`` -- finishing the P1 annotation: gold-set threshold
  calibration and the v1 dataset export with per-frame confidence.
* ``kalavinka.sx``    -- stage 2 renderer: SoulX-Singer fine-tuning with
  articulatory-composed Hindi phone embeddings and confidence-aware flow matching.
* ``kalavinka.ir``    -- stage 1: the expressive-contour IR model (note scaffold
  -> expressive F0) that SoulX renders in melody mode.
* ``kalavinka.eval``  -- held-out rendering and metrics.
* ``kalavinka.ops``   -- scripts to run all of it on a local multi-GPU box.

See ``kalavinka/README.md`` for every command.
"""

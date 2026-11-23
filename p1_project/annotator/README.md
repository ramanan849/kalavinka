# P1 gold annotator (local web app)

Rushikesh's browser-based editor for correcting the P1 gold pack (the pipeline's
Praat TextGrids: `words`, `syllables`, `vocables`, `singers` tiers). It replaced
Praat/Sonic Visualiser for the gold set because it is faster for word and
syllable boundary work (waveform + spectrogram, click-to-play segments, a
Devanagari keyboard, autosave, listened-coverage tracking).

Committed here unmodified (from the delivered `gold.zip`, 23 Nov 2026) so it is
versioned. Only the Python standard library is needed.

## Run

The server expects to live **inside** the gold folder, next to `pack/`:

```
gold/
  annotator/          <- copy this folder here
  pack/<song_id>/{vocals.wav, mix.wav, meta.json, <song_id>.TextGrid}
  corrected/          <- saved TextGrids land here
  .annotator/         <- drafts, history and state.json (created on first run)
```

```
cp -r p1_project/annotator <PROJECT>/gold/annotator
python <PROJECT>/gold/annotator/server.py          # opens http://127.0.0.1:8624
# Windows: double-click "Start Annotator.bat"; port override: ANNOTATOR_PORT=...
```

`.annotator/state.json` records per song the listened coverage (`listened_pct`),
time spent (`seconds`) and an issue count; `kalavinka/annot/calibrate.py` uses
it to tell fully reviewed gold songs from unreviewed ones.

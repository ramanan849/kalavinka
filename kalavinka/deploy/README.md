# kalavinka.deploy: shareable inference bundles

Packs the final Kalavinka model into two zips a friend can run without this repo or the dataset:

| Zip | For | Extras |
|---|---|---|
| `kalavinka_colab.zip` | Google Colab | `Kalavinka_Colab.ipynb` (step-by-step notebook) |
| `kalavinka_mac.zip` | a Mac (or any PC) | `setup_mac.sh` (one-time setup), `sing` (launcher), `MAC.md` |

Both contain the same core (see `build_bundle.py`'s docstring for the layout): our weights (V4-L = run `L4_v2`,
and the stage-1 contour model C-flow = run `I2_v1`), copies of the project modules inference needs, SoulX-Singer's
code, and a library of 17 held-out test lines (tunes) with 24 voice clips (all from held-out val/test songs) and their
profiles (`data/voice_profiles.json`). SoulX's base `model.pt` (2.7 GB) is
**not** inside; it is downloaded from Hugging Face on first use.

| File | What |
|---|---|
| `singer.py` | `KalavinkaSinger`: load once, then `sing(lyrics, tune, voice, mode)` / `resing(tune, mode)`; device auto (cuda > mps > cpu) with an MPS -> CPU fallback (only for MPS errors; CPU steps); one render at a time (a lock) |
| `seamless.py` | **seamless singing**: every tab writes one long score (`Timeline`) and `render` cuts it into model-sized pieces at rests (or, in continuous singing, with an overlap inside a held note: phase-aligned equal-power crossfade + gain match), placed sample-exact, so silence appears only where the score has a rest. `SeamlessSinger` (Sing tab: real gaps between adjacent clips of a song, `line_gap` elsewhere, SoulX's own transposition rule per line), `SeamlessSargamSinger`; `join="separate"` = the old line-by-line way |
| `cli.py` | `python -m kalavinka_app sing/resing/alankar/sargam/compose/compose-prompt/history/note/title/tag/star/delete/restore/log/rebuild-history/voices/voice-edit/prompt/tunes/demo/ui/fetch` inside a bundle; `--model kalavinka\|base\|both`, `--join seamless\|separate`, `--line-gap`, `--fresh`; `--steps/--seed/--fresh/--play/--outputs` before or after the command |
| `app.py` | Gradio page (tabs: new lyrics; compare with the real singer; alankars/sargam; compose; voices; history), model choice kalavinka / base / both, join seamless / separate; renders queued one at a time |
| `voices.py`, `voices_ui.py`, `voices_build.py`, `voice_profiles.json`, `voice_credits.json` | voice profiles: estimated gender (credited singers of the source song; pitch for duets), register within the gender, era, Sa, sung range, credited singer; filters + profile card + Listen sample on every tab; a Voices tab to browse and edit (edits in `outputs/voice_profiles.local.json`). `voices_build` makes the spec once (credits looked up on the web, sources in `voice_credits.json`) |
| `history_ui.py` | the History tab: search / filter, play, label (title, tags, notes, star), delete / restore, export zip, event log, rebuild, **Open in its tab** |
| `compose_matrix.py` | batch singing: every composition x every voice (in priority order, e.g. `17,9,all`) x both models, one worker per GPU (`--shard k/n`), resumable, with a shared `progress.log` (done/total, elapsed, ETA) and `progress.jsonl` (per take: seconds, render time, how closely it followed the curve) |
| `sargam.py` | sargam notation + JSON score (``LLM_PROMPT``), 7 built-in alankars, ``SargamSinger``, note-by-note pitch check (pYIN) |
| `compose.py` | compositions (JSON): swaras/cents, lengths, lyrics, ornaments (meend, gamak, kan, murki, khatka, zamzama, andolan, vibrato, glides, custom), raags/taals, hand-drawn strokes -> 50 fps F0 + SoulX items (melody mode); ``ComposeSinger``, follow check, ``COMPOSE_PROMPT`` |
| `compose_ui.py`, `static/f0_editor.{js,css}` | the web page's Compose tab: Start from (only the chosen source's controls), voice picker, canvas F0 editor (drag pitch/length, type a lyric line onto a phrase, in-place syllable editing with Tab, red underline for syllables the model cannot sing, lyrics strip, ornament panel, pencil/eraser, undo, in-browser preview tone), table view, autosave + restore, sing + sung-pitch overlay |
| `history.py` | takes (wav + full json sidecar, written atomically under a lock), ``history.jsonl`` (append-only), ``annotations.jsonl`` (labels, append-only), ``log.jsonl`` (event log), ``trash/``; request cache (same request = saved take reused); ``rebuild()`` from the sidecars |
| `build_bundle.py` | chooses the tunes and voices, vendors the code, cleans the checkpoints, writes both zips |
| `bundle_main.py` | becomes `kalavinka_app/__main__.py` in the bundle |
| `bundle_files/` | README, NOTICE, requirements, notebook, Mac scripts copied into the bundles |

Auto tonic: Sa is the voice clip's tonic, moved into the octave that suits the voice (``sargam.fold_tonic``; the
dataset's tonic is sometimes an octave off for the singer).

Models: `kalavinka` (V4-L) and `base` (released SoulX-Singer, Hindi phones mapped to the nearest English ones = our
zero-shot V0; loaded on first use). Pitch modes: `score` (the model's own pitch, default) and `expressive` (two-stage: C-flow at temperature 0.85 draws
the F0, SoulX sings it in melody mode). `resing` also offers `real-pitch` (the real singer's F0).

## Alankar benchmark

```
cd <bundle> && for v in 1 4 5 7 10; do python -m kalavinka_app alankar all --model both --voice $v; done
cd saregama624 && ../.venv-p1/bin/python -m kalavinka.eval.alankar_eval --history <bundle>/outputs/history.jsonl \
    --out ../runs/eval_v1/alankars          # pitch + Whisper syllable CER, paired bootstrap; alankar_bench.md
```

## Build

```
cd saregama624
../.venv-sx/bin/python -m kalavinka.deploy.build_bundle --out ../bundles      # or: kalavinka/ops/pack_bundles.sh
```

## Tests

```
../.venv-sx/bin/python -m pytest -q kalavinka/tests/test_deploy.py     # zip checks run when ../bundles exists
../.venv-sx/bin/python -m pytest -q kalavinka/tests/test_seamless.py kalavinka/tests/test_fixes_deploy.py \
    kalavinka/tests/test_history2.py kalavinka/tests/test_voices.py kalavinka/tests/test_g2p_fixes.py
```

Seamless joining was checked on a GPU on 26 Nov 2026 (both models): a 4-line song has no silence at the line joins
(separate: 0.3 s at each); a 35 s rest-free composition is cut by two overlaps inside held notes with no audible
seam or level step and follows the curve as closely as separate renders (94 % within 50 c); over all 7 alankars x 2
seeds (408 notes) the pitch is as accurate (Kalavinka 92 % vs 93 % within 50 c; base 99 % vs 97 %).

Checked on 25 Nov 2026 in fresh environments: Python 3.12 + newest PyTorch on a GPU (the notebook, run headless),
and Python 3.10 + CPU-only PyTorch (the Mac scripts on Linux). Not tested on a real Mac GPU (MPS).

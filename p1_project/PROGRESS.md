# P1 progress log (read this first in every session)

**Rules for this file:** keep it short. After every step, update "Current state" and
"Next actions", and add one line to the log. Details live in `PLAN.md` (sections as §n)
and in `ops/jobs.tsv` (Slurm jobs). Cluster rules: `~/.claude/CLAUDE.md` (mandatory: no
python on the login node; `-p X --qos=X`).

## Current state (2026-11-15)
- **Phase:** C1 pilot done; **gold pack ready, waiting for the human (H3)**. Full GPU run in progress through dependency chains (a40 ×5, l40 ×2+2, dgx ×3 jobs of 60 min, queued 2026-11-15 06:20).
- **Code:** `P1/saregama624`, working tree **uncommitted**:
  - new `stages/asr.py`, `lyrics_eval.py`, `tests/test_asr.py`;
  - edits to pipeline, align, g2p, cli, config, separate, paths, README, `configs/default.toml`;
  - 63 tests pass.
- **Env:** `P1/.venv` (py3.12, torch 2.8 cu128, transformers 4.57.6, qwen-asr 0.0.6), GPU nodes only.
- **Models cached:**
  - `~/.cache/huggingface`: Qwen3-ASR-1.7B, whisper-large-v3, Oriserve Hinglish, SoulX-Preprocess;
  - `~/.cache/torch/hub/checkpoints/model.pt` (MMS_FA);
  - `P1/models/melband-vocals-kj` (full-vocals separator).
- **Data:**
  - `P1/music_dataset/`: 1,074 files (stays in /home);
  - `P1/downloaded_hindi_songs.zip` is truncated (delete at A1).
- **Eval:** `P1/eval/` (`run_eval.py`, `run_eval.sbatch`, `results_vocals/summary.md`).

## Key findings so far (don't re-derive)
1. **ASR:** Qwen3-ASR-1.7B mean WER 0.30, Whisper-large-v3 0.33, Oriserve Hinglish 0.44
   (3 songs, loose-phonetic scoring vs LRCLIB).
2. **Alignment:** ASR timing alone gives 0.3–1.9 s line-onset error; with MMS_FA 0.06–0.12 s
   (after a tempo-drift fit), about the same as aligning the reference lyrics.
3. **Separation:** the karaoke separator silences duet and chorus singing (WER 0.85–1.0);
   use `sep_model="vocals"`.
4. **Mislabeled file:** "2013 - Dilliwaali Girlfriend [PdjC8e7GFWg]" is actually *Subhanallah*.
   Expect more mislabels (→ S7).
5. **Version mismatch:** YouTube versions differ from LRC releases (edits, 1–4% tempo drift).
   Kesariya's file is a 2:52 video edit with dialogue.
6. **Phrase-window alignment** equals global alignment when the ASR is good, and is more
   robust when the ASR is noisy.

## Decisions taken (with PLAN refs)
- Hybrid lyrics: ASR = what was sung and in what order; web = spelling; audio decides ties (§5).
- Devanagari canonical text, English words in Latin, romanization internal only (§5.1).
- Keep single-voice stretches only; diarize, don't separate overlaps (§7).
- Phone refinement additive and gated (§6). Precision over coverage (§0).
- Workers + atomic claims + per-stage `stages.json` for resumability (§3).
- No SoulX training now; DataProcessor compatibility check only (§0).

## Next actions
| Who | Where | Action |
|---|---|---|
| **Human** | PC | **H3 gold set**: copy `P1/gold/` to the PC (`scp -r 23b1529@10.195.100.101:~/projects/P1/gold ./p1_gold`; WinSCP / MobaXterm on Windows), follow `gold/README.txt` in Praat (20 windows × 75 s, about 15–25 min each), put the corrected TextGrids in `gold/corrected/` and copy them back to Prajna. |
| Claude | login | GPU chains: `squeue -u $USER`; when they end, resubmit workers (`ops/submit.sh gpu 60 gpu-allN -- p1-run --stages gpu --budget-min 50`, chains via `P1_DEP=afterany:<id>`) until `p1-status` shows asr/emissions done for all songs |
| Claude | login | finish `ops/fetch.sh work/raw/fetch_retry.tsv` (resumable; log `ops/logs/fetch_retry.log`), then `p1-refpool`; after all GPU stages: `p1-fetchlist --kind retry` again → fetch → refpool → `p1-run --stages text` (CPU, about 2 s/song) |
| Claude | code | C3 `p1-calibrate` (thresholds from gold: single-voice cues, clip bands, singer clustering); H5 review pack; H6 cross-song singer linking; E1 SoulX phone-set files + real DataProcessor check |

## Log (one line per step)
- 2026-11-14: cloned saregama624; built env; added ASR stage, Latin G2P, phrase-window alignment, CLI (`asr`, `songs-from-folder`).
- 2026-11-15: benchmark jobs 385698–385825 (a40/l40, about 69 GPU-min); results in `eval/results_vocals/`.
- 2026-11-15: wrote `~/.claude/CLAUDE.md` (cluster rules), PLAN.md, PROGRESS.md.
- 2026-11-15: checked /scratch: root-only, not writable; all data stays in /home (no quota).
- 2026-11-15: H1 approved (plan + usage rights). A1–A4 done: removed wheels/ + broken zip; git branch `p1-annotation` (baseline commit 6d795e9, local identity 23b1529); `ops/submit.sh` + `ops/run.sbatch` + `ops/jobs.tsv`; installed speechbrain, cmudict, ECAPA model, `~/bin/fpcalc`.
- 2026-11-15: ops note: the `debug` partition is often blocked (AssocGrpSubmitJobsLimit), so `submit.sh` falls back to a40/l40 without a GPU for CPU jobs; a40 CPU jobs start in seconds.
- 2026-11-15: B1 done (state/claims/workers, `p1-init/run/status/dedupe`; tests pass). B2 done: manifest 1,073 songs (`work/songs.tsv`); inventory + fingerprint for all, 0 failures (3 CPU workers, about 9 min); `work/dupes.json`: 23 groups, 26 duplicates skipped from now on, **13 groups with conflicting titles** (same audio under 2 names = mislabels, e.g. Bole Chudiyan = "Malang", Naacho Naacho = "Chammak Challo"), which S7 must resolve. Splits must group by identified song (2 Tum Hi Ho uploads differ in audio but share the performance).
- 2026-11-15: read the Prajna manual (`ops/prajna_manual.txt`); limits recorded in `~/.claude/CLAUDE.md`.
- 2026-11-15: GPU stages verified on 4 songs (job 386417, 0 failures; about 50–110 s/song warm; 13 MB/song). Node failure seen (386378 NODE_FAIL cn45-l40), so `submit.sh` now tries a40 first.
- 2026-11-15: design fixes: ASR no longer depends on `voices` (singer/overlap per phrase now computed in reconcile); dependency-stamp staleness (re-running a stage invalidates its dependants); singer clustering adds a pitch-register term (voices v2).
- 2026-11-15: LRCLIB search doesn't cover lyric text, so identification matches ASR text against a pool of all fetched lyrics (`work/refpool.json`).
- 2026-11-15: pilot list `work/pilot.txt` (44 songs: 8 per decade + 4 benchmark). GPU jobs 386494–97 (2 pilot, 2 all). Login fetch: `ops/fetch.sh` split per host (lrclib / oembed) for speed.
- 2026-11-15: whole pipeline runs end to end on the pilot (43 songs, 0 text-stage failures). Fixes on the way: Whisper greedy (ASR 35–190 s → 4–22 s/song), m4a decoded with ffmpeg first, CPU `singers` stage (clusters cached embeddings; 1–5 singers/song instead of up to 42), unit-based clip planning (1 → 297 accepted clips = 31.5 min; main reject reason: overlap). Refpool: 8,845 refs from 761 songs; pilot identity 30 claimed / 1 relabeled / 12 unidentified → second lookup round (`fetch_retry.tsv`, 458 URLs, 288 songs: title spelling variants + oEmbed titles).
- 2026-11-15: GPU workers 386545/386546/386635 on all songs (a40 + l40); about 1–1.5 min/song warm. Jobs run 60 min with a 50 min budget; resubmit when they end.
- 2026-11-15: second lookup round on the pilot: identity 36 claimed / 1 relabeled (Naacho Naacho file = Chammak Challo) / 6 unidentified; pilot: 304 accepted clips (32 min) + 27 review. Gold pack: 20 songs (`work/gold_songs.txt`, 4 per decade, solo to overlap-heavy) → `gold/pack/` (138 MB). m4a retry ok (386636).
- 2026-11-15 ~09:45: chains 1 done, 551/1047 songs GPU-complete, 0 failures; submitted round 2 chains 387447-387458 (a40x2, l40, dgx; 3x60min each). Retry fetch 456/458.

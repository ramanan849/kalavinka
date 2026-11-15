# P1 implementation plan: Hindi / Hinglish singing-voice annotation for SoulX-Singer

> Living document. **Read `PROGRESS.md` first.** It says where we are and what comes
> next; it points to the sections of this file that you need. Cluster rules are in
> `~/.claude/CLAUDE.md` and are mandatory.

## 0. Goal and quality bar

From about 1,070 Bollywood songs (`music_dataset/`, YouTube audio), build a
SoulX-Singer training set: **2–16 s clips of one known singer**, each with:
- **words that were actually sung** (repeats, partial lines, ad-libs, aakars included);
- **syllable / note timing** on SoulX's 50 fps grid, with the **phone sequence** per note;
- **F0 contour**, note pitch and note type;
- **per-clip confidence**, with a singer id and singer-disjoint splits.

**Principle: precision over coverage.** One wrong label teaches the model a wrong
pronunciation; one missing clip costs nothing (we have about 70 h of music). Every stage
can *discard*, and only confident clips are exported.

**Not in scope now: SoulX training.** SoulX ships no training code (repo README), so a
fine-tune means writing a trainer, which is a separate project. Instead we run the
**SoulX `DataProcessor` compatibility check**: its loader reads our export without errors
and produces the expected frame counts. That check is necessary and cheap. The real
"does it sing well" test comes after this plan.

## 1. Where things run (rules from `~/.claude/CLAUDE.md`)

| Where | What | Why |
|---|---|---|
| **Login node** | editing, git, `uv pip install`, `curl`/`wget` downloads (models, LRCLIB JSON, YouTube oEmbed metadata), `jq` on downloaded JSON, `rsync`, `sbatch`/`squeue`/`sacct`/`scancel` | Allowed list. **No python of any kind**, including tests, `yt-dlp` (which is python) and quick imports. |
| **Slurm GPU jobs** (a40 / l40, 1 GPU) | separation, F0, singer embeddings, ASR ×2, MMS emissions | Need a GPU; compute nodes are offline, so every model is pre-downloaded. |
| **Slurm CPU jobs** (`-p debug --qos=debug`, ≤30 min, no `--gres`) | tests, parsing fetched JSON, fingerprinting, reconciliation, alignment from cached emissions, QC, export, reports | Python is compute, so it goes through Slurm. CPU-only work requests no GPU. |
| **Your PC** | listening, gold annotation and review in Sonic Visualiser, backup of exports | Needs ears and a GUI; an off-cluster copy protects the final dataset. |

**Internet split.** Compute nodes have no internet. Any step that needs the web is cut in
two: **fetch raw files on the login node with `curl`, then parse in a Slurm job.** The
pipeline writes "fetch lists" (URLs) that the login-node script downloads.

## 2. Storage layout

| Path | Contents | Size |
|---|---|---|
Everything lives in **/home** (checked 2026-11-15: `/scratch` is a root-only system folder
we can't write to; /home has no quota on this account and is 1.3 PB with 365 TB free).

| Path | Contents | Size |
|---|---|---|
| `P1/saregama624/` | code (git) | small |
| `P1/.venv/`, `~/.cache/{huggingface,torch}` | environment, model weights | about 25 GB (exists) |
| `P1/{PLAN,PROGRESS}.md`, `P1/ops/` | plan, log, job scripts, small reports | small |
| `P1/music_dataset/` | the 1,074 source files (stay in place) | 5.3 GB |
| `P1/work/songs/<id>/` | per-song stems (**FLAC 16-bit**), arrays, JSON | about 15–20 MB/song, about 20 GB |
| `P1/export/` | SoulX clips | about 10–20 GB |

Why FLAC anyway: /home is a shared parallel filesystem. Smaller files mean less I/O load
for everyone and faster reads in our own jobs; it costs nothing in quality (lossless).

Housekeeping (step A1, optional): delete `P1/wheels/` (3.7 GB, already installed) and the
broken `downloaded_hindi_songs.zip` (3.8 GB, truncated). A copy of the dataset on your PC
is still a good backup but is no longer a prerequisite.

## 3. Robustness: how work survives crashes and lost sessions

1. **Per-song, per-stage state.**
   - Each stage writes its outputs, then a marker in `work/songs/<id>/stages.json`:
     `{stage: {done, version, config_hash, seconds, finished_at}}`.
   - Writes are **atomic** (write to `*.tmp`, then `os.replace`), so a kill can't leave a
     half-written file that looks finished.
   - A stage reruns only if its marker is missing, its `version` or `config_hash` changed,
     or `--force` is given.
   - *Reason:* any interruption costs at most the one song in progress, and changing one
     stage's code reruns only that stage and the ones after it.
2. **Workers instead of big job arrays.**
   - QoS limits are 6 queued / 3 running on a40, 5 / 4 on l40, and array tasks count
     toward the submit limit. So we submit **at most 3–4 worker jobs**.
   - Each worker loops: claim the next song that needs its stage, process it, repeat.
   - **Claim** = atomic `mkdir claims/<stage>/<song_id>` holding the job id.
   - **Stale claims** (that job is no longer in `squeue`) are released by the next worker.
   - A worker stops taking new songs 15 min before its `--time` (it reads
     `SLURM_JOB_END_TIME`) and exits cleanly.
   - *Reason:* resubmitting is always safe, no song is done twice, the queue stays small
     (good citizenship), and backfill likes short, predictable jobs.
3. **Per-song errors don't stop a worker.** The traceback goes to
   `work/songs/<id>/errors/<stage>.txt` and the song is marked `failed:<stage>`. After 2
   failures a song is parked rather than retried forever.
4. **One status command** (`saregama p1-status`, a CPU job) builds `P1/ops/status.tsv`
   (song × stage) and a summary that is pasted into `PROGRESS.md`.
5. **Job ledger** `P1/ops/jobs.tsv`: job id, partition, step, submitted, state, notes.
   Updated at each submit and check, so a new session knows what is in flight.
6. **Everything is versioned.**
   - Each run records the git commit, config hash and model ids in `stages.json`.
   - Code changes are committed on a branch (`p1-annotation`) before any batch run.
   - *Reason:* a new session can tell exactly which code produced which files.

## 4. Final per-song architecture (stages in order)

| # | Stage | Where | What it does | Key outputs |
|---|---|---|---|---|
| S0 | `inventory` | CPU job | durations (ffprobe), sha256, decode check | `inventory.tsv` |
| S1 | `meta-fetch` | **login** (curl) | YouTube oEmbed (real title, channel); LRCLIB search by claimed title and artist | `raw/meta/*.json`, `raw/lrclib/*.json` |
| S2 | `dedupe` | CPU job | Chromaprint fingerprints (static `fpcalc` binary); clusters of duplicate uploads; keep the best copy | `dupes.tsv` |
| S3 | `separate` | GPU | full-vocals Mel-RoFormer **and** karaoke model (both), de-reverb | `vocals.flac`, `lead.flac` |
| S4 | `f0` | GPU | RMVPE + FCPE + YIN ensemble on vocals; RMVPE second-peak salience | `f0.npy`, `poly.npy` |
| S5 | `voices` | GPU | single-voice mask (full vs karaoke residual + second peak + ensemble disagreement); singer embedding per single-voice phrase (ECAPA); clusters within the song | `voices.json` |
| S6 | `asr` | GPU | Qwen3-ASR-1.7B (primary) and Whisper-large-v3 per phrase; phrase plan from the single-voice-aware voicing mask | `asr.json` (both systems) |
| S7 | `identify` | CPU + **login** fetch | score the ASR text against the claimed title's lyrics; if coverage is low, write ASR-line queries, fetch them on the login node, re-score; relabel / version-variant / unidentified | `identity.json` |
| S8 | `reconcile` | CPU | choose a reference version; local alignment per phrase; word decisions (§5) | `lyrics.final.json` with per-word source and confidence |
| S9 | `align` | GPU (emissions) + CPU | cache MMS_FA emissions; acoustic choice between ASR and reference text; phrase-window alignment with optional words | `mms.npz`, `syllables.json` |
| S10 | `vocables` | CPU | voiced audio no word explains: melisma (extends a syllable) or a standalone vocable (आ / ओ / हम्म, vowel from the emissions) | updated `syllables.json` |
| S11 | `phones` | CPU | phone boundaries **inside** syllables (gated, §6); fallback to the rule-based split | `phone_spans` (optional layer) |
| S12 | `notes-qc` | CPU | notes, per-clip confidence, clip plan (single voice, one singer id) | `annotation.json`, `clips.json` |
| S13 | `export` | CPU | SoulX format, singer-disjoint splits, DataProcessor check | `P1/export/` |

**Reasons for the order:**
- **Cleaning first (S0–S2):** don't spend GPU time on duplicates or broken files.
- **Both separators (S3)** give a cheap overlap signal: where they agree, one voice is
  singing.
- **Voices before ASR (S5 → S6):** ASR phrases shouldn't straddle a change of singer.
- **Identity before reconciliation (S7 → S8):** the reference has to be the right song.
- **Every GPU stage caches its output:** all later decisions (S8–S13) are CPU-only and
  re-runnable in seconds when thresholds change, with no GPU needed.

## 5. Lyrics: hybrid decision rules (S7–S9)

**Sources.** LRCLIB (synced lyrics plus duration) is primary, fetched by title and artist
and, for identification, by ASR text. Lyrics sites are an optional later addition (only if
coverage is poor). We fetch only on the login node, rate-limited to 1 request per second,
with a generic user agent.

1. **Canonical text.**
   - Devanagari for Hindi words, Latin for English words, tagged per word (English: Latin
     script, or Devanagari that back-transliterates to an English dictionary word).
   - Phones: Devanagari G2P (the repo's rules) for Hindi, English G2P (`g2p_en` / CMUdict
     to SoulX `en_*` phones) for English.
   - Romanization is internal only: MMS_FA tokens and matching, always derived from phones.
   - *Reason:* Devanagari keeps dental/retroflex, vowel-length, nasalization and nukta
     distinctions that romanized text loses, and wrong phones are the costliest label error.
2. **Choosing the version.**
   - Score each candidate against the whole ASR transcript on the loose phonetic form:
     share of ASR words matched, WER and duration difference.
   - Best candidate above 0.4 coverage: it is the reference.
   - Below that: `no_reference`, the song is ASR-only, and S7 identification is triggered.
3. **Matching per phrase.** Each ASR phrase gets its own local alignment (Smith–Waterman)
   against the whole reference, with substitution costs from phonetic similarity.
   - *Reason:* repeats, partial lines and changed order all map correctly, because no
     line-order assumption is made.
4. **Decisions per word:**

| Case | Decision |
|---|---|
| ASR word ≈ reference word | reference spelling, at the ASR's position |
| differs, Whisper agrees with the reference | reference |
| differs, both ASRs agree against the reference | ASR (sung differently) |
| differs, no agreement | S9 acoustic choice, else low confidence |
| ASR-only word, both ASRs have it | keep (ad-lib) |
| ASR-only word, one ASR only | drop or low confidence |
| reference word missed by ASR, inside the matched stretch | optional word in alignment; kept only if the aligner uses it with good score |
| reference line never matched | ignored |
| no pitch in the phrase (dialogue) | removed |

5. **Acoustic choice (S9).** Where text candidates disagree, force-align each against the
   cached MMS frame scores and keep the version with the better average score per frame.
   The audio decides, not either text source.
6. **Aakar (S10).** Voiced audio not covered by words becomes:
   - **melisma**: continues the previous syllable's vowel, F0 continuous → slur notes (SoulX type 3);
   - **standalone vocable**: a separate vowel → its own syllable (आ, ओ, ई, हम्म), vowel
     from the MMS vowel scores, as SoulX phones (as `in_a` akaram in the repo's exporter).

   *Reason:* this is how long aakars get into the training data, and neither ASR nor web
   lyrics provide them.

## 6. Phone-level refinement (S11): strictly additive

- Search only **inside** syllable spans: syllable and note boundaries never move.
- The phone **sequence** is fixed by G2P and the lexicon: the aligner can't choose variants.
- Stored as an optional `phone_spans` layer with confidence. Each syllable falls back to
  the rule-based split (consonants short at the edges, vowel fills the rest) if any check
  fails:
  - minimum phone durations;
  - the vowel covers the voiced part;
  - unvoiced consonants sit on unvoiced or low-energy frames;
  - phones stay in order.
- Enabled dataset-wide **only if it beats the rule-based baseline on the gold set**.

Worst case is today's syllable-level quality. The SoulX export doesn't consume phone
spans at all (SoulX spreads a note's phones itself).

**Tool:** Montreal Forced Aligner, trained on our own corpus with the Indic phone set.
- Install with a micromamba static binary (wget, no conda needed).
- Training runs in a CPU Slurm job.
- Fallback option: bootstrap the repo's `annotator_model` phone head.

## 7. Singers (S5)

- **No separation of overlapping singers.** We keep single-voice stretches only.
- *Reason:* Bollywood duets mostly alternate, and separated overlaps carry artifacts the
  SVS model would learn.
- Singer ids:
  1. clusters within the song;
  2. linked across songs by embedding similarity plus LRCLIB/oEmbed artist names;
  3. a small human check of the largest clusters.
- Male/female separation of overlaps is an optional extension, only if yield is too low.

## 8. Confidence and acceptance (S12)

**Per-clip score** combines:
- ASR–ASR agreement;
- ASR–reference agreement;
- reference version coverage;
- MMS average score per frame;
- F0 ensemble agreement;
- single-voice share (must be 1.0);
- share of voiced frames with no label;
- identity status.

**Calibration (human, step C3):** fit thresholds on the gold set plus a reviewed sample
so that **auto-accepted clips have ≥95% word accuracy and syllable onsets within 50 ms
for ≥90% of syllables**.

| Band | Rule | Expected share |
|---|---|---|
| accept | high score | about 60–70% |
| review | medium score | sampled only (§9) |
| reject | low score | dropped |

Exports include only accepted clips (plus reviewed ones).

## 9. Human inputs (kept to a minimum)

| # | When | What | Where | Effort |
|---|---|---|---|---|
| H1 | before A | approve this plan; confirm rights/permission to train on these recordings | — | 10 min |
| H2 | anytime | optional: `rsync` `music_dataset/` to your PC as a backup | PC | 15 min |
| H3 | C2 | **gold set**: correct 20 clips of 60–90 s (words, syllable onsets, aakar spans, singer turns) | PC, Sonic Visualiser (layers pre-filled by the pipeline, so it's correction, not annotation from scratch) | about 6–8 h total |
| H4 | C3 | look at the calibration report; accept thresholds | — | 20 min |
| H5 | D3 | review queue: a **stratified sample of about 60 medium clips** (not all), accept or reject per clip | PC, review pack or web page | about 2–3 h |
| H6 | D3 | name the largest singer clusters (about 20–40 clusters, listen to 1 clip each) | PC | about 1 h |
| H7 | D3 | the `unidentified` songs list (expected under 5%): label or drop | PC | about 30 min |
| H8 | E | listen to 20 random exported clips; final go | PC | 30 min |

Total: about 11–14 h of human time, most of it in H3 (gold), which is what makes
everything else automatic and trustworthy.

## 10. Execution phases, steps and time estimates

Times: "Claude" means agent working time in sessions; "GPU" means job runtime; queue wait
has so far been seconds to 10 min.

### Phase A: setup and hygiene (about 1.5 h, Claude)
| Step | Where | Estimate |
|---|---|---|
| A1 housekeeping: delete `wheels/` and the broken zip; create `work/`, `export/` | login | 5 min |
| A2 git branch `p1-annotation`, commit the current work | login | 10 min |
| A3 `P1/ops/`: job templates (`gpu_worker.sbatch`, `cpu_job.sbatch`), `jobs.tsv`, `fetch_*.sh` (curl + jq) | login | 30 min |
| A4 download the extra models on the login node: ECAPA speaker model; `fpcalc` static binary; `g2p_en` data; micromamba + MFA env | login | 30 min |

### Phase B: implementation (about 14–20 h, Claude; each step ends with tests in a `debug` CPU job)
| Step | What | Estimate |
|---|---|---|
| B1 | stage/state framework (`stages.json`, atomic IO, config hash), worker loop with claims, `p1-status` | 2–3 h |
| B2 | S0 inventory, S1 fetch scripts (login), S2 dedupe | 1.5 h |
| B3 | S3 dual separation to FLAC; S4 F0 with polyphony features | 1.5 h |
| B4 | S5 single-voice mask, ECAPA embeddings, clustering | 2–3 h |
| B5 | S6 dual ASR (Qwen batch + Whisper), phrase plan from the single-voice mask | 1 h |
| B6 | S7 identify (two-phase fetch), S8 reconcile (Smith–Waterman, word rules, English tagging + `g2p_en`) | 3–4 h |
| B7 | S9 emissions cache, acoustic choice, optional-word alignment | 2 h |
| B8 | S10 vocables, S12 confidence, clip plan; S13 export + DataProcessor check | 2–3 h |
| B9 | S11 phone refinement with MFA (gated; can come after the pilot) | 2–3 h |

### Phase C: pilot and calibration (about 1 day of wall time; GPU about 1 h)
| Step | What | Where | Estimate |
|---|---|---|---|
| C1 | stratified pilot: 40 songs (decades 1980s–2020s; ballad, duet, rap/Hinglish, classical-leaning, aakar-heavy, chorus), full S0–S13 | GPU workers ×2 | about 1 h GPU |
| C2 | gold pack: 20 clips with pre-filled layers → **H3** | CPU job → PC | Claude 1 h + human 6–8 h |
| C3 | evaluation: WER, syllable onset error, vocable precision, single-voice precision, phone refinement vs baseline; threshold fitting → **H4** | CPU job | 1 h |
| C4 | fix what the pilot reveals; rerun only the affected stages (state framework) | — | 2–4 h |

### Phase D: full run (about 1 day of wall time)
| Step | What | Where | Estimate |
|---|---|---|---|
| D1 | S0–S2 on all songs; fetch S1 on login (about 1,100 LRCLIB plus oEmbed calls at 1/s, about 40 min) | CPU job + login | about 1 h |
| D2 | GPU stages S3–S6, S9 over unique songs (about 90 s/song, about 25 GPU-h) with **3 a40 workers + 1 l40 worker**, `--time=04:00:00`, resubmitted as they finish | GPU | about 7–9 h wall |
| D3 | S7 two-phase identify, S8, S10–S12 (CPU, seconds per song) → review sample **H5**, singers **H6**, unidentified **H7** | CPU + login + PC | 1 h compute + human |
| D4 | rerun S12 with final thresholds; QC report | CPU | 30 min |

### Phase E: export and handover (about 2 h)
| Step | What | Where |
|---|---|---|
| E1 | S13 export, splits, DataProcessor check, `DATASHEET.md` (sources, filtering, stats, known issues) | CPU job |
| E2 | `rsync` export + reports to your PC; **H8** | PC |
| E3 | final PROGRESS.md entry; optionally delete intermediates that are no longer needed (keep caches needed to re-export) | login |

**Overall:** about 3–4 Claude sessions of implementation, about 30 GPU-hours, about
11–14 h of human time, roughly **1–1.5 weeks** of calendar time, depending mostly on H3.

## 11. Good-citizen rules for this plan

- Every job: `-p X --qos=X`, 1 GPU (or none), `--cpus-per-task=8`, `--mem=48G`, and
  `--time` about 1.2× expected.
- No `--exclusive`; no idle `srun` shells.
- At most 4 concurrent GPU jobs in total, even though QoS allows more.
- Before D2, check `sinfo -s` and the queue; tell the user.
- If a partition refuses (group limit), wait or use another partition. Don't retry in loops.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Old (1980s–90s) mono and low-fidelity recordings: ASR and separation degrade | Stratified pilot measures it per decade; confidence gating drops what fails |
| Many songs are duets or chorus, which cuts single-voice yield | Measured in the pilot; male/female separation is the optional extension |
| LRCLIB coverage for older songs is thin | ASR-led design works without a reference (lower confidence, still usable after calibration) |
| Losing the only copy of the export | rsync exports to the PC after each phase |
| Session limits | `PROGRESS.md` plus the state framework; every step is resumable |
| Copyright | H1: confirm usage rights; lyrics kept internal, not redistributed |

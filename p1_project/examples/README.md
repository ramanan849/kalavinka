# P1 example runs (4 songs, complete pipeline)

Chosen to show different cases:

| Song | Case | Accepted clips |
|---|---|---|
| 2021 Aashiquii Kaa Gum | clean: reference lyrics found (coverage 0.92) | 9 |
| 2022 Kesariya | the YouTube upload is an edited version, so the reference lyrics are excluded | 8 |
| 2022 "Naacho Naacho (Hindi)" | mislabeled file, identified as **Chammak Challo** | 2 |
| 2022 Jo Bhi Kasmein Khai Thi Humne | unidentified, ASR-only lyrics | 4 |

Per song folder:
- `inventory.json`: format and length
- `identity.json`: which lyrics it matched
- `asr.json`: Qwen3 and Whisper transcripts per phrase
- `lyrics.final.json`: reconciled words with their source label
- `syllables.json`: aligned syllables, vocables and melisma (50 frames/s)
- `singers.json`, `voices.json`: singer turns and clusters
- `clips.json`: planned clips with features and rejection reasons
- `annotation.json`: notes and F0 summary
- `stages.json`: stage versions and timing

`soulx_export/` is the SoulX-Singer-format output for these 4 songs (`--include-review`):
- `wavs/`: 25 accepted and review clips. These are short vocal excerpts; full songs and stems are not committed.
- train/val/test `.jsonl` files, with per-phone text, phonemes and durations
- `excluded.jsonl` and `stats.json`

`report.tsv`: the per-song summary row.

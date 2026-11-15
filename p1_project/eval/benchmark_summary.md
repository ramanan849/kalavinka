## Lyrics transcription (loose-phonetic WER / CER vs LRCLIB)

| system | hi_film_2013_dilliwaali_girlfriend | hi_film_2013_tum_hi_ho | hi_film_2014_abhi_toh_party_shuru_hui_hai | hi_film_2022_kesariya (ref. mismatch) | mean WER (matched refs) |
|---|---|---|---|---|---|
| qwen3-asr-1.7b | 0.30 / 0.12 | 0.24 / 0.08 | 0.35 / 0.22 | 0.66 / 0.46 | **0.30** |
| qwen3-asr-1.7b (mix) | 0.31 / 0.10 | 0.26 / 0.12 | 0.35 / 0.19 | 0.64 / 0.46 | **0.31** |
| whisper-large-v3 | 0.34 / 0.15 | 0.20 / 0.07 | 0.44 / 0.24 | 0.64 / 0.44 | **0.33** |
| whisper-hinglish-prime | 0.48 / 0.30 | 0.50 / 0.38 | 0.33 / 0.17 | 0.93 / 0.83 | **0.44** |

## Line-onset timing vs LRCLIB

Median |error| after removing an offset / after removing offset + tempo drift; share of lines within 0.5 s (drift fit); lines scored.

| method | hi_film_2013_dilliwaali_girlfriend | hi_film_2013_tum_hi_ho | hi_film_2014_abhi_toh_party_shuru_hui_hai | hi_film_2022_kesariya (ref. mismatch) |
|---|---|---|---|---|
| reference text + MMS_FA (oracle) | 0.09 / 0.09 s; 94%; 35/35 | 0.86 / 0.18 s; 78%; 45/45 | 1.50 / 0.12 s; 82%; 39/39 | 32.59 / 30.54 s; 18%; 39/39 |
| qwen3-asr-1.7b: ASR phrase times | 0.80 / 0.81 s; 33%; 30/35 | 0.73 / 0.56 s; 48%; 29/45 | 1.58 / 1.93 s; 29%; 21/39 | 1.47 / 0.76 s; 29%; 17/39 |
| qwen3-asr-1.7b + MMS_FA global | 0.10 / 0.10 s; 93%; 30/35 | 0.74 / 0.12 s; 86%; 29/45 | 1.38 / 0.06 s; 81%; 21/39 | 0.31 / 0.16 s; 53%; 17/39 |
| qwen3-asr-1.7b + MMS_FA phrase | 0.10 / 0.11 s; 93%; 30/35 | 0.74 / 0.12 s; 86%; 29/45 | 1.38 / 0.06 s; 81%; 21/39 | 0.31 / 0.16 s; 53%; 17/39 |
| whisper-hinglish-prime: ASR phrase times | 1.54 / 1.46 s; 35%; 17/35 | 0.63 / 0.56 s; 46%; 22/45 | 1.41 / 0.47 s; 50%; 28/39 | 0.78 / 0.78 s; 33%; 3/39 |
| whisper-hinglish-prime + MMS_FA global | 0.18 / 0.11 s; 65%; 17/35 | 0.61 / 0.18 s; 96%; 22/45 | 1.52 / 0.09 s; 79%; 28/39 | 0.03 / 0.03 s; 67%; 3/39 |
| whisper-hinglish-prime + MMS_FA phrase | 0.20 / 0.11 s; 65%; 17/35 | 0.61 / 0.17 s; 86%; 22/45 | 1.30 / 0.05 s; 96%; 28/39 | 0.03 / 0.03 s; 67%; 3/39 |
| whisper-large-v3: ASR word timestamps | 0.41 / 0.91 s; 13%; 30/35 | 0.57 / 0.32 s; 79%; 33/45 | 2.55 / 0.33 s; 67%; 21/39 | 34.92 / 32.89 s; 20%; 15/39 |
| whisper-large-v3 + MMS_FA global | 0.10 / 0.10 s; 93%; 30/35 | 0.79 / 0.12 s; 85%; 33/45 | 1.83 / 0.09 s; 95%; 21/39 | 35.45 / 35.02 s; 33%; 15/39 |
| whisper-large-v3 + MMS_FA phrase | 0.10 / 0.09 s; 97%; 30/35 | 0.79 / 0.12 s; 85%; 33/45 | 1.83 / 0.09 s; 95%; 21/39 | 35.43 / 34.94 s; 33%; 15/39 |

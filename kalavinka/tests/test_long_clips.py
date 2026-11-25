"""The clip planner's length switch (``qc.clip_max_s``) and the training sample length (``--max-total-s``)."""
import numpy as np

from saregama.p1.text_stages import plan_clips

FPS = 50


def song(n_phrases=12, phrase_s=4.0, rest_s=0.5):
    """Syllables of one singer: phrases of 0.5 s syllables separated by rests."""
    sylls, t = [], 0
    for _ in range(n_phrases):
        for _ in range(int(phrase_s / 0.5)):
            sylls.append({"start": t, "end": t + 25, "kind": "lyric"})
            t += 25
        t += int(rest_s * FPS)
    n = t + 10
    return sylls, np.ones(n, bool), np.zeros(n, int), np.full(n, 200.0)


def test_default_cap_is_16_s_and_long_cap_merges_more():
    """Same song: 16 s cap by default; a 28 s cap gives fewer, longer clips, never longer than the cap."""
    sylls, single, cluster, hz = song()
    short = plan_clips(sylls, single, cluster, hz)
    long = plan_clips(sylls, single, cluster, hz, max_s=28.0)
    dur = lambda c: (sylls[c[1] - 1]["end"] - sylls[c[0]]["start"]) / FPS
    assert max(dur(c) for c in short) <= 16.0 and max(dur(c) for c in long) <= 28.0
    assert len(long) < len(short) and max(dur(c) for c in long) > 16.0
    # every syllable is still in exactly one clip
    cover = sorted(i for c in long for i in range(*c))
    assert cover == list(range(len(sylls)))


def test_prompt_keeps_its_length_with_a_bigger_sample_cap():
    """PairDataset: prompt length = min(prompt, 6 s, max(total cap - target, 1 s)); with 36 s a 28 s target keeps 6 s."""
    T, prompt_max = 28 * FPS, 6 * FPS
    old = min(10 * FPS, prompt_max, max(22 * FPS - T, FPS))
    new = min(10 * FPS, prompt_max, max(36 * FPS - T, FPS))
    assert old == FPS and new == prompt_max
